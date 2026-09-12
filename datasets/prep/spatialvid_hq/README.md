# SpatialVID-HQ → slipstream video stores

Reproducible pipeline from the raw Hugging Face mirror (downloaded by `datasets/downloads/spatialvid_hq.py`)
to slipstream stores, an index and versioned splits. Decisions and measurements behind every setting are in
`slipstream/experiments/spatialvid_inspection/DECISIONS.md`.

```
<raw>  (read-only HF mirror, e.g. .../DataSets/VideoDatasets/SpatialVID-HQ)
  data/train/SpatialVID_HQ_metadata.csv   videos/group_XXXX.tar.gz   annotations/group_XXXX.tar.gz

<out>  (working tree on local NVMe)
  index/clips.parquet                    one row per clip: metadata + source_id + timestamps + n_annot/annot_ok
  index/annotations/group_XXXX.parquet   per-clip poses/intrinsics/frame indices (flat lists), caption, instructions
  splits/<version>.parquet + .report.md  clip_id -> split (source-level, stratified)
  shards/<res>/group_XXXX/               per-group slipstream shard (resumable unit)
  stores/spatialvid-hq-h265-<res>/       final slipstream cache + records.parquet + store_manifest.json
```

## Stages

| # | command | reads | writes | full-HQ cost (machina) |
| - | --- | --- | --- | --- |
| 1 | `python -m datasets.prep.spatialvid_hq.build_index --raw R --out O` | CSV, SpatialVID-RAW source CSV, 74 annotation tars | `index/` | ~10 min (gzip-bound, 8 procs) |
| 2 | `python -m datasets.prep.spatialvid_hq.make_splits --out O --version v1 --val-clips 12000` | `index/clips.parquet` | `splits/v1.*` | seconds |
| 3 | `python -m datasets.prep.spatialvid_hq.encode --raw R --out O [--groups 1-74] [--limit N] [--ffmpeg BIN]` | video tars + `index/` | `shards/<res>/group_XXXX/` | ~44 h both resolutions (x265 medium, 16×4 threads); resumable per group |
| 4 | `python -m datasets.prep.spatialvid_hq.merge --out O` | shards | `stores/` | minutes (sequential copy) |

Run with `uv run --group video python -m ...` from the repo root.

**System requirement (no conda):** FFmpeg with libx265 on `PATH`, and its shared libraries for torchcodec.
On the lab's Ubuntu 22.04 containers `apt-get install -y ffmpeg` (4.4.2: libx265, hevc_nvenc, hevc_cuvid, cuda
hwaccel) covers both; add it to the container image. NVIDIA NPP for torchcodec's CUDA path comes from pip
(`nvidia-npp-cu12`, in the `video` group) and is preloaded by `visionlab.datasets.video`. macOS: `brew install
ffmpeg` and `export DYLD_LIBRARY_PATH=/opt/homebrew/lib`. Any FFmpeg 4.4–7 works (`-vsync`/`-fps_mode` chosen
by version).

## Data model

One **record = one clip** (2–15 s, one source recording segment). Splits and experiment subsets are index sets
over records (`records.parquet`: `record_idx ↔ clip_id`), never separate stores. Fields (`common.FIELD_TYPES`):

| field | type | contents |
| --- | --- | --- |
| `video` | bytes | MP4, HEVC Main yuv420p, `hvc1`, faststart, no audio, **keyframe every 1 s**, x265 crf 29; 640×360 or 456×256 |
| `clip_id`, `source_id`, `group_id` | str, str, int | SpatialVID clip uuid; YouTube id of the source recording; HF packaging group |
| `width`, `height`, `fps`, `num_frames`, `duration_s` | int, int, float, int, float | of the stored video (frame count equals the source; asserted at encode) |
| `src_start_us`, `src_end_us` | int | clip position inside the source recording (µs), from SpatialVID-RAW |
| `n_annot` | int | number of annotated frames (stride `int(fps/5)`, ≈5–6 Hz) |
| `annot_frame_idx` | bytes → int32 `(n_annot,)` | video frame index of each annotation row |
| `poses` | bytes → float32 `(n_annot, 7)` | `[tx ty tz qx qy qz qw]`, **world→camera**, OpenCV axes (x right, y down, z forward); scale is not metric |
| `intrinsics` | bytes → float32 `(n_annot, 4)` | normalized `[fx fy cx cy]`; pixel values = `fx·width, fy·height, cx·width, cy·height` |
| `instructions`, `caption` | str (JSON) | authors' merged motion-instruction spans; structured caption (scene/camera text, tags, motion trends) |
| `scene_type`, `motion_tags`, `brightness`, `time_of_day`, `weather`, `crowd_density` | str | metadata CSV labels |
| `aesthetic_score` … `dynamic_ratio`, `dist_level` | float / int | metadata CSV scores |

Array fields are `np.save` blobs (ragged `n_annot`); `visionlab.datasets.video.VideoStore` decodes them. Pixel-space
annotations (dynamic masks, depth) are not stored; they live at 720p in the raw archives.

## What a training sample looks like

Storage unit ≠ sample unit. A sample is a **window spec** `(record_idx, start_frame, T, stride)`; the loader
decodes `[T, 3, H, W]` uint8 from the record's bytes (torchcodec, ≤1 s of extra decode thanks to the GOP) and
attaches `[T, 7]` poses interpolated to those frames (`VideoStore.poses_at`) plus derived relative motion
`[T-1, 6]` (`visionlab.datasets.video.relative_motion`, camera-t axes). T=2 with stride `int(fps/5)` reproduces the
native annotated pairs; any other rate uses interpolated poses. Anchors are valid when `start + (T-1)·stride <
num_frames`. Until slipstream has a native window sampler, `notebooks/spatialvid_hq_preview.ipynb` shows this
with `VideoStore`.

## Exclusions (annotation integrity)

`build_index` marks a clip `annot_ok = False` when `poses.npy`, `intrinsics.npy` or `indexes.txt` is missing or
unreadable, or when their row counts disagree (`len(poses) == len(intrinsics) == len(indexes)` is required, since
row *i* of each must describe the same video frame). `make_splits` puts such clips in `split = "excluded"` and
`encode` skips them, so they are absent from every store.

**HQ v1 result: 66 of 365,362 clips (0.018 %) excluded, spread over 41 groups**
(`excluded_clips_v1.csv`, regenerated as `<out>/index/excluded_clips.csv`). All 66 have all six annotation files
present and parseable; the download itself was size-verified against the Hub. The defect is in the released
annotations: `poses.npy`/`intrinsics.npy` have fewer rows than `indexes.txt` (e.g. 70 poses for 76 indexed frames,
11 for 64), and in 5 cases `poses` and `intrinsics` disagree with each other (e.g. 26 vs 12). This looks like the
authors' MegaSaM reconstruction covering only part of the clip; the excluded clips are shorter than average
(median 8.5 s vs 14.3 s). Because it is unknowable which frames the surviving rows refer to, no repair is
attempted. `caption.json`/`instructions.json` of these clips are still in the index for completeness.
