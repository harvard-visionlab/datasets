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
  index/sources.parquet                  one row per YouTube source: title/description/tags/channel/duration (carrier evidence)
  index/sources_raw/youtube_*.jsonl      verbatim API responses (archive; videos disappear over time)
  splits/<version>.parquet + .report.md  clip_id -> split (whole videos / whole channels, stratified); v3 = train/val/test
  subsets/<name>.parquet + .report.md    a named clip population (person_carried_v0)
  shards/<axis>/group_XXXX/              per-group slipstream shard (resumable unit); axis = <res> or <res>-<fps>fps
  shards/<axis>/group_XXXX.claim         fleet mode: which host owns the group
  stores/spatialvid-hq-h265-<axis>/      final slipstream cache + records.parquet + store_manifest.json
  logs/encode_<host>_<axis>.log          fleet logs (status.py reads them)
```

**Store axes.** `res` ∈ {640x360, 456x256} × `fps` ∈ {native, 30, 15}. An fps store holds every clip decimated by an
integer factor k (`common.decimation_factor`: k = round(src/F) lowered until src/k ≥ F − 0.1), so 60 fps → 30, 50 → 50,
24 → 24 at F = 30 and 60 → 15, 50 → 16.7, 30 → 15, 24 → 24 at F = 15. Frames are exactly 0, k, 2k, … with timestamps on
the k/src grid (`select=not(mod(n\,k)),setpts=N*k/FRAME_RATE/TB,fps=src/k`; the plain `fps` filter emits frame jk+1 for
k = 3, 4 — measured). `fps`/`num_frames`/`duration_s` describe the stored video; `src_fps`/`src_num_frames` the source, and
`annot_frame_idx` keeps *source* frame indices (annotation time = `annot_frame_idx / src_fps`). Decode cost per window
follows the stored fps (60 → 30 fps halves it); bytes are ~80 % of native, not 50 %, because x265's CRF scales with the
stream frame rate.

## Stages

| # | command | reads | writes | full-HQ cost (machina) |
| - | --- | --- | --- | --- |
| 1 | `python -m datasets.prep.spatialvid_hq.build_index --raw R --out O` | CSV, SpatialVID-RAW source CSV, 74 annotation tars | `index/` | ~10 min (gzip-bound, 8 procs) |
| 1b | `python -m datasets.prep.spatialvid_hq.fetch_sources --out O [--api-key K]` | `index/clips.parquet`, YouTube Data API v3 (`videos.list`, ~450 quota units; `--backend ytdlp` fallback) | `index/sources_raw/*.jsonl` (verbatim archive), `index/sources.parquet` (title, description, tags, channel, category, duration, stats per source) | minutes; resumable |
| 2b | `python -m datasets.prep.spatialvid_hq.make_subset --out O [--name person_carried_v0 --carriers walk,rig --max-speed 0.5]` | `index/channels.parquet`, `index/sources.parquet`, `index/clips.parquet` | `index/carrier_v1.parquet` (clip → carrier + rule), `subsets/<name>.parquet` + `.report.md` | seconds |
| 2 | `python -m datasets.prep.spatialvid_hq.make_splits --out O --version v3 --subset subsets/person_carried_v0.parquet --dims scene_l1,tod,weather_c,crowd,motion,carrier --val-unit three --test-clips 11000 --val-clips 15000 --max-source-clips 200 --max-unit-frac 0.015` | `index/`, subset | `splits/v3.*` (train / val / test) | seconds |
| 3 | `python -m datasets.prep.spatialvid_hq.encode --raw R --out O [--fps 30] [--groups 1-74] [--claim] [--limit N] [--ffmpeg BIN]` | video tars + `index/` | `shards/<axis>/group_XXXX/` | native both res ~44 h on machina; fps passes on the 5-host fleet ≈ 10 h (30) + 8 h (15); resumable per group |
| 3f | `python -m datasets.prep.spatialvid_hq.fleet launch --fps 30` / `launch --fps 15 --after-fps 30` / `finish --fps-list 30,15` / `status` / `tail` / `ps` / `stop` / `sync` | ssh + `docker exec` on machina, thrace, vesper, leeloo, stelline | per-host `encode --groups all --claim`; `finish.py` merges + S3-syncs each axis when its 74 shards exist | run from any machine with ssh to the hosts |
| 4 | `python -m datasets.prep.spatialvid_hq.merge --out O [--fps 30]` | shards | `stores/` | minutes (sequential copy); field types come from the shard manifests |

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
| `width`, `height`, `fps`, `num_frames`, `duration_s` | int, int, float, int, float | of the stored video (native stores: frame count equals the source; fps stores: `ceil(src_num_frames / k)`; asserted at encode) |
| `src_fps`, `src_num_frames` | float, int | of the source clip (stores built from 2026-09-17; absent in the native stores built earlier, where they equal `fps` / `num_frames`) |
| `src_start_us`, `src_end_us` | int | clip position inside the source recording (µs), from SpatialVID-RAW |
| `n_annot` | int | number of annotated frames (stride `int(fps/5)`, ≈5–6 Hz) |
| `annot_frame_idx` | bytes → int32 `(n_annot,)` | **source** frame index of each annotation row (time = `annot_frame_idx / src_fps`; in fps stores stored frame `j` is source frame `j·k`) |
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
11 for 64), and in 11 cases `poses` and `intrinsics` also disagree with each other (e.g. 26 vs 12); in 37 of the 66 only 1–6 rows are missing. This looks like the
authors' MegaSaM reconstruction covering only part of the clip; the excluded clips are shorter than average
(median 8.5 s vs 14.3 s). Because it is unknowable which frames the surviving rows refer to, no repair is
attempted. `caption.json`/`instructions.json` of these clips are still in the index for completeness.

## Known decoder caveat (torchcodec 0.16, NVDEC)

With `device="cuda"`, torchcodec raises *"Requested next frame while there are no more frames left to decode"*
when one of the last 1–2 frames of a clip is requested, for about 6 % of the HEVC clips (9/150 for the final
frame, 2/150 for the last annotated frame; interior frames never fail; the CPU decoder always succeeds and its
exact frame count equals the stored `num_frames` on 300/300 sampled records). `VideoStore` retries such requests
on a CPU decoder. A window sampler should treat the final two frames of a clip as invalid anchors/targets on CUDA,
or use the same fallback.
