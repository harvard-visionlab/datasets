# SpatialVID-HQ: next steps (as of 2026-09-13)

State: raw HF mirror on QNAP Flash (`.../VideoDatasets/SpatialVID-HQ`, 1.2 TB, verified); derived tree
`.../VideoDatasets/SpatialVID-HQ-slipstream/` with `index/clips.parquet` (365,362 clips, source ids, annotation
integrity flags), `splits/v1.parquet` (12,012 val, 66 excluded), and two verified stores
`stores/spatialvid-hq-h265-{640x360,456x256}` (314.8 / 178.5 GB, 365,296 records each). S3 sync to
`s3://visionlab-datasets/slipstream-cache/spatialvid-hq/` started 2026-09-13 (check with `s5cmd ls`).
Background and every measured decision: `slipstream/experiments/spatialvid_inspection/DECISIONS.md`;
data model: `datasets/prep/spatialvid_hq/README.md`; reader: `visionlab.datasets.video.VideoStore`.

## 1. Registry integration (`load("spatialvid-hq", split="val", fmt="h265", res="640x360")`)
- Add a `res` axis to `DatasetConfig.remote_cache` keys `(split, fmt, res)` with `res=None` for image datasets and
  a per-config default; accept aliases (`"640"`, `"360p"`).
- Video datasets have **one store per (fmt, res)**, splits are index sets: config lists `stores[(fmt,res)] → s3 path`
  and `splits[version] → s3 parquet`; `load(...)` returns a `VideoStore`-backed dataset plus `indices` for the split
  (and later a `subset`), and `SlipstreamLoader(..., indices=)` uses them. Also resolve a QNAP path when present
  (machina/workers) instead of downloading 315 GB into `SLIPSTREAM_CACHE_DIR`.
- Normalization stats: compute per store (`compute_normalization_stats` over decoded frames), key `"rgb"`.

## 2. Dataset cards (`datasets/cards/<name>.md`, one per dataset, methods-ready)
Template: source + license + citation; what a record is; counts (clips, hours, sources, frames); preprocessing
(codec, crf 29, keyframe 1 s, resolutions, frame-count assertion); annotations kept/dropped and their conventions
(w2c OpenCV, normalized intrinsics, ≈5 Hz, non-metric scale); **exclusions** (criterion, 66 clips, list);
**splits** (unit = YouTube source id → no recording straddles splits; greedy source-level stratification on scene,
time of day, weather, crowd, motion class, duration; val 12,012 clips / 1,126 sources, max source 0.3 % of val;
report file); known caveats (NVDEC last-frame bug, captions noisy); versions (store build date, split version,
slipstream/visionlab-datasets versions). Generate the numeric parts from `index/`, `splits/*.report.md`,
`store_manifest.json` so the card cannot drift. Write `spatialvid-hq.md` first, then cards for the image datasets.

## 3. Subset = "human walking POV" (the actual training set)
Metadata has no carrier label; caption keywords reach ~26 % confident walk / 44 % unknown; motion stats do not
separate walk from drive (DECISIONS.md §3). Two signals not yet used:
- **YouTube source metadata.** `index/clips.parquet` has the `source_id` for every clip (22,543 videos). Titles,
  descriptions, tags and channel names ("4K walking tour", "dashcam", "drone", "train ride", "POV bike") label
  *whole recordings* at once, which is exactly the split unit. Fetch via the YouTube Data API (`videos.list`, 50
  ids/call ≈ 450 quota units total) or `yt-dlp --dump-json --skip-download`; store in `index/sources.parquet`.
  The SpatialVID-RAW `metadata_long_duration.csv` has the same fields as the short one (no carrier), skip it.
- **VLM on frames** (3 frames + caption per clip, or per source) to validate/refine; hand-label 300 clips first
  (contact-sheet workflow in `slipstream/experiments/spatialvid_video_test/`).
Deliverables: `index/carrier_v1.parquet` (clip_id → carrier, confidence, evidence), `subsets/walk_v1.parquet`
(record indices per store), split v2 with carrier as a stratification dimension. slipstream side: a `subset`
concept is just `indices=`; what's missing is a **named subset registry** in visionlab-datasets and an
indices-aware `warmup_cache` (done in 0.6.0).

## 4. slipstream indexes worth building
`OptimizedCache` indexes are per-field value → record indices (used for class subsetting). Useful here:
`scene_l1`, `time_of_day`, `weather`, `crowd_density`, `motion class`, `group_id`, `source_id` (for
leave-source-out eval), and later `carrier`. Cheap (scalar/str fields); build once after the store exists and
sync with the store. Duration/fps buckets can be derived at load from `num_frames`/`fps` arrays.

## 5. Demo dataloader: T frames per clip with interpolated poses
Design (decide, then implement):
- **Sample spec** `(record_idx, frame_idx[T])`; two samplers: `uniform_random` (T sorted random frames from the
  clip) and `window(start, stride)`; both seeded per (seed, epoch, sample) so a given seed reproduces the exact
  frame sets; exclude the last 2 frames on CUDA.
- **Decode**: one torchcodec decoder per record per batch, `get_frames_at(indices)`; CPU fallback; output
  `[B,T,3,H,W]` uint8; poses interpolated to `frame_idx` (`interpolate_poses`) → `[B,T,7]`, relative motion
  `[B,T-1,6]`, plus `t_sec`, `clip_id`, `source_id`.
- **Where**: prototype in this repo (`datasets/video.py`: `ClipFrameSampler` + `spatialvid_collate`, torch
  `DataLoader` with workers over `VideoStore`, or slipstream's loader for bytes + a decode stage in
  `after_batch_transforms`); then move into slipstream as a `VideoDecode` pipeline stage + `SequenceSampler` so
  the seeded-reproducibility contract matches the image pipelines (per-sample seeding already exists there).
- Acceptance: notebook shows a `[B,16,3,360,640]` batch with trajectories; two runs with the same seed give
  identical frame indices; throughput ≥ training demand on machina (CPU 48 workers ≈ 775 8-frame windows/s measured).

## 6. Housekeeping
- Container image: add `apt-get install -y ffmpeg`; document `UV_PROJECT_ENVIRONMENT` for root containers.
- Upstream issue to torchcodec for the NVDEC last-frame failure (repro: any store clip, `get_frame_at(n-1)`).
- Version bump visionlab-datasets when the registry axis lands; update `README.md` usage.
