# Changelog

All notable changes to this project are documented here.

## [0.9.0] - 2026-09-12

### Added
- **SpatialVID-HQ preparation pipeline** (`datasets/prep/spatialvid_hq/`): `build_index` (metadata +
  SpatialVID-RAW source ids + per-clip annotations), `make_splits` (source-level stratified train/val),
  `encode` (640x360 and 456x256 HEVC, keyframe every 1 s, resumable per-group slipstream shards),
  `merge` (shards -> one slipstream store per resolution). Data model in that directory's README.
- `datasets/downloads/spatialvid_hq.py`: resumable, verified download of the raw HF release.
- `visionlab.datasets.video.VideoStore`: reads h265 video stores (torchcodec window decode from the
  record bytes, pose interpolation, relative camera motion). New `video` dependency group.
- `notebooks/spatialvid_hq_preview.ipynb`: record contents, frames, trajectory, window samples, loader batch.

### Changed
- slipstream pinned to 0.6.0 (indices-aware `warmup_cache`, bank-eligible `bytes` fields, owned
  `{data, sizes}` payloads for secondary bytes fields) — required for video stores.
- Lock resolves torch 2.14 on all platforms (torchcodec 0.16 needs torch >= 2.11).

## [0.7.0] - 2026-05-21

### Changed
- **Normalization stats are now keyed by decoded colorspace, not storage
  format.** The `metadata["stats"]` dict in every dataset config now uses
  `"rgb"` (decoded RGB output — the common decode path) and `"yuv420"` (raw
  YUV planes — only for niche `DecodeYUV*` consumers) keys, replacing the old
  storage-format `"jpeg"`/`"yuv420"` keys. Stat *values* are unchanged.
- `load(...)` now exposes `.stats_by_colorspace` (all colorspaces) and defaults
  `.stats` to the rgb stats regardless of the storage `fmt`.

### Fixed
- `load(fmt='yuv420').stats` previously returned raw-YUV stats, which
  mis-normalized RGB pixels (G/B channels ~3-4x off) because every `Decode*`
  decoder emits RGB. It now returns the correct rgb stats. Genuine YUV-plane
  consumers (`DecodeYUV*`) must read `.stats_by_colorspace["yuv420"]`.
