# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Changed
- Dev pin and README examples moved to slipstream v0.9.3 (build-only: always relinks the decoder).
  Consumers of 0.13.1 can already use it by pinning `@v0.9.3`.

## [0.13.1] - 2026-09-30

### Fixed
- uv consumers can now pick any in-range slipstream tag. 0.13.0 pinned slipstream v0.9.2 in
  `[tool.uv.sources]`, which uv applies to git dependents, so a consumer pinning another tag failed
  with "conflicting URLs". datasets' own pin moved to the `dev` dependency group (lock unchanged: v0.9.2).
- README: uv install example uses `[tool.uv.sources]` and notes consumers must declare the torch indexes.

## [0.13.0] - 2026-09-30

### Changed
- **Install change: slipstream is declared as a range (`visionlab-slipstream>=0.9,<0.10`), not a git tag.**
  Consumers must now pin slipstream themselves (any in-range tag, e.g. `@v0.9.2`), since it isn't on
  PyPI; an install without that pin fails to resolve, and an out-of-range tag fails loudly. slipstream
  patch releases no longer need a datasets release. See README "Installation".
- datasets' own dev env/lock still pins slipstream v0.9.2 via `[tool.uv.sources]`. No API or stream changes.

## [0.12.1] - 2026-09-30

### Changed
- slipstream pinned to 0.9.2. Seeded streams unchanged vs 0.9.0; unseeded random decoders now draw fresh
  entropy (0.9.1); faster lazy `import slipstream`; `slipstream status` reports the decoder build.
- **Install requirement:** slipstream's build now fails (previously warned) when the libslipstream
  decoder can't be built, so TurboJPEG (libturbojpeg + `turbojpeg.h`) must be installed first, or set
  `TURBOJPEG_ROOT` / `SLIPSTREAM_SKIP_EXT=1`. See README "Installation".

## [0.12.0] - 2026-09-29

### Changed
- **Breaking (reproducibility): slipstream pinned to 0.9.0.** Seeded augmentation/shuffle streams
  differ from both 0.8.0 and 0.7.x for the same seed: every stream is keyed by `(seed, rank, epoch)`,
  the loader reseeds decoders and seeded transforms each epoch (exact resume), DDP ranks draw
  different augmentations, and per-view seeds are hashed (`slipstream.derive_seed`) instead of added.
  Stay on datasets 0.11.0 (slipstream 0.8.0) or 0.10.0 (slipstream 0.7.1) to keep older streams.
  No datasets API changes.

## [0.11.0] - 2026-09-29

### Changed
- **Breaking (reproducibility): slipstream pinned to 0.8.0**, which derives every seeded stream via a
  hashed `SeedSequence`. Sample orders, crops and other seeded augmentations differ from slipstream
  0.7.x for the same seed. Projects that must reproduce 0.7.x streams should stay on datasets 0.10.0
  (slipstream 0.7.1). No datasets API changes; 0.7.1's fixes are included.

## [0.10.0] - 2026-09-29

### Added
- Video registry: `load('spatialvid-hq')` -> `VideoDataset` (stores per format/resolution/fps, split and
  subset parquets, `fps="native"`, population subset applied by default, batched `poses_at` /
  `ego_motion_at`, `cache_path` so `SlipstreamLoader(ds, ...)` works directly, `ego_motion_transform()`).
- SpatialVID-HQ prep: fps as a store axis, fleet encode (`fleet.py`), `--retry-failed` patch pass,
  carrier review / `make_subset`, three-way `make_splits`; per-store RGB normalisation stats; dataset card.
- `cli list` shows video stores / splits / subsets.

### Changed
- slipstream pinned to 0.7.1 (0.7.0: `DecodeVideoWindow`, `sample_data`, array fields; 0.7.1: RandomRotate
  under bf16, `DecodeMultiResizeCropEmbed` with yuv420, `set_epoch` resets the embed decoder's crop counter).
  Seeded streams unchanged vs 0.7.0.

### Fixed
- `relative_motion` / `quat_to_rotmat` compute in float64 (float32 arccos error ~1e-3 rad on small rotations).

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
