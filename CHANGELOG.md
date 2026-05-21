# Changelog

All notable changes to this project are documented here.

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
