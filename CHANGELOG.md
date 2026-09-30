# Changelog

All notable changes to this project are documented here.

## [0.15.0] - 2026-09-30

### Added
- `sync --source DIR`: copy caches from a local/NFS copy (e.g. the lab_storage master on FASRC) with
  parallel ranged `pread`/`pwrite` (`--readers`, default 16) into the staging dir, under the same
  lock / verify / manifest-last rules. The source copy must pass its own integrity check; otherwise
  that cache falls back to S3.
- `status`: in a group-writable cache dir, flags caches with items lacking group write (`group_writable`
  in `--json`) and prints the owner's `chmod -R g+w` fix.

### Changed
- `sync --concurrency` defaults to `auto`: 16 parts per file for files >= 256 MB, 1 below (was 1 for all;
  one 3.8 GB object from S3 went 18 -> 54 MB/s on a Mac, and FASRC saw ~nothing -> 15 MB/s).
  Pass `--concurrency 1` when writing to NFS that collapses under parallel part writes.

### Fixed
- Group-writable bases (e.g. setgid 2775 lab dirs): the lock file, the `SYNC_INCOMPLETE.json` marker and s5cmd's
  output (run with umask 002) are now group-writable too, regardless of the caller's umask 022, so
  other members can break a stale lock and resume or repair a sync. A cache dir owned by someone else
  without group write now fails with the owner's `chmod` command instead of a traceback.

## [0.14.0] - 2026-09-30

### Added
- **Safe sync onto shared cache dirs** (`visionlab.datasets.sync`, used by `visionlab-datasets sync`
  and by `load()` / video-store download when a cache is missing):
  - per-cache lock `.<cache>.sync.lock` (O_EXCL; owner/host/pid/start; 60 s heartbeat; stale when
    the process is gone or the heartbeat is > 15 min old, then broken automatically; `--break-lock`);
    a live lock makes `sync` exit 1 untouched, and `load()` waits for the other sync instead.
  - download into `.<cache>.sync.partial/`, verify (size; sha256 when the manifest has `file_sha256`),
    rename into place with `manifest.json` last. Nothing in the cache is ever deleted. On failure the
    staging dir is kept with `SYNC_INCOMPLETE.json` and the next sync resumes from it.
  - repair: only missing / wrong-size files are fetched (`--deep`: also right-size files whose sha256
    mismatches). A remote manifest that differs from the local one (cache rebuilt) is refused without `--force`.
  - group-writable files in group-writable cache dirs; progress and MB/s per sync.
- `status --deep` (sha256 against `file_sha256`; `unavailable` for manifests without hashes; mismatch
  is a problem, exit 1). `status --json` rows gain `sync_lock`, `sync_partial`, `deep_status`,
  `deep_problems`; a live sync shows as `syncing`.

### Changed
- `sync` no longer calls slipstream's `download_s3_cache` (which copied straight into the cache dir);
  it drives `s5cmd run` itself. `load()` for image and video stores uses the same path.
- Dev pin and README examples moved to slipstream v0.9.5 (0.9.3: decoder always relinked; 0.9.4: loader prefers this interpreter's decoder build; 0.9.5: fixes a 10-20x SlipstreamLoader threading slowdown under numba 0.67).
  Consumers of 0.13.1 can already use it by pinning `@v0.9.5`.
- README: netscratch guidance (sync repairs purged caches) replaces "do not use netscratch".

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
