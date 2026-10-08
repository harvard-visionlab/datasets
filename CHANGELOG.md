# Changelog

All notable changes to this project are documented here.

## [0.18.0] - 2026-10-08

### Added
- Datasets `procthor-walks-objects` and `procthor-walks-empty`: 24,000 paired 1,000-step Brownian walks each through
  ProcTHOR-10k houses (furnished / same houses emptied), rendered in AI2-THOR 5.0.0 at 160x120 by Rupert
  Tawiah-Quashie (Harvard Vision Lab). Stores: h264 yuv444p crf 10, GOP 60, one record per walk (video + positions +
  headings), one store per source folder (objects 46.5 / 4.5 / 4.5 GB, empty 35.8 / 3.5 / 3.5 GB). Splits `v1`
  (ProcTHOR house split) and `rtq-r160` (the student's r160 runs). Card: `datasets/cards/procthor-walks.md`; demo
  notebooks `notebooks/datasets/procthor_walks_{objects,empty}.ipynb`; reproducible build
  `datasets/prep/procthor_walks/` (index, encode, verify, publish, card_figure).
- Video configs may map a store key to `{part: S3 path}`; `load()` picks the part holding the requested split
  (`video_dataset.store_part`).
- `visionlab.datasets.demo`: `download_plan` (store, S3 size, local path, before downloading) and
  `remove_local_copy` (refuses shared lab copies).
- `procthor` dependency group and ProcTHOR loader/codec benchmarks in `benchmarks/`.

### Changed
- Dev pin slipstream v0.11.1 (thread-safe `load_batch`); range unchanged (`>=0.9,<0.12`).

## [0.17.1] - 2026-09-30

### Changed
- slipstream range widened to `>=0.9,<0.12` for slipstream 0.11.0 (progressive-resolution training; no
  manifest, cache or CLI changes). Dev pin and README examples: slipstream v0.11.0.

### Added
- `visionlab.datasets.prep.hash_masters`: coordinated `file_sha256` pass for the master caches. `hash-s3`
  streams sha256 from S3; `hash-local` hashes a local master read-only; `compare` cross-checks two sources;
  `publish-s3` backs up and replaces the S3 manifest; `apply-local` writes the published manifest into a
  local master. Manifests are serialised byte-identically to `slipstream hash`.

## [0.17.0] - 2026-09-30

### Changed
- slipstream range widened to `>=0.9,<0.11` for slipstream 0.10.0: manifests carry `file_sha256`,
  `check_integrity(deep=True)` hashes, and `slipstream hash` adds hashes to an existing cache. The loader
  gains `on_invalid_cache` and raises CacheIntegrityError instead of rebuilding or wiping a cache dir
  whose data has no manifest. Dev pin and README examples: slipstream v0.10.0.
- **Upgrade before the master caches get hashes.** datasets 0.14-0.16 compare manifests byte for byte, so
  once the S3/lab_storage manifests gain `file_sha256`, re-syncing an existing (e.g. purged) local copy with
  those versions is refused as "cache rebuilt upstream" (workaround: `--force`). 0.17.0 treats the two as the
  same cache build.
- docs/plans/imagenet1k-val-slipcache-cleanup.md: marked done (lab_storage + S3 duplicates deleted, 10.3 GB each).

### Fixed
- sync no longer treats a locally hashed copy (`slipstream hash`) as a different cache build: manifests are
  compared without `file_sha256` (hashes must agree only when both sides have them). When the source has no
  hashes, the local ones verify fetched files and `--deep` repairs, and the local manifest is kept.

## [0.16.1] - 2026-09-30

### Changed
- `sync --concurrency auto` uses 32 parts (was 16) for files >= 256 MB. FASRC -> S3 measured 31.9 MB/s at 16
  vs 46.0 MB/s at 32.
- `sync --source`: new `--chunk-mb` (default 64) to tune ranged-copy job size.

## [0.16.0] - 2026-09-30

### Changed
- **sync copies only the cache's own files**: manifest.json, the files it names (`file_sizes` and each
  field's storage files), `<field>_index.npy` (slipstream `write_index` output, auto-discovered by the
  loader but not in the manifest), and datasets' video-store sidecars (`records.parquet`,
  `store_manifest.json`). Other files in the source are skipped and reported, e.g. the duplicate
  `slipcache/` copy inside the imagenet1k val caches (7.6 GB -> 3.8 GB for jpeg-val). A manifest
  with neither `fields` nor `file_sizes` keeps everything.

### Added
- `status`: flags files in a local cache that aren't part of it (`unlisted` in `--json`; informational).
- docs/plans/imagenet1k-val-slipcache-cleanup.md: S3 audit results + proposed deletion (not run).

### Fixed
- `status --deep` reports `unavailable` whenever the manifest has no `file_sha256`, even if slipstream's
  `check_integrity(deep=True)` exists (a size-only pass must not read as a passed sha256 check).
  The hash format (`file_sha256` next to `file_sizes`) is slipstream's proposal, pending George's decision.

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
