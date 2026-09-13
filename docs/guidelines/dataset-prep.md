# Dataset preparation: guidelines and lessons (2026-09)

Distilled from the SpatialVID-HQ build (1.07 TB raw video → two slipstream video stores in ~2 days). Keep short.

## Process that worked
1. **Measure before designing.** Every setting (resolution, GOP, codec, frame rate, split size) was decided from a
   small measured experiment on real files, written down with the numbers. Decisions live next to the experiment
   (`slipstream/experiments/<dataset>_inspection/DECISIONS.md`) so nobody re-derives them.
2. **Raw mirror is archival, derived data is disposable.** Download the source release once, size-verify against
   the Hub (`datasets/downloads/*.py`), never extract multi-million-file archives onto a NAS. Everything else is a
   re-runnable pipeline (`datasets/prep/<dataset>/`) with stages: index → splits → encode/extract → merge.
3. **Storage unit ≠ sample unit.** Store per clip/image once; define samples (windows, pairs, crops) in the loader.
   Splits and experiment subsets are **index sets** over one store (`records.parquet`: record_idx ↔ id), so a
   split revision costs a parquet file, not a re-encode. slipstream's page-cache footprint follows the records
   actually read, so a "heavier than needed" store is free at train time (warm it with `indices=`).
4. **Per-group shards + merge** makes long encodes resumable and lets several machines write into one shared
   output tree (one shard dir per group; merge concatenates). Write the shard manifest slipstream's merge expects.
5. **Test run first** (a few dozen clips end to end, plus the preview notebook), then the full run.
6. **Document exclusions** with counts and the exact criterion at the moment you implement them
   (`annot_ok` in SpatialVID: `len(poses)==len(intrinsics)==len(indexes)`), and keep the list in the repo.
7. **Splits:** unit = source recording (YouTube id) so clips from one recording never straddle train/val;
   stratify at source level greedily on the metadata marginals; write a human-readable report next to the parquet;
   one val split only (no test set we would hill-climb on). Version splits (`splits/v1.parquet`).

## Environment rules
- **No conda.** System FFmpeg via the container image (`apt-get install -y ffmpeg`; Ubuntu 22.04 → 4.4.2,
  24.04 → 6.1.1, both with libx265/NVENC/CUVID). torchcodec needs FFmpeg *shared* libs; PyAV bundles its own and
  is fine for CPU-only offline work. Pin torchcodec to the same PyTorch index as torch (PyPI ships CUDA-13 builds).
  NVIDIA NPP for torchcodec CUDA comes from pip (`nvidia-npp-cu12`) and is preloaded by `visionlab.datasets.video`.
- torchvision ≥ 0.22 has no video decoding. Use torchcodec (train-time, CUDA) / PyAV (offline).
- torchcodec 0.16 NVDEC fails on the last 1–2 frames of ~6 % of HEVC clips; always keep a CPU fallback and never
  make the final two frames a window target on CUDA.
- FFmpeg < 5.1 uses `-vsync passthrough`, newer `-fps_mode passthrough`; both keep every frame + timestamp
  (assert output frame count == source frame count when re-encoding annotated video).
- x265 faster presets produce *smaller* files at the same crf, i.e. lower quality — not a free speed-up.

## Workstation fleet (until the fleet is under ansible)
- Work inside the `jupyter-grez72` container on each host: `C=$(docker ps --format '{{.ID}} {{.Names}}' | grep -E 'jupyter-grez72$' | awk '{print $1}')`,
  then `docker exec $C ...`; `docker exec -u root` for apt. Changes to running containers do not survive rebuilds:
  record them in the image recipe.
- machina: 64 threads, 2× A6000, **10 GbE direct link to the QNAP (1.2 GB/s)**, Flash at `~/work/DataExactitudeFlash`;
  its `~/work/DataLocal` is a 3.6 TB SATA SSD (small free space), not the NVMe root.
- leeloo / vesper / stelline / thrace: 32 threads, RTX 4090, campus 1 GbE to the QNAP (~118 MB/s), Flash at
  `~/work/DataRemote/qnap/exactitude/Flash`, containers run as **root** with the repo on a root-squashed mount:
  use `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/datasets`, `uv sync --no-install-project`, `uv run --no-sync`.
  No git credentials inside these containers (pull happens by other means).
- CPU-bound work parallelises across all of them: a 14 GB group reads in ~2 min over 1 GbE vs 60 min of encoding.
  Balance by measured rate (machina ≈ 1.4× a worker), launch in `tmux new -d -s <name>`, log to the shared tree.
- **Never `pkill -f <pattern>` from a shell whose own command line contains the pattern** (it kills the launcher).
  Use a bracket trick (`enc[o]de`) and run kill and launch as separate `docker exec` calls.
- CIFS: file mtimes seen from another client can lag hours; judge liveness by processes on the host, not by log
  mtimes. Long-open appended logs are fine, they just look stale remotely.
- Heredoc scripts over `ssh host 'docker exec ... bash -lc "..."'` break on nested quotes and Python 3.11
  f-strings: `scp` the script and `docker cp` it in instead.

## Working with the slipstream session
- slipstream is ours: ask for loader/cache changes rather than working around them. Give the slipstream agent the
  exact record layout, the failure you observed, and the acceptance test; require the full test suite before/after,
  new tests for the change, a version bump + CHANGELOG, and stable `slipstream.cli` helper signatures.
- Verify its branch yourself against the real store before merging (byte-exact payloads with `batches_ahead > 1`),
  then bump the visionlab-datasets pin (`uv lock --upgrade-package visionlab-slipstream`) and our version together.
