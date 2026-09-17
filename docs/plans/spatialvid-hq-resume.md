# SpatialVID-HQ: where were we? (resume here) — updated 2026-09-17 (evening session, datasets2)

Read this first, then `spatialvid-hq-subsets-and-loader.md` (design + all measurements) and
`spatialvid-hq-next-steps.md` (older plan, §2 cards still valid). Prep pipeline docs:
`datasets/prep/spatialvid_hq/README.md`. Decisions history: `slipstream/experiments/spatialvid_inspection/DECISIONS.md`.

## State (all committed to `main`, data on QNAP Flash `.../VideoDatasets/SpatialVID-HQ-slipstream/`, index + splits + subsets on S3)

| artefact | where | status |
| --- | --- | --- |
| h265 stores 640x360 / 456x256, native fps, 365,296 clips | `stores/`, S3 | done, verified |
| **fps stores** `…-{640x360,456x256}-{30,15}fps` (integer decimation, exact timestamps, `src_fps`/`src_num_frames` fields) | `shards/<res>-<fps>fps/` → `stores/`, S3 | **encoding on the fleet since 2026-09-17 ~23:00** (machina, thrace, vesper, leeloo, stelline; 30 fps pass first, each host chains into the 15 fps pass; a finisher on machina merges + S3-syncs each axis when its 74 shards exist). ~10 clips/s fleet-wide → 30 fps pass ≈ 10 h (63/74 groups at 08:20 on 09-17), 15 fps pass ≈ 8 h. **116 clips (0.03 %) failed the 30 fps pass** on over-strict rate/frame-count checks (slightly variable-rate sources); checks relaxed (2 %, floor/ceil) before the 15 fps pass, and a **patch pass** (`encode --retry-failed`, `fleet.py launch --extra=--retry-failed --log-tag _retry`) re-encodes exactly the missing clips into `group_XXXX_patch` shards; `finish.py` merges only when all 74 shards exist and every failure is covered (user wants the full clip set in every store). **Check:** `python datasets/prep/spatialvid_hq/fleet.py status --fps 30` (or `--fps 15`), `fleet.py tail`, `logs/finish.log` in the tree |
| YouTube metadata, channel review (129/136), clip carrier | `index/` | done |
| training population `person_carried_v0` | `subsets/` + `.report.md` | done: 213,103 clips (walk 187,713 / rig 25,390), 644 h, 82 channels |
| **split v3 (adopted)** = train / val / test | `splits/v3.parquet` + report, S3 | **done**: train 186,710 · val 15,029 (whole held-out videos of channels in train, ≤ 1.2 pp off train per stratum) · test 11,364 (7 whole channels never in train: 4K Nature and City Walks, The Flying Dutchman, Justwalk, TokyoNinjaWalk, Drifted Films, Hui Chen, Trillionex Travel; walk-only, Cloudy-heavy — the transfer metric, reported separately). `make_splits --val-unit three --test-clips 11000 --val-clips 15000 --max-source-clips 200 --max-unit-frac 0.015`. v2a–g candidates kept for reference, not used. |
| **registry**: `load("spatialvid-hq", split=, subset=, rate_hz=, res=, fps=, where=, channel_cap=)` → `VideoDataset` | `datasets/video_dataset.py`, `_configs/spatialvid_hq.py`, tests `tests/test_video_registry.py` | **done** (13 tests). Store pick: sparsest store whose fps divides `rate_hz`, unbuilt stores skipped with a warning (so today 15 Hz → native store; → 15 fps store once synced). Local order: `$SLIPSTREAM_CACHE_DIR/<store>` → QNAP mount → S3 download of that store only. `ds.window_sampler`, `ds.poses_at(rec, t_sec)` (batched, time-based, uses `src_fps`). |
| slipstream 0.7.0 | `origin/main` = 6e41138, tag `v0.7.0` (slipstream2, 2026-09-17) | **done**; pinned here (`pyproject.toml` git rev `v0.7.0`, `uv.lock` updated, local env synced, 78 tests pass). **machina container venv still has 0.6.0**: run `uv sync --group video` there *after* the fleet encode finishes (encode.py imports slipstream.cache lazily; do not swap packages under running encoders). Until then `PYTHONPATH=/tmp/ss` for loader work on machina. |
| local-SSD copy of the native 456x256 store on machina | `~/work/DataLocal/slipstream-cache/spatialvid-hq-h265-456x256` (169 GB) | done; `SLIPSTREAM_CACHE_DIR=~/work/DataLocal/slipstream-cache` makes `load()` use it |
| demo (registry → anchors → `DecodeVideoWindow` → `poses_at`) | `/tmp/demo_e2e.py` in the machina container (scratch; to become `notebooks/spatialvid_hq_loader_demo.ipynb`) | run 2026-09-17 while the host encoded; see the session summary / re-run for numbers |

## Decisions taken 2026-09-17 (user)

1. Split: **three-way** (test = whole typical channels, val = whole videos stratified to the remaining population) instead of v2f's single val. Adopted as v3.
2. slipstream 0.7.0: merged + tagged v0.7.0, pinned here. Pose interpolation is *not* a slipstream transform: the stage returns `video_t_sec`, `visionlab.datasets` interpolates (`VideoDataset.poses_at`, `video.interpolate_poses`).
3. fps stores: **30 fps first, then 15 fps, both resolutions** (4 stores); resolution and rate decided separately; 224p store not now. Decimation rule in `common.decimation_factor`: k = round(src/F) lowered until src/k ≥ F − 0.1 (60→30, 50→50, 24→24 at F=30; 60→15, 50→16.7, 30→15, 24→24 at F=15). Filter `select+setpts+fps` (the plain `fps` filter emits frame jk+1 for k=3,4 — measured).
4. Window default for the first run: 4 s + 4 s at 15 Hz (T = 120).

## Findings to remember (details and tables in the design doc)

- 5 Hz is too slow, 10 Hz threshold, 15 Hz good → video store canonical; rate is a loader parameter; poses interpolated to the true frame times.
- torchcodec `get_frames_played_at(t)` returns the frame *playing at* t (pts ≤ t < pts + dur, i.e. floor, not nearest); harmless because the true pts come back and poses use them (demo: t0 2.866 → first frame pts 2.833 at 30 fps).
- Demo 2026-09-17 (`/tmp/demo_e2e.py`, machina while encoding, 8 workers): `load` 3.7 s from the local-SSD store; val 15,029 clips → 10,210 anchors ≥ 8 s; `[8,120,3,224,398]` uint8 batches with `video_t_sec`, `video_rec`; `poses_at` → `[8,120,7]`; seed-reproducible across loaders; `channel_cap=0.05` on train-walk → 134,312 clips, top channel 6.1 %.
- x265 CRF depends on the stream frame rate (same CRF at 30 fps spends ~2× bytes per frame vs 60 fps); the fps store bytes are ~80 % of the native store, not 50 %.
- `torch.set_num_threads(1)` / `OMP_NUM_THREADS=1` in every process that hosts `DecodeVideoWindow` (DDP rule).
- CIFS page cache is per open-file holder: `warmup_cache()` per process start or stage the store to node-local disk.
- Fleet hosts (thrace, vesper, leeloo, stelline) run docker with `userns-remap`: a bare `docker exec jupyter-grez72` is **root-in-namespace** (host uid 100000) and owns nothing — always `docker exec -u jovyan`. (2026-09-17 I ran as root there and wrongly concluded 'no GitHub key' / 'DataLocal read-only'; both are fine as jovyan.) jovyan-in-container = host uid 101000:100100; the datasets repo on those hosts was host-`george`-owned (and my host-side rsync made it worse) → chowned to 101000:100100 on 2026-09-17. machina has no userns (exec = jovyan) and its own mount layout: QNAP Flash at `~/work/DataExactitudeFlash` vs `~/work/DataRemote/qnap/exactitude/Flash` on the fleet. The 30/15 fps encoders launched on 2026-09-17 on the fleet run as ns-root (harmless: shards/claims/logs live on the uid-mapped CIFS; scratch `/tmp/spatialvid`); future launches run as jovyan with scratch in `DataLocal`.
- Channel holdout with 82 heavy-tailed channels: pick *typical* channels (closest strata mix), cap 1.5 % of clips each; video-level val needs `--max-source-clips 200` so long walking videos are eligible.

## Next work items, in order

1. Watch the fleet (`fleet.py status`); when `finish.log` says an axis is synced, `load(..., rate_hz=15)` picks it up automatically (config already lists all six stores). Verify one fps store with `VideoStore` (fps, src_fps, pts grid) — `verify_store.py` pattern from this session.
2. `uv sync --group video` on machina + fleet hosts once encodes finish (slipstream 0.7.0); move the demo into a notebook with throughput cold vs warm, local SSD vs CIFS, native vs 30 fps vs 15 fps store.
3. Per-store normalization stats (`metadata["stats"][store]`), `datasets-cli list` support for video stores.
4. Dataset card (next-steps §2): population definition, v3 split, channel concentration, exclusions, fps stores.
5. Later: per-clip stabilisation / gait statistics; VLM audit sample for carrier precision; RA-4M prep (`datasets/prep/relate_anything_4M/SEED.md`).

## How to resume, mechanically

- Repo tools: `datasets/prep/spatialvid_hq/{make_splits,encode,merge,status,fleet,finish}.py`; run with
  `uv run --no-sync --group video python -m datasets.prep.spatialvid_hq.<tool> --out <tree>` inside the `jupyter-grez72` container
  (`ssh machina; docker exec -it jupyter-grez72 bash`), repo at `~/work/GitHub/datasets`. `fleet.py` runs from any machine with ssh to the hosts.
- slipstream branch checkout for the loader: `/tmp/ss` in the machina container (`PYTHONPATH=/tmp/ss`).
- Review page DB (channel labels): https://claude.ai/code/artifact/b8e6e6d8-f9c5-477a-94e0-2a5fbfe698cd, collection `labels`.
