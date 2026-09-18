# SpatialVID-HQ: where were we? (resume here) — updated 2026-09-18 11:00 (datasets2 session ended at 43 % context)

Read this first, then `spatialvid-hq-subsets-and-loader.md` (design + all measurements) and
`spatialvid-hq-next-steps.md` (older plan, §2 cards still valid). Prep pipeline docs:
`datasets/prep/spatialvid_hq/README.md`. Decisions history: `slipstream/experiments/spatialvid_inspection/DECISIONS.md`.
Fleet access (ssh, container, paths, rules): the global `/workstation` skill.

## State — everything below is done, committed to `main`, data on QNAP Flash `.../VideoDatasets/SpatialVID-HQ-slipstream/`, mirrored on S3 `s3://visionlab-datasets/slipstream-cache/spatialvid-hq/`

| artefact | where | status |
| --- | --- | --- |
| **six h265 stores** `spatialvid-hq-h265-{640x360,456x256}[-{30,15}fps]`, 365,296 clips each | `stores/`, S3 | done + verified 2026-09-18 (clip sets identical; fps/`src_fps`/pts grids checked). Sizes: native 296 / 169 GB; 30 fps 311 / 176 GB; 15 fps 295 / 167 GB. fps stores: integer decimation k (60→30/15, 30→15, 50→50/16.7, 24→24), exact timestamps, fields `src_fps`, `src_num_frames`; `annot_frame_idx` indexes *source* frames. |
| training population `person_carried_v0` | `subsets/` + `.report.md`, S3 | 213,103 clips (walk 187,713 / rig 25,390), 644 h, 82 channels |
| **split v3** = train / val / test | `splits/v3.parquet` + report, S3 | train 186,710 · val 15,029 (whole held-out videos of channels in train, ≤ 1.2 pp off train per stratum) · test 11,364 (7 whole channels never in train; walk-only; the transfer metric). `make_splits --val-unit three --test-clips 11000 --val-clips 15000 --max-source-clips 200 --max-unit-frac 0.015` |
| **registry** `load("spatialvid-hq", split=, subset=, rate_hz=, res=, fps=, where=, channel_cap=)` → `VideoDataset` | `datasets/video_dataset.py`, `_configs/spatialvid_hq.py`, `tests/test_video_registry.py` | done; `rate_hz` picks the sparsest store whose fps divides it (15→15 fps, 10/30→30 fps, 5→15 fps); local order `$SLIPSTREAM_CACHE_DIR/<store>` → QNAP → S3 download of that store only; `ds.clips`, `ds.indices`, `ds.window_sampler(window_s)`, `ds.poses_at(rec, t_sec)` (batched, time-based). `datasets-cli list` shows stores/splits/subsets. |
| slipstream 0.7.0 | tag `v0.7.0` = 6e41138; pinned in `pyproject.toml`/`uv.lock` | done; machina container venv synced to 0.7.0 on 2026-09-18 (fleet-host venvs still 0.6.0: `uv sync --group video` there when next used) |
| local-SSD copy on machina | `~/work/DataLocal/slipstream-cache/spatialvid-hq-h265-456x256` (native, 169 GB) | done; set `SLIPSTREAM_CACHE_DIR=~/work/DataLocal/slipstream-cache`. **Not yet copied: the 456x256-15fps store** (167 GB; ~245 GB free there before) |
| end-to-end demo | `/tmp/demo_e2e.py` in the machina container (scratch) | ran 2026-09-17 on the native store: `load` 3.7 s, `[8,120,3,224,398]` batches with `video_t_sec`/`video_rec`, `poses_at` → `[8,120,7]`, seed-reproducible. Not yet run against the 15 fps store; not yet a notebook |
| fleet tooling | `datasets/prep/spatialvid_hq/{encode,fleet,status,finish}.py` | `fleet.py launch/sync/finish/status/ps/stop/tail`, claim-based load balancing, `encode --retry-failed` patch pass, VFR fallback. No jobs running as of 2026-09-18 11:00 |

## Decisions taken (user, 2026-09-17)

1. Split is **three-way** (v3): test = whole typical channels, val = whole videos stratified to the remaining population.
2. Pose interpolation lives in `visionlab.datasets` (`VideoDataset.poses_at`, `video.interpolate_poses`), not in slipstream; the stage returns true frame times.
3. fps is a store axis next to res; 30 and 15 fps built for both resolutions; no 224p store for now.
4. First-run window default: 4 s + 4 s at 15 Hz (T = 120; 145k clips ≥ 8 s, 87 % of frames).
5. Stores must contain the **full** clip set (no dropped clips) — hence the patch pass.

## Findings to remember (details and tables in the design doc)

- 5 Hz too slow, 10 Hz threshold, 15 Hz good → video store canonical; rate is a loader parameter.
- Decode cost per window follows the *stored* fps; a 15 fps store should ≈ halve `DecodeVideoWindow` cost vs 60 fps sources (measured raw: 200 → 94 ms per 8 s window). **Not yet benchmarked end-to-end.**
- x265 CRF scales with stream frame rate: fps stores are ~80 % of native bytes, not 50 %.
- torchcodec `get_frames_played_at(t)` = frame playing at t (floor, not nearest); harmless since true pts return.
- `OMP_NUM_THREADS=1` / `torch.set_num_threads(1)` in every process hosting the decode stage (DDP rule). CIFS page cache is per open-file holder → warm per process or stage to local SSD.
- ffmpeg decimation: `select=not(mod(n,k)),setpts=N*k/FRAME_RATE/TB,fps=src/k` is exact; the plain `fps` filter emits frame jk+1 for k=3,4. VFR sources (~0.03 %) need the `select`-only fallback. Exact rate/frame-count asserts must tolerate them.
- Channel holdout with heavy-tailed channels: pick *typical* channels (closest strata mix), cap 1.5 % each; val needs `--max-source-clips 200`.
- Fleet: always `docker exec -u jovyan` (fleet hosts use userns-remap; bare exec = root-in-ns). Details and paths in `/workstation`.

## Next work items, in order

1. Copy `stores/spatialvid-hq-h265-456x256-15fps` to machina's `~/work/DataLocal/slipstream-cache/` (check free space first; retire the native copy if tight).
2. Benchmark `DecodeVideoWindow` (T=120, 15 Hz, resize 224, 64 workers, OMP=1, warm) on the 15 fps vs 30 fps vs native store — `slipstream/benchmarks/bench_video_window.py` or the demo script; record in the design doc §3.
3. Turn `/tmp/demo_e2e.py` into `notebooks/spatialvid_hq_loader_demo.ipynb` (seed-reproducible batch + trajectories, cold vs warm throughput).
4. Per-store normalization stats → `metadata["stats"]` in `_configs/spatialvid_hq.py`.
5. Dataset card (next-steps §2): population definition, v3 split, channel concentration, exclusions, fps stores.
6. Later: per-clip stabilisation / gait statistics from poses; VLM audit sample for carrier precision; RA-4M prep (`datasets/prep/relate_anything_4M/SEED.md`, questions listed there).

## How to resume, mechanically

- Repo tools: `datasets/prep/spatialvid_hq/*.py`; run with `uv run --no-sync --group video python -m datasets.prep.spatialvid_hq.<tool> --out <tree>`
  inside the `jupyter-grez72` container on machina as jovyan (`ssh machina docker exec -u jovyan jupyter-grez72 bash -lc '...'`), repo `~/work/GitHub/datasets`.
  Tree on machina: `~/work/DataExactitudeFlash/DataSets/VideoDatasets/SpatialVID-HQ-slipstream` (fleet hosts: `~/work/DataRemote/qnap/exactitude/Flash/DataSets/VideoDatasets/...`).
- `fleet.py` runs from the Mac (ssh to the hosts); `fleet.py status --fps 30|15` for shard/patch state.
- Scratch scripts from the last session in the machina container: `/tmp/demo_e2e.py`, `/tmp/verify_fps_store.py`, `/tmp/verify_store.py`.
- Review page DB (channel labels): https://claude.ai/code/artifact/b8e6e6d8-f9c5-477a-94e0-2a5fbfe698cd, collection `labels`.
