# SpatialVID-HQ: where were we? (resume here) — written 2026-09-17, end of a long session

Read this first, then `spatialvid-hq-subsets-and-loader.md` (design + all measurements) and
`spatialvid-hq-next-steps.md` (older plan, §1 registry / §2 cards still valid). Prep pipeline docs:
`datasets/prep/spatialvid_hq/README.md`. Decisions history: `slipstream/experiments/spatialvid_inspection/DECISIONS.md`.

## State (all committed to `main`, data on QNAP Flash `.../VideoDatasets/SpatialVID-HQ-slipstream/`, index + subsets also on S3)

| artefact | where | status |
| --- | --- | --- |
| h265 stores 640x360 / 456x256, 365,296 clips, 60 fps sources | `stores/`, S3 | done, verified |
| YouTube metadata per source | `index/sources.parquet` (+ raw jsonl) | done (97 % of clips have a channel) |
| channel carrier review | page https://claude.ai/code/artifact/b8e6e6d8-f9c5-477a-94e0-2a5fbfe698cd (DB = labels); `index/channels.parquet` via `channel_review import` | 129/136 channels labelled (95.4 % of clips); 7 small channels unlabelled |
| clip carrier | `index/carrier_v1.parquet` (`make_subset.py`) | done |
| training population | `subsets/person_carried_v0.parquet` + `.report.md` | done: 213,103 clips (walk 187,713 / rig 25,390), 644 h, 82 channels; Rain Everyday 14.8 % |
| split v2 candidates | `splits/v2{a,b,c,d,e,f,g}-*.parquet` + reports (`make_splits.py` v2) | generated; **not adopted** |
| slipstream 0.7.0 | branch `feat/window-loader-array-fields` (DecodeVideoWindow, sample_data, float32[7] fields, seed_repeat, warmup touch, residency check) | ready both sides; **not merged/tagged** |
| loader throughput | 124–132 windows/s (T=120 @ 15 Hz, resize 224, 64 CPU workers, OMP=1, warm) on machina | closed; see §3 of the design doc |
| local-SSD copy of the 456x256 store | `~/work/DataLocal/slipstream-cache/spatialvid-hq-h265-456x256` on machina (was copying at session end, ~850 MB/s; `/tmp/copy_store.log` in the container) | check it finished (`ls`, no `.partial`) |

## Open decisions (the user's)

1. **Split v2**: adopt `v2f-hybrid` (3 typical held-out channels = 6,004 clips + 15,008 held-out videos; max val−train gap 4.8 pp) → rename to `splits/v2.parquet`, sync S3. Alternative v2g (5 channels, 11,350 clips, 12 pp).
2. **Merge + tag slipstream 0.7.0**; then pin `visionlab-slipstream >= 0.7.0` here.
3. **Frame-rate stores** (user's idea, 2026-09-17): support **several fps per store** rather than choosing one.
   Decode cost per window is set by the *source* fps (all frames between keyframes are decoded), so a 30 fps store
   halves the work for the 60 fps half of the data and a 15 fps store quarters it (measured 1 thread: 200 / 122 /
   94 ms per 8 s window). Since the archive 60 fps stores exist, extra stores are just more encodes (≈20 host-hours
   each for the subset, +73–81 GB each): make **fps a store axis next to res** — `spatialvid-hq-h265-456x256-30fps`
   etc., registry key `(fmt, res, fps)`, `load(..., rate_hz=15)` picks the sparsest store whose fps is a multiple of the
   requested rate, and the S3 sync / local cache only pulls the stores a job uses. Also consider a 224-short-side
   store (removes the 20 % decoder-side resize, ~25 % fewer pixels). Candidates: 30 fps 456x256 (exact 5/10/15/30 Hz),
   15 fps 224p (fastest, exact 5/15 Hz). **decide** which to encode first; `encode.py` needs an `--fps` (ffmpeg
   `fps=` filter, `keyint=fps`) and `--res` preset.
4. **Window default** for the first run: 4 s + 4 s at 15 Hz (T=120, clips ≥ 8 s: 68 % of clips / 87 % of frames) vs 2 s + 2 s (94 % of clips).

## Findings to remember (details and tables in the design doc)

- 5 Hz is too slow for the task, 10 Hz threshold, 15 Hz good (DECISIONS addendum 2) → video store canonical, no
  frame store; rate is a loader parameter; poses interpolated to the true frame times.
- DecodeVideoWindow: 22 → 124 windows/s through in-flight depth, HWC output, `OMP_NUM_THREADS=1`, all 64 hardware
  threads. **Rule:** `torch.set_num_threads(1)` in every process that hosts the stage (N DDP ranks otherwise decode at
  the speed of one: torch's per-process intra-op pool). Allocator / GIL / wait-policy: no effect.
- **CIFS page cache** (QNAP mount `cache=strict`, `actimeo=1`): pages read by a process are dropped when the last
  process holding the file exits; a fresh process starts cold (residency 0.0, ~65 windows/s, ~1k major faults for
  640 records); within a process epoch 2+ is warm (131/s); a concurrent holder keeps pages alive for other processes
  (DDP ranks on one node share). Random mmap faults from many processes without warmup: 2.4 windows/s. Rules:
  `warmup_cache()` once per process start (or stage the store to node-local disk, the normal slipstream pattern);
  subset = 100.3 GB in the 456x256 store, fits machina's 503 GB RAM; warmup reads at 200–380 MB/s from the QNAP.
- Channel holdout: with 82 heavy-tailed channels, deficit-greedy channel selection picks atypical channels (house-tour
  rigs) and skews val 17 pp; selecting *typical* channels (closest strata mix) keeps it under 5 pp.
- Carrier = who moves the camera (walk, rig, drive, drone, train, boat, bike, mixed, other); "house" is a genre word.
  Walk includes stabilised "floaty" walking; stabilisation level is a future per-clip attribute (pose statistics).

## Next work items, in order

1. Adopt a split (decision 1) → `splits/v2.parquet`, S3.
2. Registry (next-steps §1): `res` (and `fps`) axes, video config `stores / splits / subsets`, `load()` returning
   `ds.cache`, `ds.clips`, `window_sampler`, `poses_at`; `where=` / `channel_cap=`; slipstream pin; image datasets
   unchanged (`load("imagenet1k", split="val")`).
3. Extra-fps store(s) (decision 3): `encode.py --fps --res`, fleet run, merge, S3 sync of only that store.
4. Demo notebook: seed-reproducible `[B,120,3,224,398]` batch with trajectories at 15 Hz; measure windows/s cold vs
   warm from the local-SSD copy vs CIFS.
5. Dataset card (next-steps §2) incl. population definition, channel concentration, exclusions.
6. Later: per-clip stabilisation / gait statistics from poses; per-clip VLM audit sample (300 clips) for carrier precision.

## How to resume, mechanically

- Repo: `datasets/prep/spatialvid_hq/{fetch_sources,channel_review,make_subset,make_splits}.py`; run with
  `uv run --no-sync --group video python -m datasets.prep.spatialvid_hq.<tool> --out <QNAP tree>` inside the
  `jupyter-grez72` container on machina (`ssh machina`, `docker exec`), repo at `~/work/GitHub/datasets`.
- slipstream branch checkout used for benchmarks: `/tmp/ss` in the container (`PYTHONPATH=/tmp/ss`); bench:
  `/tmp/ss/benchmarks/bench_video_window.py --cache <store> --indices /tmp/recs.npy --T 120 --rate 15 --resize 224
  --batch-size 8 --workers 64 --reuse-output --warm`; scratch scripts `/tmp/{bench_windows,raw_conc2,diag,warmtest,resid}.py`.
- slipstream2 session (the slipstream maintainer agent) is reachable by cross-session message; it has the full
  throughput history.
- Review page DB: `Artifact read_db` on the URL above, collection `labels`; export via `channel_review import`.
