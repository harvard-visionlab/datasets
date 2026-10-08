# ProcTHOR walks → slipstream video stores

Reproducible pipeline from Rupert Tawiah-Quashie's (Harvard Vision Lab) H5 walk files to two slipstream video stores,
their index and their split tables. Measurements behind every setting: `benchmarks/procthor_encode_trial.py`,
`benchmarks/bench_procthor_h5.py`, `benchmarks/bench_procthor_slipstream.py`, `benchmarks/procthor_quality_sbs.py`
(results summarised below and in `tjepa-dev/docs/BENCHMARK_RESULTS.md`).

## Source

```
/n/netscratch/kempner_konkle_lab/Everyone/rtawiahquashie/habitat_data/
  datasets_r160_zstd/<cell>/{train,val,test}/house_XXXXX/seed_{1000,1001}.h5     cell = pt_objects_brownian | pt_empty_brownian
  runs/<cell>__{bridge,lr6e4,lrsplit,cov01}_r160[__passive]/active_reproducibility/{train,val}_windows.tsv
```

Each H5 file is one 1,000-step walk through one ProcTHOR-10k house, rendered with AI2-THOR 5.0.0 at 160×120 (hfov 90°,
camera height 1.454 m, physics paused, agent invisible):

| H5 path | shape / type | contents |
| --- | --- | --- |
| `trajectory/visual_scenes` | (1000, 120, 160, 3) uint8 | RGB frames; blosc-zstd level 3 + bitshuffle, 16-frame chunks (~1.9:1, ~28 MB/walk) |
| `trajectory/agent_positions` | (1000, 2) float64 | agent (x, y) in metres |
| `trajectory/agent_headings` | (1000, 1) float64 | heading in radians |
| `meta/metadata` | JSON string | generator settings and house facts (rooms, objects, materials, step_std 0.1, angle_std 0.15, ...) |
| `meta/env_name`, `meta/n_steps`, `meta/seed`, `meta/{min,max}_{x,z}` | scalars | |

The two cells are paired: same houses, same seeds, same camera paths (`paired_trajectories: true`); "empty" is the
furnished house with the furniture removed. 10,000 / 1,000 / 1,000 houses in train / val / test, 2 walks each.
The metadata's `fps` (~26) is render throughput, not a frame rate: the walk is step-indexed and has no time axis.

## Layout

```
<work> = $LAB_SCRATCH/procthor-walks
  <dataset>/index/walks.parquet              one row per walk in store order: clip_id, folder, house, seed, src_path, metadata
  <dataset>/splits/{v1,rtq-r160}.parquet     clip_id -> split, + .report.md (counts)
<store-root> = /n/netscratch/alvarez_lab/Lab/datasets/slipstream      (shared training cache; built here)
  <dataset>-h264-160x120-{train,val,test}/   one store per source folder: slipstream fields + records.parquet + store_manifest.json
masters: /n/lab_storage/alvarez_lab/Lab/datasets/slipstream/<store>/ and s3://visionlab-datasets/slipstream-cache/<dataset>/
```

`clip_id` = `<folder>/house_XXXXX/seed_XXXX`, identical in both datasets for paired walks; record order within a store
(house, seed) is identical too, so `record_idx` also pairs the walks. **One store per source folder** (train ~48 GB,
val and test ~5 GB each), so a demo or an evaluation downloads only what it needs; `load()` picks the store that holds
the requested split (the config's store entry is `{folder: S3 path}`; `video_dataset.store_part`). `walk_idx` in
records.parquet is the stage-1 index across folders.

## Stages

Run from the repo root on a compute node (`. lab_env.sh; uv run --no-sync python -m ...`), once per dataset
(`--dataset procthor-walks-objects | procthor-walks-empty`).

| # | command | writes | cost (FASRC `shared`) |
| - | --- | --- | --- |
| 1 | `datasets.prep.procthor_walks.build_index --dataset D --out <work>` | `index/walks.parquet`, `splits/v1`, `splits/rtq-r160` (+ reports); checks the 8 runs per cell share one window list | ~5 min, 8 threads |
| 2 | `datasets.prep.procthor_walks.encode --dataset D --folder {train,val,test} --out <work> --store-root <store-root> --workers 32` (`--limit N` = test store) | one store | train ~50 min, val/test ~5 min at 32 cores (~4.8 core-s per walk) |
| 3 | `datasets.prep.procthor_walks.verify --dataset D --store <store> --out <work>` (per store) | VERIFY line (exit 1 on failure) | minutes |
| 4 | `datasets.prep.procthor_walks.publish --dataset D --out <work> --store <store>` (per store) | lab_storage master, private S3 copy, anonymous-GET check | ~1 h (S3 from the cluster is slow) |

## Data model

One **record = one walk** (`common.py`, `encode.WalkSource`):

| field | type | contents |
| --- | --- | --- |
| `video` | bytes | mp4, h264 High 4:4:4 (`yuv444p`), crf 10, preset medium, GOP 60 (keyframes at 0, 60, ..., 960), 30 fps, frame i at pts i |
| `positions` | float32[1000, 2] | source `agent_positions` (cast from float64) |
| `headings` | float32[1000] | source `agent_headings` (cast from float64) |
| `fps` | float | 30 (convention: frame i is at t = i / 30 s) |
| `num_frames` | int | 1000 |
| `duration_s` | float | 33.33 |

`records.parquet` (in the store) = `index/walks.parquet`: `record_idx`, `clip_id`, `folder`, `house`, `seed`,
`src_path`, and per-walk generator metadata (`n_rooms`, `room_types`, `n_objects`, `n_windows`, `n_doors`,
`wall_materials`, `floor_materials`, `time_of_day`, `skybox`, `mean_brightness`, `n_reachable`, `stalls`, `step_std`,
`angle_std`, `camera_height_m`, `hfov_deg`, `vfov_deg`, `ai2thor`, `env_name`, ...).

**Training windows.** The student's runs use 61-frame windows (20 warm-up + 40 predicted + 1) at starts 0, 60, ..., 900:
16 per walk. With GOP 60 every window starts on a keyframe, so a window decodes exactly 61 frames. Decode by frame
index (`torchcodec ... get_frames_in_range(s, s + 61)`), or with slipstream's `DecodeVideoWindow(T=61, rate_hz=30)`
and `t0 = (s + 0.5) / 30` (mid-frame, so float rounding cannot select a neighbour).

## Splits

| version | train | val | other | definition |
| --- | --- | --- | --- | --- |
| `v1` (default) | 20,000 | 2,000 | test 2,000 | the source folder |
| `rtq-r160` | 18,000 | 2,000 | val_unseen 400, test 2,000, unused 1,600 | the student's r160 runs: train and val both from the train folder (their window lists); val_unseen = first 400 sorted val-folder walks (their held-out-houses set) |

## Why these settings (2026-10-07/08, details in the benchmark scripts)

- **h264 yuv444p crf 10.** 4:2:0 chroma caps PSNR near 38 dB on these 160×120 renders at any CRF (visible as soft
  colour edges); 4:4:4 crf 10 gives 42.6 dB mean (objects; ~46 empty) at 2.4 MB/walk, 12× smaller than the H5 files,
  and was judged indistinguishable from the originals in a blink comparison (George, 2026-10-08). JPEG q100 4:2:0
  (40.5 dB, 12 MB/walk) also looked identical but is 5× larger and was slower in slipstream (per-frame records).
- **GOP 60** = window stride, so no window decodes frames it does not return.
- **One record per walk** (not per window or frame): windows overlap the walk's GOPs exactly; positions/headings ride
  along as fixed-size array fields.
- **Loader throughput** (val split, 32k windows, B=256): student H5 loader 288 windows/s (16 CPU) / 462 (24) / 602 (95);
  slipstream mp4 from page cache 395 (16) / 816 (24) / 1,109 (96, one process). A 4×H100 DDP run of the student's
  model needs ~1,900 windows/s.
