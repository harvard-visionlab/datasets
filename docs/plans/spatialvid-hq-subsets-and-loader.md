# SpatialVID-HQ: subsets, frame rate and the training loader (design, 2026-09-16)

Decisions for the first training population and how clips become batches. Facts measured on the built stores;
open choices are marked **decide**. Background: `spatialvid-hq-next-steps.md` §3, DECISIONS.md §4–5, slipstream2
handoff of 2026-09-16.

## 0. Facts the design rests on

- Both h265 stores (`640x360`, `456x256`) hold the same 365,296 clips in the same record order → one `record_idx`
  addresses a clip in either store.
- Carrier decisions: 129/136 channels labelled (95.4 % of clips): walk 213k clips, drive 52k, rig 33k, mixed 21k,
  bike 9k, drone 8k, train 7k, other 6k (`index/channels.parquet`, per-video title/tags overrides included).
- `subsets/person_carried_v0.parquet` (built by `make_subset.py` from `index/carrier_v1.parquet`; reproduces
  slipstream2's hand-run table to 0.15 %): walk + rig, no `stationary` motion tag, speed ≤ 0.5 → **213,103 clips**
  (walk 187,713 / rig 25,390), 644 h, 106.7 M source frames, 13,222 videos, 82 channels. Report: `subsets/person_carried_v0.report.md`.
  fps: 60 (51 %), 30 (38 %), 24 (6 %), 50 (3 %), 25 (2 %). Duration floors: ≥4 s keeps 88 % clips / 97 % frames,
  ≥6 s 77 / 92, ≥8 s 68 / 87 (144,866 clips). Rain Everyday is 14.8 % of the subset, top-5 channels 36.7 %.
- Split v1 ∩ subset = 206,698 train / **6,612 val (3.1 %)**: long recordings (all the big walking channels) were
  val-ineligible in v1. v1 is unusable as the walking val set.
- Poses exist at ≈5 Hz (every `int(fps/5)` frames): exactly 5 Hz at 25/30/50/60 fps, 6 Hz at 24 fps.
  `VideoStore.poses_at` interpolates (linear t, slerp q) to any frame time.
- slipstream 0.6.0: `SlipstreamLoader(indices=)`, `warmup_cache(indices=)`, bytes primary fields. **No window /
  sequence sampling yet** (DECISIONS §4 gap 1), no fixed-shape array fields (gap 2).
- torchcodec: `VideoDecoder(bytes).get_frames_played_at(seconds)` returns the frames nearest to the requested
  times, so fps never enters a time-based sampler. Cost: every frame from the preceding keyframe (1 s GOP) through
  the window is decoded, e.g. ~480 decoded frames to return 40 at 5 Hz over 8 s of 60 fps video. Measured CPU
  throughput on machina: ~775 windows/s for 8 consecutive frames → order 100 windows/s for 8 s windows (estimate;
  NVDEC untested for this pattern).

## 1. Subset = definition; cutoffs = parameters

**Decision.** A *named subset* is a versioned parquet that fixes a population: which carriers, which clip-level
motion filters, which channel-decision version. Everything derivable from its columns is a **load-time parameter**,
not a new file: minimum duration, walk-only vs walk+rig, per-channel cap, split membership. The table has 213k rows;
any such query is milliseconds. So "person, ≥2 s / ≥4 s / ≥8 s" are not three subsets. In fact the duration floor is
never chosen by hand: a window of `window_s` seconds implies `duration_s ≥ window_s`, and the sampler applies it.

```
subsets/person_carried_v0.parquet   clip_id, record_idx, source_id, channel_id, carrier, duration_s, num_frames,
                                    fps, speed, motion_tags           (+ .report.md: definition, counts, top channels)
```

A new **definition** (bike joins, carrier labels v2, a hand-vetted walking set) → `person_carried_v1.parquet`.
A new **view** (walk only, cap Rain Everyday at 5 %, ≥8 s) → parameters, recorded in the run config.

## 2. Frame rate: a loader parameter on a time grid (10–15 Hz needed)

**Decision (revised 2026-09-16 evening).** DECISIONS.md addendum 2: 5 Hz reads as separate snapshots, 10 Hz is the
threshold, 15 Hz looks continuous. So samples are defined in seconds on a grid `t_k = t0 + k / rate_hz` with
`rate_hz` a loader parameter (15 default; 5/10/30 allowed). torchcodec `get_frames_played_at` returns the nearest
frame per time regardless of source fps, and the **true** presentation times come back with the frames; poses are
interpolated to those times (exact at 25/30/50/60 fps, ≤ 21 ms nearest-frame error at 24/48 fps), so ego-motion
always uses the true `dt`.

## 3. Storage: the h265 video store is canonical; decode in the loader

**Decision.** No frame store. At 15 Hz a JPEG frame store of the subset alone would be ~800 GB against 178 GB of
HEVC (addendum 2 reached the same conclusion at 5 Hz already). Training decodes windows from the 456x256 h265
store inside the loader: slipstream `DecodeVideoWindow` (branch `feat/window-loader-array-fields`, 0.7.0):
`(record, t0, T, rate_hz)` per sample, seeded random or given `t0`, CPU thread pool or NVDEC, true `t_sec`
returned, augmentation drawn once per window (`seed_repeat = T`), last two frames never requested (NVDEC bug),
CPU retry.

Measured 2026-09-16 on machina, 8 s windows at 15 Hz (T = 120), 456x256 store, warm cache:

| path | windows/s | note |
| --- | ---: | --- |
| raw torchcodec, 1 thread, 60 fps source | 5.0 /core | 200 ms per window → ~240/s ideal on 48 cores |
| raw, 30 fps re-encode (81 % of the bytes) | 8.2 /core | 122 ms |
| raw, 15 fps re-encode (73 % of the bytes) | 10.7 /core | 94 ms |
| `DecodeVideoWindow` 1475443, CPU pool 32 or 48 | 22 | only one batch's decodes in flight (bug) |
| `DecodeVideoWindow` a5f9d33 (pipelined), CPU 16 / 24 / 32 / 48 workers, resize 224 | 43 / 59 / 72 / 87 | ~linear to the core count; ~370 ms per window per worker vs 200 ms raw |
| `DecodeVideoWindow` a5f9d33, NVDEC 8 decoders one GPU / 16 two GPUs / 32 two GPUs, resize 224 | 25 / 44 / 71 | scales with worker count → host-bound, not NVDEC-bound |
| `DecodeVideoWindow` 45cf3e2 (HWC output, reuse_output), CPU 48 workers, default torch threads | 93 | |
| same, `OMP_NUM_THREADS=1`, 48 / 64 workers | 111 / **124** | torch/OMP parallel regions inside torchcodec cost ~19 %; use all hardware threads, 1 ffmpeg thread per decoder |
| allocator (jemalloc, glibc thresholds), GIL switch interval, OMP_WAIT_POLICY | ±3 % | ruled out |
| raw torchcodec in N separate *processes*, warm bytes | 2.4 total, any N | torch's per-process intra-op pool (32 threads × N, spinning): `torch.set_num_threads(1)` in each child restores full speed (N = 8: 2,965 → 185 ms per window). Rule for DDP: cap torch threads in every process that hosts the decode stage |

**Consequences.** (a) The stage, not the data, was the bottleneck: in-flight depth (22 → 87), HWC output (→ 93),
`OMP_NUM_THREADS=1` and all 64 hardware threads (→ 124 windows/s). That is ~2× what a first training run needs
(~64 at B = 32, 0.5 s/step). Trainer settings: `num_workers = os.cpu_count()`, 1 ffmpeg thread per decoder,
`reuse_output=True`, `OMP_NUM_THREADS=1` in the loader process (slipstream warns when it is not set). Cold reads
from the CIFS mount cost ~400 ms per record and serialize, so `warmup_cache(indices=)` before each epoch is mandatory
(`page_cache_residency()` checks it). (b) A 15 fps re-encode buys ≤ 2× decode and 27 % storage while pinning the
rate; only worth it if the tuned stage plus NVDEC still falls short. A 30 fps re-encode keeps 5/10/15/30 Hz exact
and is the fallback of choice. (c) Realistic requirement: a batch of 32 windows per optimizer step at ~0.5 s/step
is ~64 windows/s; at T = 120 and full 456x256 that is 5 GB/s of uint8 frames, so the decoder-side `resize=` to
the training resolution is part of the design, not an option. Target for the stage: ≥ 100 windows/s at T = 120
with resize to the model input, CPU or one GPU.

The optional on-the-fly `VideoStore.frames_at_seconds` path stays for notebooks and eval at 640x360.

## 4. Loader API

```python
from visionlab.datasets import load
ds = load("spatialvid-hq", split="train", fmt="h265", res="456x256",
          subset="person_carried_v0",
          where="carrier == 'walk'",          # optional pandas query over the subset table
          channel_cap=0.05)                    # optional: max share of clips per channel (random thinning, seeded)
# ds.cache        slipstream OptimizedCache (the h265 store)
# ds.clips        DataFrame: the selected clips (subset ∩ split ∩ where ∩ cap) with record_idx
# ds.stats        normalization stats for the store

stage = DecodeVideoWindow(T=120, rate_hz=15, seed=0, resize=224, transforms=[RandomResizedCropBatch(...), flip])
# clips shorter than window_s = T / rate_hz are filtered here (the stage raises on a misfit at decode time)
recs, t0 = ds.window_sampler(window_s=8.0, anchors_per_clip=k, seed=0).sample(epoch)   # or let the stage draw t0
loader = SlipstreamLoader(ds.cache, indices=recs, sample_data={"t0": t0}, batch_size=32, pipelines={"video": [stage]}, ...)
for batch in loader:
    batch["video"]        # [B, T, 3, H, W] uint8, same crop/flip/color across T
    batch["video_t_sec"]  # [B, T] true frame times → poses = ds.poses_at(batch["video_rec"], batch["video_t_sec"])  [B, T, 7]
    rel = relative_motion(poses)   # [B, T-1, 6] camera-t axes, from visionlab.datasets.video
    batch["clip_id"], batch["source_id"], batch["channel_id"], batch["carrier"]
```

`split` is resolved from `splits/<version>.parquet` (clip → split) and intersected with the subset; `split="all"`
gives everything. Image datasets are untouched (`load("imagenet1k", split="val")` unchanged; `res` gets a per-dataset
default as planned in next-steps §1).

Model-side defaults (**decide**): 15 Hz, 4 s warm-up + 4 s prediction = 120 steps, `window_s = 8` → 144,866
clips, 87 % of frames; alternative 2 s + 2 s keeps 94 % of clips. Nothing about rate or window is baked into the
store; poses come from the clip's annotation arrays, interpolated to the returned frame times.

## 5. Splits before any training run

v1 cannot be used (3.1 % val, no big walking channel in val). Split **v2**: unit = source video, stratify on
carrier × scene × time of day × weather × motion, **and hold out whole channels** for a leave-channel-out val
(the honest test for "generalizes to rain / night", see next-steps §3). Raise or drop the `--max-source-clips`
val-eligibility cap for walk. Report within-subset balance (walk_v0 val fraction, per-dim tables, channel
concentration per stratum). Then `train` = subset ∩ v2-train etc.

## 6. slipstream (status 2026-09-16 evening)

Delivered on `feat/window-loader-array-fields` (0.7.0, not merged): `DecodeVideoWindow` as specified in §3,
loader `sample_data=` (per-sample side arrays aligned with `indices`, e.g. `t0`), fixed-shape array fields
(`float32[7]`), `seed_repeat` (one augmentation draw per window), `RandomResizedCropBatch`, and record-window
expansion `window=(T, stride)` (built for the abandoned frame store; harmless, default off). Open: the stage's
throughput (§3 table): profile the CPU path (GIL / per-sample decoder construction with `seek_mode="exact"` /
main-thread stacking), try `seek_mode="approximate"` (1 s GOP), `get_frames_in_range` with step, larger B, decode
inside the prefetch thread; NVDEC with 8 persistent decoders per GPU on both GPUs. Merge + tag is the user's call.

## 7. Work plan

1. `subsets/person_carried_v0` report + `datasets/prep/spatialvid_hq/make_subset.py` (re-creates the parquet from
   `channels.parquet` + `clips.parquet`, so the definition is code).
2. Stage throughput to ≥ 100 windows/s (slipstream2), measured with `benchmarks/bench_video_window.py` on machina;
   fallback: 30 fps re-encode of the subset.
3. Split v2 (§5) and its report.
4. Registry: `res` axis, video config shape (`stores`, `splits`, `subsets`), `load()` returning `ds.clips`,
   `window_sampler`, `poses_at`; `where` / `channel_cap`; pin slipstream ≥ 0.7.0 once merged.
5. Demo notebook: `[B,120,3,224,224]` batch at 15 Hz with trajectories, seed-reproducible, throughput measured.
7. Card: population definition, counts, channel concentration, the 7 unlabelled channels (4.6 % of clips) and
   `mixed` (21k clips) excluded.
