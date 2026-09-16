# SpatialVID-HQ: subsets, frame rate and the training loader (design, 2026-09-16)

Decisions for the first training population and how clips become batches. Facts measured on the built stores;
open choices are marked **decide**. Background: `spatialvid-hq-next-steps.md` §3, DECISIONS.md §4–5, slipstream2
handoff of 2026-09-16.

## 0. Facts the design rests on

- Both h265 stores (`640x360`, `456x256`) hold the same 365,296 clips in the same record order → one `record_idx`
  addresses a clip in either store.
- Carrier decisions: 129/136 channels labelled (95.4 % of clips): walk 213k clips, drive 52k, rig 33k, mixed 21k,
  bike 9k, drone 8k, train 7k, other 6k (`index/channels.parquet`, per-video title/tags overrides included).
- `subsets/person_carried_v0.parquet` (from slipstream2's pass, moved here with `record_idx`): walk + rig, no
  `stationary` motion tag, speed ≤ 0.5 → **213,310 clips** (walk 187,914 / rig 25,396), 645 h, 11.6 M frames at 5 Hz.
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

## 2. Frame rate: fixed 5 Hz time grid, everywhere

**Decision.** Samples are defined in seconds on a 5 Hz grid `t_k = 0.2·k`, `k = 0 … floor(5·duration_s) − 1`.
For each `t_k` the frame with nearest presentation time is used and its **true** `t_sec` is stored; poses are
interpolated to that `t_sec` (exact at 25/30/50/60 fps; ≤ 21 ms nearest-frame error at 24/48 fps). Ego-motion is
computed from the stored poses, so `dt` is always the true one. 5 Hz matches the annotation rate; other rates are a
research option handled by the on-the-fly path (§3b), not by the default trainer.

## 3. Storage for training: a 5 Hz frame store built from the subset

**Decision.** Build `stores/spatialvid-hq-frames5hz-456x256/` (slipstream `ImageBytes` cache) once from the
person_carried_v0 clips, one record per 5 Hz frame, records of a clip contiguous:

| field | type | note |
| --- | --- | --- |
| `image` | JPEG bytes | 456×256, quality **decide** (85 → ~15–20 KB → 175–230 GB total; measure on 1k clips first, also q75 / 384×216) |
| `clip_rec` | int | `record_idx` into the h265 stores (join key for everything clip-level) |
| `k`, `n_k` | int | frame index within the clip on the 5 Hz grid, and the clip's grid length |
| `t_sec` | float | true frame time |
| `pose` | float32[7] | w2c `[tx ty tz qx qy qz qw]` at `t_sec` (needs slipstream fixed-shape fields; fallback: 7 scalars) |
| `intrinsics` | float32[4] | normalized |
| `clip_id`, `source_id`, `channel_id`, `carrier` | str | denormalized for filtering and reporting |
| `fps` | float | source fps (diagnostics only) |

Why a frame store instead of decoding h265 in the loader: (a) throughput — slipstream's JPEG pipeline does
thousands of images/s per node with per-sample seeded augmentation, vs ~100 8 s windows/s of CPU HEVC decode;
(b) determinism — the same frames every epoch, no decoder-version drift, no NVDEC last-frame bug; (c) the 5 Hz
grid is exactly the pose grid. The h265 stores stay the archive: any other rate or resolution is a re-extraction,
never a re-encode. Extraction: PyAV or torchcodec CPU on the fleet, ~11.6 M frames; per-group shards + merge as
for the h265 build (`datasets/prep/spatialvid_hq/extract_frames.py`, to write).

### 3b. On-the-fly path (kept, not default)

`VideoStore.frames_at_seconds(idx, seconds)` over the h265 store already implements "n frames over 8 s from clips
of mixed fps" via `get_frames_played_at`. Use it for rates ≠ 5 Hz, for the demo notebook, and for eval at full
resolution; benchmark NVDEC before making it a training path.

## 4. Loader API

```python
from visionlab.datasets import load
ds = load("spatialvid-hq", split="train", fmt="frames5hz", res="456x256",
          subset="person_carried_v0",
          where="carrier == 'walk'",          # optional pandas query over the subset table
          channel_cap=0.05)                    # optional: max share of clips per channel (random thinning, seeded)
# ds.cache        slipstream OptimizedCache (frame store)
# ds.clips        DataFrame: the selected clips (subset ∩ split ∩ where ∩ cap) with record ranges in the frame store
# ds.stats        normalization stats for the store

sampler = ds.window_sampler(window_s=8.0, rate_hz=5, anchors_per_clip="all", seed=0)
#   valid anchors a: frame records with k + T <= n_k, T = window_s * rate_hz; clips shorter than window_s drop out here
loader = SlipstreamLoader(ds.cache, indices=sampler.anchors(epoch), window=(T, 1), batch_size=64, ...)
for batch in loader:
    batch["image"]      # [B, T, 3, H, W] uint8 (or decoded float after the pipeline), same crop/flip across T
    batch["pose"]       # [B, T, 7]   batch["t_sec"]  # [B, T]
    rel = relative_motion(batch["pose"])   # [B, T-1, 6] camera-t axes, from visionlab.datasets.video
    batch["clip_id"], batch["source_id"], batch["channel_id"], batch["carrier"]
```

`split` is resolved from `splits/<version>.parquet` (clip → split) and intersected with the subset; `split="all"`
gives everything. Image datasets are untouched (`load("imagenet1k", split="val")` unchanged; `res` gets a per-dataset
default as planned in next-steps §1).

Model-side defaults (**decide**, from slipstream2's suggestion): 5 Hz, 4 s warm-up + 4 s prediction = 40 steps,
`window_s = 8` → 144,866 clips, 87 % of frames; alternative 2 s + 2 s keeps 94 % of clips. Nothing about this is
baked into the store.

## 5. Splits before any training run

v1 cannot be used (3.1 % val, no big walking channel in val). Split **v2**: unit = source video, stratify on
carrier × scene × time of day × weather × motion, **and hold out whole channels** for a leave-channel-out val
(the honest test for "generalizes to rain / night", see next-steps §3). Raise or drop the `--max-source-clips`
val-eligibility cap for walk. Report within-subset balance (walk_v0 val fraction, per-dim tables, channel
concentration per stratum). Then `train` = subset ∩ v2-train etc.

## 6. What slipstream needs (request to slipstream2)

1. **Window expansion in `SlipstreamLoader`**: `window=(T, stride)`. `indices` are anchors; the loader reads records
   `a, a+stride, …, a+(T−1)·stride`, returns every field as `[B, T, …]`, repeats the per-sample augmentation seed
   across the T frames (same crop / flip / color for a window), shards and shuffles at anchor level, and
   `warmup_cache(indices=)` warms the expanded set. Acceptance: two runs with the same seed give identical anchor
   order and identical pixels; `[B,40,3,256,456]` at ≥ 500 windows/s on machina from a warm cache.
2. **Fixed-shape array fields** (`float32[7]`, `float32[4]`) in writer + storage, returned as `[B, T, 7]` tensors.
3. `OptimizedCache.build` from a record iterator with image bytes + array fields (works today; confirm with
   contiguous-per-clip ordering and the field types above).
4. Later: `VideoBytes` decode stage (torchcodec, time-based) for the on-the-fly path.

Until 1 lands: visionlab-datasets expands anchors to a flat index list itself, runs the loader with
`shuffle=False` over that list, and reshapes `[B·T] → [B, T]` in `after_batch_transforms` (seeds differ across T,
so augmentation must be off or fixed for that interim).

## 7. Work plan

1. `subsets/person_carried_v0` report + `datasets/prep/spatialvid_hq/make_subset.py` (re-creates the parquet from
   `channels.parquet` + `clips.parquet`, so the definition is code).
2. Frame-store extractor: measure quality/size on 1k clips → **decide** q / res → full extraction on the fleet →
   merge → integrity check → S3 sync.
3. Split v2 (§5) and its report.
4. Registry: `res` axis, video config shape (`stores`, `splits`, `subsets`), `load()` returning `ds.clips` +
   `window_sampler`; `where` / `channel_cap`.
5. Ask slipstream2 for §6; interim flat-index sampler meanwhile.
6. Demo notebook: `[B,40,3,256,456]` batch with trajectories, seed-reproducible, throughput measured.
7. Card: population definition, counts, channel concentration, the 7 unlabelled channels (4.6 % of clips) and
   `mixed` (21k clips) excluded.
