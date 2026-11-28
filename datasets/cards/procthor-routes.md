# ProcTHOR routes (`procthor-routes-objects`, `procthor-routes-empty`)

Short first-person videos of a camera taking **scripted routes through simulated houses**: walking from place to
place, turning a full circle, walking out and coming back, walking a loop, and wandering at random. Every clip is
160×120 at eye height, with the camera's exact position and heading at every frame. As in
[procthor-walks](procthor-walks.md), each clip exists twice: in `procthor-routes-objects` the house is furnished, and
in `procthor-routes-empty` it is the *same* house with every object removed, on the *same* camera path.

It is an evaluation set for world models trained on procthor-walks (random Brownian walks through the same kind of
houses). The routes test what a random walk never asks for: remembering what is behind you after a full turn, returning to
a view you left, closing a loop. Brownian clips are included as a control. The camera, renderer and houses are the
training ones, so only the motion differs. 13,602 clips per dataset come from the 200 held-out val houses that are
procthor-walks' `val_unseen` houses: 7 route types × 2 clip lengths. Training sets with the same route types will
follow.

Rendered by the lab (2026-11-28) with Rupert Tawiah-Quashie's procthor-walks camera, AI2-THOR 5.0.0 and the
ProcTHOR-10k houses.

## Source

**ProcTHOR-10k** (Deitke et al., 2022): 10,000 train / 1,000 val / 1,000 test procedurally generated houses (1 to 10
rooms), rendered by **AI2-THOR** (Kolve et al., 2017), both Apache-2.0. We keep ProcTHOR's house split: the val set
uses val houses 0–199, and later training sets use only train houses. Citations are in the
[procthor-walks card](procthor-walks.md#source).

| | |
| --- | --- |
| houses | 200 ProcTHOR-10k val houses (0–199; 51 of them have one room) |
| camera | as procthor-walks: 160×120 px, 90° horizontal / 73.7° vertical FOV, 1.454 m high, agent invisible, physics paused, no gait or head bob |
| clip | **T = 61** frames (20 warm-up + 40 rollout + 1 target) or **T = 121** (20 + 100 + 1), plus 4 held frames at the end |
| clips per house | up to 5 per route type and length |
| lighting | as shipped: Midday 76 %, GoldenHour 19 %, BlueHour 5 % of clips |
| renderer | Kempner A100, ai2thor 5.0.0 CloudRendering, in the procthor-walks generator's environment |

## The routes

Every scripted clip starts the same way. For the 20 warm-up frames the camera walks along a path to a point A that
has room for the event. The event starts at A at frame 20, the first rollout frame, and afterwards the camera keeps
walking (goal to goal) until the clip ends. Speeds stay within the training data's range: walking pace is 0.115 m per
frame (the training median), and turns are at most 0.15 rad per frame (68.6 % of training turns are that small).
The 61-frame turn, loop and out-and-back pivot need 0.157 rad per frame to fit in 40 frames; 72 % of training turns
are ≤ 0.16 rad.

| subset (`_61` / `_121`) | what the camera does | clips 61 / 121 |
| --- | --- | --- |
| `goal_directed` | walks from goal to goal along smooth shortest paths (legs 2–5 m) | 1,000 / 1,000 |
| `full_turn` | turns 360° on a tight 0.33 m circle (0.05 m forward per frame) | 980 / 979 |
| `out_and_back` | walks out along a path (61: ~1 m; 121: 2–3 m), pivots 180° slowly (0.02 m per frame), retraces the path to A | 985 / 959 |
| `loop` | walks one full circle at walking pace (61: radius 0.73 m; 121: 0.77–1.28 m) and returns to A | 854 / 845 |
| `brownian` | the procthor-walks training motion: Gaussian 0.1 m steps in any direction, Gaussian 0.15 rad turns | 1,000 / 1,000 |
| `brownian_inertia80` | Brownian, but each frame repeats the previous step (dx, dz, dθ) with probability 0.8 | 1,000 / 1,000 |
| `brownian_inertia95` | same with probability 0.95 | 1,000 / 1,000 |

Counts fall below 1,000 where a house has no room for the event. One-room houses mostly can't fit a loop circle.
Every planner attempt is counted in the prep output.

Motion per clip, averaged over each subset (61 / 121):

| subset | path length (m) | farthest from A (m) | steps < 0.032 m |
| --- | --- | --- | --- |
| goal_directed | 3.4 / 8.7 | 2.6 / 4.2 | 3 % / 4 % |
| full_turn | 2.0 / 7.0 | 0.6 / 3.3 | 0 % / 2 % |
| out_and_back | 2.3 / 8.3 | 1.1 / 2.9 | 33 % / 18 % (the slow pivot) |
| loop | 4.6 / 9.9 | 1.5 / 3.2 | 0 % / 2 % |
| brownian | 5.2 / 13.0 | 1.0 / 1.5 | 7 % / 7 % |
| brownian_inertia80 | 4.8 / 12.0 | 1.6 / 2.4 | 9 % / 9 % |
| brownian_inertia95 | 4.5 / 11.1 | 1.7 / 2.6 | 12 % / 13 % |

The inertia variants keep training-sized steps but correlate them in time, so the camera drifts farther instead of
jittering in place. A step that hits a wall ends the repeated run; those wall steps are why near-zero steps become
more common as p grows.

## What our build adds

**Format.** One [slipstream](https://github.com/harvard-visionlab/slipstream) store per dataset holds the whole val
split, both lengths: `procthor-routes-<cell>-lossless-160x120-val`, 13,602 records, 18.0 GB (furnished) / 15.6 GB (empty).
Frames are **byte-exact** renders (H.264 RGB lossless), not a compressed approximation. Each clip is verified, frame
for frame, against the renderer output.

**Pairing.** Both datasets have the same `clip_id`s in the same order (so the same `record_idx`) and byte-identical
positions and headings (verified).

**Subsets.** `split="val"`; `subset=` picks the route and length (14 subsets). Always pass a subset (see Usage).

**Not bit-identical to procthor-walks.** AI2-THOR rendering isn't exactly reproducible, even with the original script.
Re-rendering stored procthor-walks frames on an A100 gives ±1 grey level noise, plus a few isolated pixels, with the
original generator and with ours alike (70–100 % of frames within ±1 level). These clips are therefore not pixel-for-pixel
what the training renderer would have produced, but the difference is at the level of renderer noise.
An RTX 4090 differed more (7–10 % of pixels), so all clips were rendered on A100s.

## What a record looks like

`load("procthor-routes-objects", split="val", subset="full_turn_61")`, one record (`clip_id`
`val/house_00042/full_turn_61_0`):

| field / column | value |
| --- | --- |
| `video` | mp4 bytes, 65 frames (61 + 4 held), 160×120 RGB, H.264 RGB lossless, ~1.3 MB per clip on average |
| `positions` | float64 `[125, 2]`: Habitat (x, y) in metres per frame; rows 65–124 are NaN (61-frame clip) |
| `headings` | float64 `[125]`: radians per frame, NaN beyond `num_frames` |
| `fps` / `num_frames` / `duration_s` | 30 (a container convention: no time axis, a frame is a step) / 65 / 2.17 |
| `subset`, `route`, `window_frames` | `full_turn_61`, `full_turn`, 61 |
| `house_idx`, `n_rooms`, `n_objects`, `time_of_day` | 42, 10, 63, GoldenHour |
| `event_start`, `event_end` | 20, 60: the turn runs from frame 20 to 60 (empty for Brownian clips) |
| `turn_rate`, `p_keep` | 0.157 rad/frame, — (inertia probability for Brownian-with-inertia clips) |
| `info`, `metrics` | JSON: event geometry (start point A, loop radius, out distance, …) and per-clip motion metrics |

Positions and headings use the same convention as procthor-walks, so its egomotion code applies unchanged.

## Usage

```python
from visionlab.datasets import load
ds = load("procthor-routes-objects", split="val", subset="full_turn_61")      # or procthor-routes-empty
recs, t0 = ds.eval_windows(T=61, rate_hz=30)                                    # one window per clip, from frame 0
```

**Why `T=61` with `rate_hz=30`:** a clip holds T + 4 frames, so `eval_windows(T)` fits exactly one window, starting
at frame 0 (`t0 = 0.5 / 30`), the same decode path and normalisation as procthor-walks. Use the `_121` subsets with
`T=121`. Without a subset, `T=61` would also find 61-frame windows inside the 121-frame clips at other starts, so
always pick a subset.

Decode with `DecodeVideoWindow(T=T, rate_hz=30, t0_key="t0")`, as in the procthor-walks card. RGB normalisation
(`ds.stats`) is the procthor-walks training statistics of the same cell, so eval inputs are normalised exactly as in
training. For paired furnished/empty analyses, load both datasets with the same subset and index the same
`record_idx`.

## Methods paragraph

> To evaluate {world models} on structured camera motion, we rendered scripted routes through 200 held-out ProcTHOR-10k
> validation houses (Deitke et al., 2022) in AI2-THOR 5.0.0 (Kolve et al., 2017), with the camera of our training
> walks (160×120 px, 90° horizontal field of view, 1.454 m above the floor). Routes were planned on each house's
> walkable grid at the training data's walking pace (0.115 m per frame) and turn rate (≤ 0.15 rad per frame, 0.157 for
> 61-frame turns, loops and pivots). Each clip began with a 20-frame warm-up walk to the event start, followed by
> {40 | 100} rollout frames: goal-directed walks, a 360° turn, out-and-back (walk out, turn 180°, retrace), one loop,
> Brownian motion as in training, and Brownian motion with inertia (each step repeated with probability 0.8 or 0.95).
> Each clip was rendered in the furnished house and in the same house with all objects removed, along an identical
> camera path ({n} clips per condition).

## Build

`datasets/prep/procthor_routes/` (README there): planner, renderer port, render-equivalence check, store build,
verification and publishing. Stores: netscratch `Lab/datasets/slipstream`, lab_storage master, and private S3
(owner-only; ProcTHOR assets are Apache-2.0, but the lab keeps these renders private like procthor-walks).
