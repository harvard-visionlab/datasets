"""ProcTHOR routes: scripted egocentric camera routes (goal-directed walks, full turns, out-and-back, loops, Brownian
controls) through ProcTHOR-10k houses, rendered in AI2-THOR 5.0.0 at 160x120 with the procthor-walks camera. Two
datasets, one per cell, with identical houses and camera paths (paired windows share clip_id and record_idx):

  procthor-routes-objects   furnished houses
  procthor-routes-empty     the same houses with no furniture

split = ProcTHOR's house split (val: val houses 0-199, the procthor-walks val_unseen houses; train: later, train houses
only). subset = <route>_<T>, T = 61 or 121 window frames (20 warm-up + rollout + 1 target):

    ds = load("procthor-routes-objects", split="val", subset="full_turn_61")     # fmt="lossless" is the only store
    idx, t0 = ds.eval_windows(T=61)                                              # one window per record, from frame 0

One store per (cell, split) holding both lengths: `<name>-lossless-160x120-<split>`. One record = one window of T+4
frames (4 held frames serve the decoder margin): `video` (mp4, H.264 RGB lossless, byte-exact renders), `positions`
float64[125,2] and `headings` float64[125] (Habitat convention, as procthor-walks; NaN beyond num_frames),
`fps`/`num_frames`/`duration_s`. records.parquet adds subset, route, window_frames, house, event frames, turn rate,
inertia p. Built by `datasets/prep/procthor_routes/` (README there); card `cards/procthor-routes.md`.
"""
from ..registry import DatasetConfig, register
from .procthor_walks import _EGO, _STATS

S3 = "s3://visionlab-datasets/slipstream-cache"
ROUTES = ("goal_directed", "full_turn", "out_and_back", "loop", "brownian", "brownian_inertia80", "brownian_inertia95")
SUBSETS = tuple(f"{r}_{t}" for r in ROUTES for t in (61, 121))
SPLITS = ("val",)                 # train: later

for _cell in ("objects", "empty"):
    _name = f"procthor-routes-{_cell}"
    register(DatasetConfig(
        name=_name,
        num_classes=0,
        stores={("lossless", "160x120", None): {s: f"{S3}/{_name}/{_name}-lossless-160x120-{s}" for s in SPLITS}},
        splits={"v1": f"{S3}/{_name}/splits/v1.parquet"},
        subsets={s: f"{S3}/{_name}/subsets/{s}.parquet" for s in SUBSETS},
        metadata={
            "tree": _name,
            "default_fmt": "lossless", "default_res": "160x120", "default_split": "val", "default_split_version": "v1",
            "default_rate_hz": None, "default_subset": None,
            "res_aliases": {"160": "160x120", "120p": "160x120"},
            "routes": list(ROUTES), "window_frames": [61, 121], "warmup_frames": 20, "pad_frames": 4,
            "fps_convention": 30,
            "poses": "positions (x, y) metres and headings (radians) per frame, Habitat convention as procthor-walks; "
                     "NaN beyond num_frames",
            # normalisation = procthor-walks' training statistics of the same cell (eval inputs normalised as in training)
            "stats": {**_STATS[f"procthor-walks-{_cell}"], "egomotion": _EGO},
        },
    ))
