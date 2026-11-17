"""SpatialVID-HQ (365,296 clips, 2–15 s, YouTube walking/driving/drone footage with MegaSaM camera poses at ~5 Hz).

Video dataset: h265 slipstream stores per (fmt, res, fps); splits and subsets are parquet index sets over clip_id.
Built by `datasets/prep/spatialvid_hq/` (README there = data model); design in docs/plans/spatialvid-hq-*.md.

    ds = load("spatialvid-hq", split="train", subset="person_carried_v0", rate_hz=15)   # -> VideoDataset

Splits (v3, 2026-09-17, unit = whole YouTube video / whole channel, stratified on scene × time-of-day × weather ×
crowd × motion × carrier over the person_carried_v0 population):
  train 186,710 clips · val 15,029 (held-out whole videos of channels seen in training; in-distribution, ≤ 1.2 pp
  off train per stratum) · test 11,364 (7 whole channels never seen in training; the new-channel transfer metric).
Stores: native fps (the archive; 60/30/24/50/25 fps sources), 30 fps and 15 fps (integer decimation, exact
timestamps, `src_fps` / `src_num_frames` fields). `rate_hz` picks the sparsest store whose fps divides evenly.
"""
from ..registry import DatasetConfig, register

S3 = "s3://visionlab-datasets/slipstream-cache/spatialvid-hq"

_EGO = {
    "person_carried_v0": {"rate_hz": 15, "mean": (-3.30169e-05, -3.14994e-05, 0.00749829, -8.14747e-07, 6.6549e-05, 1.14942e-05),
                          "std": (0.00406336, 0.00164502, 0.00704761, 0.00187087, 0.00708602, 0.00119285)},
    "person_walking_v0": {"rate_hz": 15, "mean": (-3.07235e-05, -3.88232e-05, 0.00891854, 3.1888e-06, 9.06934e-05, 1.20771e-05),
                          "std": (0.00435559, 0.00175941, 0.00680596, 0.00171853, 0.00661057, 0.00103231)},
}

register(DatasetConfig(
    name="spatialvid-hq",
    num_classes=0,
    stores={
        ("h265", "640x360", None): f"{S3}/spatialvid-hq-h265-640x360",
        ("h265", "456x256", None): f"{S3}/spatialvid-hq-h265-456x256",
        ("h265", "640x360", 30): f"{S3}/spatialvid-hq-h265-640x360-30fps",
        ("h265", "456x256", 30): f"{S3}/spatialvid-hq-h265-456x256-30fps",
        ("h265", "640x360", 15): f"{S3}/spatialvid-hq-h265-640x360-15fps",
        ("h265", "456x256", 15): f"{S3}/spatialvid-hq-h265-456x256-15fps",
    },
    splits={"v1": f"{S3}/splits/v1.parquet", "v3": f"{S3}/splits/v3.parquet"},
    subsets={"person_carried_v0": f"{S3}/subsets/person_carried_v0.parquet",
             # person_carried_v0 minus clips that mostly stand still or spin in place (prep/spatialvid_hq/motion_subset.py);
             # carries the per-clip motion index columns (moving_frac, spin_frac, step_p50, fwd_frac, turn_dps, ...) for where=
             "person_walking_v0": f"{S3}/subsets/person_walking_v0.parquet"},
    metadata={
        "tree": "SpatialVID-HQ-slipstream",          # directory name under the lab's shared DataSets/VideoDatasets tree
        "default_fmt": "h265", "default_res": "456x256", "default_split": "train", "default_split_version": "v3",
        "default_rate_hz": 15,
        # The split table labels every index clip (365,362 rows incl. 66 `excluded`); the training *population* is the
        # subset (`in_subset`). load() applies it by default; pass subset="all" for the whole store.
        "default_subset": "person_carried_v0",
        # methods paragraph for papers (model-rearing `mr methods`: {"text", "refs"}; text is str.format'ed with this
        # metadata, so no braces). Run-specific details (subset used, window, store, stats) are written by the harness.
        "methods": {
            "text": (
                "SpatialVID-HQ: 365,362 clips of up to 15 s (1,112 hours) cut from 22,543 YouTube videos of real-world "
                "scenes filmed by a moving camera, each annotated with camera poses estimated at ~5 Hz by MegaSaM, a "
                "structure-and-motion method for casual videos (poses are consistent within and across clips but not "
                "metric). Annotations are released under CC BY-NC-SA 4.0 for research use; the videos remain their "
                "uploaders'. We used the visionlab-datasets curation of it: split v3, which assigns whole source videos "
                "to one split, stratified over scene type, time of day, weather, crowd density, motion tags and camera "
                "carrier (validation holds out whole videos from training channels; test holds out seven entire "
                "channels never seen in training); the person_carried_v0 subset, 213,103 clips from 82 channels in "
                "which the camera is carried by a walking person (hand-held, head-mounted or on a stabiliser), assigned "
                "by manual review of each channel; and its person_walking_v0 subset, which further drops clips where "
                "the camera moves (at least 0.04 pose units/s, about 0.3 m/s) for less than half of the clip or rotates "
                "faster than 30 deg/s for 10% of it or more (177,085 clips: 154,799 train, 12,521 validation, 9,765 "
                "test). Clips were re-encoded from 1280 x 720 pixels to 456 x 256 or 640 x 360, keeping every frame or "
                "every k-th frame (integer decimation), and poses were interpolated to frame times (linear translation, "
                "spherical-linear rotation)."),
            "refs": [
                {"cite": "Wang et al., 2025",
                 "full": "Wang, J., et al. (2025). SpatialVID: A large-scale video dataset with spatial annotations. "
                         "arXiv:2509.09676.",
                 "doi": "10.48550/arXiv.2509.09676"},
                {"cite": "Li et al., 2025",
                 "full": "Li, Z., Tucker, R., Cole, F., Wang, Q., Jin, L., Ye, V., Kanazawa, A., Holynski, A., & Snavely, "
                         "N. (2025). MegaSaM: Accurate, fast and robust structure and motion from casual dynamic videos. "
                         "In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).",
                 "doi": "10.48550/arXiv.2412.04463"},
            ],
        },
        "res_aliases": {"640": "640x360", "360p": "640x360", "456": "456x256", "256p": "456x256"},
        "num_records": 365_296,
        "num_train": 186_710, "num_val": 15_029, "num_test": 11_364,           # person_carried_v0 ∩ v3
        "poses": "world->camera [tx ty tz qx qy qz qw], OpenCV axes, non-metric scale; ~5 Hz, interpolate with ds.poses_at",
        # Per-store RGB frame stats on [0, 1] (prep/spatialvid_hq/stats.py, 2026-09-18: 4,000 (456x256) / 2,000 (640x360)
        # seeded 8 s train-population windows at 15 Hz, decoder resize 224, exact uint8 histogram on a 4x4 stride).
        # All six agree to ~0.005; "rgb" is the fallback for any store not listed. Source JSON: <tree>/index/stats_*.json.
        "stats": {
            "rgb": {"mean": (0.434, 0.426, 0.407), "std": (0.231, 0.228, 0.251)},
            "spatialvid-hq-h265-456x256":       {"mean": (0.436378, 0.427394, 0.409534), "std": (0.230930, 0.228455, 0.251683)},
            "spatialvid-hq-h265-456x256-30fps": {"mean": (0.434145, 0.426447, 0.408044), "std": (0.229374, 0.226503, 0.249886)},
            "spatialvid-hq-h265-456x256-15fps": {"mean": (0.431815, 0.424298, 0.406280), "std": (0.230099, 0.227918, 0.251444)},
            "spatialvid-hq-h265-640x360":       {"mean": (0.434176, 0.425239, 0.404285), "std": (0.231950, 0.229325, 0.251976)},
            "spatialvid-hq-h265-640x360-30fps": {"mean": (0.432961, 0.424706, 0.404042), "std": (0.231258, 0.228758, 0.250860)},
            "spatialvid-hq-h265-640x360-15fps": {"mean": (0.432552, 0.426263, 0.406928), "std": (0.232041, 0.229050, 0.251962)},
            # Ego-motion z-score stats (prep/spatialvid_hq/ego_stats.py, 2026-11-16): `ds.ego_motion_at` deltas
            # [dx dy dz rx ry rz] (camera-t axes; translation in raw MegaSaM pose units, ~8 m per unit at walking pace,
            # no per-clip rescale; rotation in rad) between consecutive frames at 15 Hz, over 4 epochs of seeded 61-frame
            # train windows. Deltas scale with the frame interval: valid at rate_hz=15 only. `ds.ego_stats` picks the
            # entry for the loaded subset. Source JSON: <tree>/index/ego_stats_<subset>.json.
            "egomotion": _EGO["person_carried_v0"],
            "egomotion_subsets": _EGO,
        },
    },
))
