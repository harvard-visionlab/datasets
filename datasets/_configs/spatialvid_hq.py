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
    subsets={"person_carried_v0": f"{S3}/subsets/person_carried_v0.parquet"},
    metadata={
        "tree": "SpatialVID-HQ-slipstream",          # directory name under the lab's shared DataSets/VideoDatasets tree
        "default_fmt": "h265", "default_res": "456x256", "default_split": "train", "default_split_version": "v3",
        "default_rate_hz": 15,
        "res_aliases": {"640": "640x360", "360p": "640x360", "456": "456x256", "256p": "456x256"},
        "num_records": 365_296,
        "num_train": 186_710, "num_val": 15_029, "num_test": 11_364,           # person_carried_v0 ∩ v3
        "poses": "world->camera [tx ty tz qx qy qz qw], OpenCV axes, non-metric scale; ~5 Hz, interpolate with ds.poses_at",
        "stats": {},                                                            # per-store rgb stats: not computed yet
    },
))
