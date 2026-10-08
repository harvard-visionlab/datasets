"""ProcTHOR walks: egocentric 1,000-step Brownian walks through ProcTHOR-10k houses, rendered in AI2-THOR at 160x120 by
Rupert Tawiah-Quashie (Harvard Vision Lab). Two datasets, one per source cell, with identical houses, seeds and camera
paths (paired walks share clip_id and record_idx):

  procthor-walks-objects   furnished houses
  procthor-walks-empty     the same houses with no furniture

Stores: one per source folder (`<name>-h264-160x120-{train,val,test}`); load() picks the store holding the split
(rtq-r160 train/val both live in the train store). One record = one walk: `video` (mp4, h264 yuv444p crf 10, keyframe every 60 frames), `positions` float32[1000,2]
(metres), `headings` float32[1000] (radians), `fps`/`num_frames`/`duration_s`. The walks have no time axis; 30 fps is a
container convention (frame i at t = i / 30). Built by `datasets/prep/procthor_walks/` (README there).

    ds = load("procthor-walks-objects", split="train")                              # 20,000 walks (v1 = source folders)
    ds = load("procthor-walks-objects", split="train", split_version="rtq-r160")   # the student's 18,000 training walks

Splits: v1 = source folder (train 20,000 / val 2,000 / test 2,000 walks; 10,000 / 1,000 / 1,000 houses, 2 seeds each).
rtq-r160 = what the student's r160 runs used: train 18,000 and val 2,000 (both from the train folder), val_unseen 400
(first 400 val-folder walks), test 2,000, unused 1,600. Training windows: 61 frames at starts 0, 60, ..., 900.
"""
from ..registry import DatasetConfig, register

S3 = "s3://visionlab-datasets/slipstream-cache"

# Normalisation the student's runs used (config.yaml img_* / egomotion_*; computed on each cell's training pool).
# egomotion = (dx_ego, dy_ego, dtheta) between consecutive frames, see the card.
_STATS = {
    "procthor-walks-objects": {
        "rgb": {"mean": (0.4436736617978414, 0.43350608327865603, 0.4364025541114807),
                "std": (0.1886715604682288, 0.18334427017048754, 0.1998636650813531)}},
    "procthor-walks-empty": {
        "rgb": {"mean": (0.45224063844680784, 0.44146551462809247, 0.4426655810769399),
                "std": (0.17919848591594303, 0.17248396495193655, 0.19035107014533156)}},
}
_EGO = {"mean": (6.029963437523e-06, -3.3422506159682e-05, 1.6754411665459e-05),
        "std": (0.10539046576849434, 0.10543259447047094, 0.14997959917859977)}

for _name, _cell in (("procthor-walks-objects", "pt_objects_brownian"), ("procthor-walks-empty", "pt_empty_brownian")):
    register(DatasetConfig(
        name=_name,
        num_classes=0,
        # one store per source folder (train ~48 GB, val/test ~5 GB each); load() picks the one holding the split
        stores={("h264", "160x120", None): {f: f"{S3}/{_name}/{_name}-h264-160x120-{f}" for f in ("train", "val", "test")}},
        splits={"v1": f"{S3}/{_name}/splits/v1.parquet", "rtq-r160": f"{S3}/{_name}/splits/rtq-r160.parquet"},
        subsets={},
        metadata={
            "tree": _name,
            "source_cell": _cell,
            "default_fmt": "h264", "default_res": "160x120", "default_split": "train", "default_split_version": "v1",
            "default_rate_hz": None, "default_subset": None,
            "res_aliases": {"160": "160x120", "120p": "160x120"},
            "num_records": 24_000, "num_train": 20_000, "num_val": 2_000, "num_test": 2_000,
            "num_frames": 1000, "window": {"len": 61, "stride": 60, "starts": list(range(0, 901, 60))},
            "fps_convention": 30,
            "poses": "positions (x, y) metres and headings (radians) per frame; egomotion = step rotated into the "
                     "agent's frame at t plus wrapped heading change (see the card)",
            "stats": {**_STATS[_name], "egomotion": _EGO},
        },
    ))
