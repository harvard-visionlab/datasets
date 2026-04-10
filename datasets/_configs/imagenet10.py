"""ImageNet-10 dataset configuration (10-class subset of ImageNet-100/1K).

Each sample contains multiple label fields:
  - label:        native in10 label (0-9)
  - in10_label:   same as label (0-9)
  - in100_label:  ImageNet-100 label (0-99)
  - in1000_label: original ImageNet-1K label (0-999)
"""
from .imagenet_subsets import IN10_TO_IN1000, IN10_TO_IN100
from ..registry import DatasetConfig, register

REMOTE_CACHE_BASE = "s3://visionlab-datasets/slipstream-cache/imagenet10"

register(DatasetConfig(
    name="imagenet10",
    num_classes=10,
    remote_cache={
        ("val", "jpeg"): f"{REMOTE_CACHE_BASE}/imagenet10-s256_l512-jpeg-val",
        ("val", "yuv420"): f"{REMOTE_CACHE_BASE}/imagenet10-s256_l512-yuv420-val",
        ("train", "jpeg"): f"{REMOTE_CACHE_BASE}/imagenet10-s256_l512-jpeg-train",
        ("train", "yuv420"): f"{REMOTE_CACHE_BASE}/imagenet10-s256_l512-yuv420-train",
    },
    metadata={
        "num_train": 12_964,
        "num_val": 500,
        "preprocessing": "s256_l512_q100",
        "in10_to_in1000": IN10_TO_IN1000,
        "in10_to_in100": IN10_TO_IN100,
        "label_fields": ["label", "in10_label", "in100_label", "in1000_label"],
    },
))
