"""ImageNet-100 dataset configuration (100-class subset of ImageNet-1K).

Each sample contains multiple label fields:
  - label:        native in100 label (0-99)
  - in100_label:  same as label (0-99)
  - in1000_label: original ImageNet-1K label (0-999)
"""
from .imagenet_subsets import IN100_TO_IN1000
from ..registry import DatasetConfig, register

REMOTE_CACHE_BASE = "s3://visionlab-datasets/slipstream-cache/imagenet100"

register(DatasetConfig(
    name="imagenet100",
    num_classes=100,
    remote_cache={
        ("val", "jpeg"): f"{REMOTE_CACHE_BASE}/imagenet100-s256_l512-jpeg-val",
        ("val", "yuv420"): f"{REMOTE_CACHE_BASE}/imagenet100-s256_l512-yuv420-val",
        ("train", "jpeg"): f"{REMOTE_CACHE_BASE}/imagenet100-s256_l512-jpeg-train",
        ("train", "yuv420"): f"{REMOTE_CACHE_BASE}/imagenet100-s256_l512-yuv420-train",
    },
    metadata={
        "num_train": 126_689,
        "num_val": 5_000,
        "preprocessing": "s256_l512_q100",
        "in100_to_in1000": IN100_TO_IN1000,
        "label_fields": ["label", "in100_label", "in1000_label"],
    },
))
