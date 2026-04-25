"""ImageNet-100 dataset configuration with s292_l584 preprocessing.

A 100-class subset of ImageNet-1K, built directly from the raw
``torchvision.datasets.ImageNet`` source with shortest-edge resized to 292
(longest edge capped at 584 to preserve the 2:1 aspect cap).

Each sample contains multiple label fields:
  - label:        native in100 label (0-99)
  - in100_label:  same as label (0-99)
  - in1000_label: original ImageNet-1K label (0-999)
"""
from .imagenet_subsets import IN100_TO_IN1000
from ..registry import DatasetConfig, register

REMOTE_CACHE_BASE = "s3://visionlab-datasets/slipstream-cache/imagenet100"

register(DatasetConfig(
    name="imagenet100_s292",
    num_classes=100,
    remote_cache={
        ("val", "jpeg"): f"{REMOTE_CACHE_BASE}/imagenet100-s292_l584-jpeg-val",
        ("val", "yuv420"): f"{REMOTE_CACHE_BASE}/imagenet100-s292_l584-yuv420-val",
        ("train", "jpeg"): f"{REMOTE_CACHE_BASE}/imagenet100-s292_l584-jpeg-train",
        ("train", "yuv420"): f"{REMOTE_CACHE_BASE}/imagenet100-s292_l584-yuv420-train",
    },
    metadata={
        "num_train": 126_689,
        "num_val": 5_000,
        "preprocessing": "s292_l584_q100",
        "stats": {
            "jpeg": {
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
            "yuv420": {
                "mean": (0.455585, 0.470487, 0.515044),
                "std": (0.264127, 0.068042, 0.064571),
            },
        },
        "in100_to_in1000": IN100_TO_IN1000,
        "label_fields": ["label", "in100_label", "in1000_label"],
    },
))
