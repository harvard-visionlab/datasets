"""ImageNet-1K dataset configuration."""
from ..registry import DatasetConfig, register

REMOTE_CACHE_BASE = "s3://visionlab-datasets/slipstream-cache/imagenet1k"

register(DatasetConfig(
    name="imagenet1k",
    num_classes=1000,
    remote_cache={
        ("val", "jpeg"): f"{REMOTE_CACHE_BASE}/imagenet1k-s256_l512-jpeg-val",
        ("val", "yuv420"): f"{REMOTE_CACHE_BASE}/imagenet1k-s256_l512-yuv420-val",
        ("train", "jpeg"): f"{REMOTE_CACHE_BASE}/imagenet1k-s256_l512-jpeg-train",
        ("train", "yuv420"): f"{REMOTE_CACHE_BASE}/imagenet1k-s256_l512-yuv420-train",
    },
    metadata={
        "num_train": 1_281_167,
        "num_val": 50_000,
        "preprocessing": "s256_l512_q100",
    },
))
