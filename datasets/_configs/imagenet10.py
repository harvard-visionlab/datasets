"""ImageNet-10 dataset configuration (10-class subset of ImageNet-1K)."""
from ..registry import DatasetConfig, register

REMOTE_CACHE_BASE = "s3://visionlab-datasets/slipstream-cache/imagenet10"

register(DatasetConfig(
    name="imagenet10",
    num_classes=10,
    remote_cache={
        # TODO: build and upload imagenet10 caches
    },
))
