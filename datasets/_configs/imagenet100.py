"""ImageNet-100 dataset configuration (100-class subset of ImageNet-1K)."""
from ..registry import DatasetConfig, register

REMOTE_CACHE_BASE = "s3://visionlab-datasets/slipstream-cache/imagenet100"

register(DatasetConfig(
    name="imagenet100",
    num_classes=100,
    remote_cache={
        # TODO: build and upload imagenet100 caches
    },
))
