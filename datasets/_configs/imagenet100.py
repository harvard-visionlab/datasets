"""ImageNet-100 dataset configuration (100-class subset of ImageNet-1K)."""
from ..registry import DatasetConfig, register

S3_BASE = "s3://visionlab-datasets/imagenet100/pre-processed"

register(DatasetConfig(
    name="imagenet100",
    num_classes=100,
    splits={
        "train": f"{S3_BASE}/s256-l512-jpgbytes-q100-streaming/train/",
        "val": f"{S3_BASE}/s256-l512-jpgbytes-q100-streaming/val/",
    },
    remote_cache="s3://visionlab-datasets/slipstream-cache/imagenet100/",
    source_type="streaming",
    metadata={
        "preprocessing": "s256_l512_q100",
    },
))
