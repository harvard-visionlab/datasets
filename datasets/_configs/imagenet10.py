"""ImageNet-10 dataset configuration (10-class subset of ImageNet-1K)."""
from ..registry import DatasetConfig, register

S3_BASE = "s3://visionlab-datasets/imagenet10/pre-processed"

register(DatasetConfig(
    name="imagenet10",
    num_classes=10,
    splits={
        "train": f"{S3_BASE}/s256-l512-jpgbytes-q100-streaming/train/",
        "val": f"{S3_BASE}/s256-l512-jpgbytes-q100-streaming/val/",
    },
    remote_cache="s3://visionlab-datasets/slipstream-cache/imagenet10/",
    source_type="streaming",
    metadata={
        "preprocessing": "s256_l512_q100",
    },
))
