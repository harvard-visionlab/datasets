"""ImageNet-1K dataset configuration."""
from ..registry import DatasetConfig, register

# S3 bucket: visionlab-datasets
# Pre-processed with s256_l512 (short resize 256, long crop 512, JPEG q100)
S3_BASE = "s3://visionlab-datasets/imagenet1k/pre-processed"

register(DatasetConfig(
    name="imagenet1k",
    num_classes=1000,
    splits={
        "train": f"{S3_BASE}/s256-l512-jpgbytes-q100-streaming/train/",
        "val": f"{S3_BASE}/s256-l512-jpgbytes-q100-streaming/val/",
    },
    remote_cache="s3://visionlab-datasets/slipstream-cache/imagenet1k/",
    source_type="streaming",
    metadata={
        "num_train": 1_281_167,
        "num_val": 50_000,
        "preprocessing": "s256_l512_q100",
    },
))
