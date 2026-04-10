"""Upload ImageNet-100 and ImageNet-10 train caches to S3.

Usage::

    uv run python scripts/sync_train_to_s3.py
"""
from pathlib import Path

from visionlab.datasets.runtime_platform import get_platform_cache_dir
from slipstream.s3_sync import upload_s3_cache

S3_BASE = "s3://visionlab-datasets/slipstream-cache"

CACHES = [
    ("imagenet100-s256_l512-jpeg-train", "imagenet100"),
    ("imagenet100-s256_l512-yuv420-train", "imagenet100"),
    ("imagenet10-s256_l512-jpeg-train", "imagenet10"),
    ("imagenet10-s256_l512-yuv420-train", "imagenet10"),
]

cache_base = Path(get_platform_cache_dir())
print(f"Cache base: {cache_base}")

for cache_name, subset in CACHES:
    local = cache_base / cache_name
    remote = f"{S3_BASE}/{subset}/{cache_name}"
    if not local.exists():
        print(f"\nSkipping {cache_name} (not found at {local})")
        continue
    print(f"\nUploading {cache_name} -> {remote}")
    upload_s3_cache(local, remote)
    print(f"Done: {cache_name}")
