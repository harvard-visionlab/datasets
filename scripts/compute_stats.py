"""Compute image normalization stats for all dataset caches.

Prints mean/std values for each dataset x format combination,
formatted for pasting into config metadata.

Usage::

    uv run python scripts/compute_stats.py
"""
from pathlib import Path

from slipstream import compute_normalization_stats
from slipstream.cache import OptimizedCache
from visionlab.datasets.runtime_platform import get_platform_cache_dir

cache_base = Path(get_platform_cache_dir())
print(f"Cache base: {cache_base}")

DATASETS = {
    "imagenet1k": "imagenet1k-s256_l512-{fmt}-val",
    "imagenet100": "imagenet100-s256_l512-{fmt}-val",
    "imagenet10": "imagenet10-s256_l512-{fmt}-val",
}

FORMATS = ["jpeg", "yuv420"]

for ds_name, pattern in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"  {ds_name}")
    print(f"{'='*60}")
    for fmt in FORMATS:
        cache_dir = cache_base / pattern.format(fmt=fmt)
        if not cache_dir.exists():
            print(f"\n  {fmt}: SKIPPED (not found at {cache_dir})")
            continue

        cache = OptimizedCache.load(cache_dir, verbose=False)
        colorspace = "yuv" if fmt == "yuv420" else "rgb"
        stats = compute_normalization_stats(
            cache, image_format=fmt, colorspace=colorspace, verbose=True,
        )
        mean = tuple(round(v, 6) for v in stats["mean"])
        std = tuple(round(v, 6) for v in stats["std"])
        print(f"\n  {fmt}:")
        print(f"    mean: {mean}")
        print(f"    std:  {std}")
