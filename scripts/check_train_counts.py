"""Print train sample counts for subset caches."""
from pathlib import Path
from slipstream.cache import OptimizedCache
from visionlab.datasets.runtime_platform import get_platform_cache_dir

cache_base = Path(get_platform_cache_dir())
for name in ["imagenet100-s256_l512-jpeg-train", "imagenet10-s256_l512-jpeg-train"]:
    c = OptimizedCache.load(cache_base / name, verbose=False)
    print(f"{name}: {c.num_samples} samples")
