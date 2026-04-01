"""Dataset registry and load() function.

Provides a declarative registry of lab datasets with their S3 remote cache
paths and metadata. The ``load()`` function resolves a dataset name + split
+ format to a ``SlipstreamDataset`` backed by a pre-built remote cache.

Usage::

    from visionlab.datasets import load

    dataset = load("imagenet1k", split="val")
    dataset = load("imagenet1k", split="val", fmt="yuv420")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .runtime_platform import configure_slipstream_cache


@dataclass
class DatasetConfig:
    """Configuration for a registered dataset.

    Args:
        name: Short identifier (e.g., "imagenet1k", "imagenette").
        num_classes: Number of classes.
        remote_cache: Mapping of (split, fmt) → S3 remote cache path.
        metadata: Arbitrary extra metadata (label maps, class lists, etc.).
    """
    name: str
    num_classes: int
    remote_cache: dict[tuple[str, str], str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# Global registry
REGISTRY: dict[str, DatasetConfig] = {}


def register(config: DatasetConfig):
    """Register a dataset configuration."""
    REGISTRY[config.name] = config
    return config


def list_datasets():
    """Return names of all registered datasets."""
    return sorted(REGISTRY.keys())


def get_config(name: str) -> DatasetConfig:
    """Get the configuration for a registered dataset."""
    if name not in REGISTRY:
        available = ", ".join(list_datasets()) or "(none)"
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {available}"
        )
    return REGISTRY[name]


def load(name: str, split: str = "val", fmt: str = "jpeg", **kwargs):
    """Load a registered dataset as a SlipstreamDataset.

    Downloads the pre-built cache from S3 if not already present locally.
    Automatically configures the slipstream cache directory based on the
    detected platform.

    Args:
        name: Registered dataset name (e.g., "imagenet1k").
        split: Dataset split (e.g., "train", "val").
        fmt: Image format ("jpeg" or "yuv420"). Default "jpeg".
        **kwargs: Additional arguments passed to SlipstreamDataset.

    Returns:
        A SlipstreamDataset instance ready for use with SlipstreamLoader.
    """
    from pathlib import Path
    from slipstream import SlipstreamDataset
    from slipstream.cache import MANIFEST_FILE

    config = get_config(name)

    key = (split, fmt)
    if key not in config.remote_cache:
        available = [f"split={s}, fmt={f}" for s, f in config.remote_cache.keys()]
        raise KeyError(
            f"No remote cache for {name!r} split={split!r} fmt={fmt!r}. "
            f"Available: {available}"
        )

    # Set SLIPSTREAM_CACHE_DIR based on detected platform
    cache_base = configure_slipstream_cache()

    remote_cache_path = config.remote_cache[key]
    # Derive local cache dir name from the remote path's last component
    cache_name = remote_cache_path.rstrip("/").rsplit("/", 1)[-1]
    local_cache_dir = Path(cache_base) / cache_name

    # Download from S3 if not present locally
    manifest = local_cache_dir / MANIFEST_FILE
    if not manifest.exists():
        from slipstream.s3_sync import download_s3_cache
        print(f"Downloading {name} ({split}, {fmt}) from S3...")
        success = download_s3_cache(
            remote_cache_path,
            local_cache_dir,
        )
        if not success:
            raise RuntimeError(
                f"Failed to download cache from {remote_cache_path}. "
                f"Check your S3 credentials and network connection."
            )

    return SlipstreamDataset(local_dir=str(local_cache_dir), **kwargs)
