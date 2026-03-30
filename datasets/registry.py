"""Dataset registry and load() function.

Provides a declarative registry of lab datasets with their S3 paths,
metadata, and configuration. The ``load()`` function resolves a dataset
name + split to a ``SlipstreamDataset`` with the correct source URL and
platform-appropriate cache directory.

Usage::

    from visionlab.datasets import load

    dataset = load("imagenet1k", split="val")
    dataset = load("imagenette", split="train")
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
        splits: Mapping of split name to source URL or local path.
        field_types: Mapping of field name to type string (for documentation).
        remote_cache: S3 path for shared slipstream caches (optional).
        source_type: One of "imagefolder", "streaming", "ffcv", "custom".
        metadata: Arbitrary extra metadata (label maps, class lists, etc.).
    """
    name: str
    num_classes: int
    splits: dict[str, str]
    field_types: dict[str, str] = field(default_factory=lambda: {
        "image": "ImageBytes",
        "label": "int",
        "index": "int",
        "path": "str",
    })
    remote_cache: str | None = None
    source_type: str = "streaming"
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


def load(name: str, split: str = "val", **kwargs):
    """Load a registered dataset as a SlipstreamDataset.

    Automatically configures the slipstream cache directory based on the
    detected platform before creating the dataset.

    Args:
        name: Registered dataset name (e.g., "imagenet1k", "imagenette").
        split: Dataset split (e.g., "train", "val").
        **kwargs: Additional arguments passed to SlipstreamDataset.

    Returns:
        A SlipstreamDataset instance ready for use with SlipstreamLoader.
    """
    from slipstream import SlipstreamDataset

    config = get_config(name)

    if split not in config.splits:
        available = ", ".join(config.splits.keys())
        raise KeyError(
            f"Unknown split {split!r} for dataset {name!r}. "
            f"Available: {available}"
        )

    # Set SLIPSTREAM_CACHE_DIR based on detected platform
    configure_slipstream_cache()

    source_url = config.splits[split]

    return SlipstreamDataset(remote_dir=source_url, **kwargs)
