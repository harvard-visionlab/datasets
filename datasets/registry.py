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
        remote_cache: Mapping of (split, fmt) → S3 remote cache path (image datasets: one store per split).
        metadata: Arbitrary extra metadata (label maps, class lists, etc.).
        stores: Video datasets: mapping of (fmt, res, fps) → S3 store path; fps None = native frame rate. Splits
            are index sets, not stores.
        splits: Video datasets: split version → S3 parquet (clip_id → split).
        subsets: Video datasets: subset name → S3 parquet (the clip population of a named subset).
    """
    name: str
    num_classes: int
    remote_cache: dict[tuple[str, str], str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    stores: dict[tuple, str] = field(default_factory=dict)
    splits: dict[str, str] = field(default_factory=dict)
    subsets: dict[str, str] = field(default_factory=dict)

    @property
    def is_video(self) -> bool:
        return bool(self.stores)


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


def load(name: str, split: str | None = None, fmt: str | None = None, **kwargs):
    """Load a registered dataset: a SlipstreamDataset (image datasets) or a VideoDataset (video datasets).

    Downloads the pre-built cache from S3 if not already present locally.
    Automatically configures the slipstream cache directory based on the
    detected platform.

    Image datasets (``load("imagenet1k", split="val")``):
        split: Dataset split (default "val").
        fmt: Image format ("jpeg" or "yuv420"). Default "jpeg".
        **kwargs: Additional arguments passed to SlipstreamDataset.

    Video datasets (``load("spatialvid-hq", split="train", subset="person_carried_v0", rate_hz=15)``), see
    :mod:`visionlab.datasets.video_dataset`:
        split: "train" | "val" | "test" | "all" (default from the config). fmt: default "h265".
        res: store resolution (e.g. "456x256", aliases "256p"); rate_hz: sampling rate the loader will use — picks
        the sparsest store whose fps is a multiple of it; fps: pick a store's fps explicitly; subset: named clip
        population; split_version: e.g. "v3"; where: pandas query over the clip table; channel_cap: max share of
        clips per channel (seeded random thinning); seed.

    Returns:
        A SlipstreamDataset instance with normalization stats attached:

        - ``.stats``: ``{"mean": (R, G, B), "std": (R, G, B)}`` for the
          decoded RGB output (the common decode path), or ``None`` if stats
          have not been computed yet. These are correct regardless of the
          storage ``fmt`` because every ``Decode*`` path emits RGB.
        - ``.stats_by_colorspace``: per-colorspace stats, e.g. ``"rgb"`` and
          ``"yuv420"``. The ``"yuv420"`` entry is only correct for consumers
          that keep YUV planes (the niche ``DecodeYUV*`` decoders).
    """
    from pathlib import Path
    from slipstream import SlipstreamDataset
    from slipstream.cache import MANIFEST_FILE

    config = get_config(name)
    if config.is_video:
        from .video_dataset import load_video
        return load_video(config, split=split, fmt=fmt, **kwargs)
    split = split or "val"; fmt = fmt or "jpeg"

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

    dataset = SlipstreamDataset(local_dir=str(local_cache_dir), **kwargs)

    # Normalization stats depend on the DECODED colorspace (chosen by the
    # decoder/pipeline downstream), not the storage `fmt`. Every Decode* path
    # emits RGB; only DecodeYUV* keeps YUV planes. Expose all colorspaces and
    # default `.stats` to rgb (the common decode path).
    all_stats = config.metadata.get("stats", {})
    dataset.stats_by_colorspace = all_stats
    dataset.stats = all_stats.get("rgb")

    return dataset
