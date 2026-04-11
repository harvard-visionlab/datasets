"""visionlab.datasets — Lab dataset registry for slipstream.

Quick start::

    from visionlab.datasets import load
    dataset = load("imagenet1k", split="val")

    from slipstream import SlipstreamLoader
    from slipstream.pipelines import supervised_val
    loader = SlipstreamLoader(dataset, batch_size=256, pipelines=supervised_val(224))
"""
from .version import __version__
# Registry and load function
from .registry import (
    DatasetConfig,
    REGISTRY,
    register,
    list_datasets,
    get_config,
    load,
)

# Platform detection
from .runtime_platform import (
    Platform,
    detect_platform,
    get_platform_cache_dir,
    configure_slipstream_cache,
)

# Preprocessing transforms
from .transforms import ConvertToRGB, ResizeShortCropLong

# Register all built-in dataset configs
from . import _configs  # noqa: F401


def __getattr__(name):
    """Lazy import for backward compatibility with deprecated StreamingDataset."""
    if name == "StreamingDataset":
        from .streaming_dataset import StreamingDatasetVisionlab
        return StreamingDatasetVisionlab
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
