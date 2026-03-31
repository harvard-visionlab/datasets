"""Imagenette dataset configuration (10-class subset, fast-download friendly).

Label mapping to ImageNet-1K class indices:
  0:tench(0), 1:English springer(217), 2:cassette player(482),
  3:chain saw(491), 4:church(497), 5:French horn(566),
  6:garbage truck(569), 7:gas pump(571), 8:golf ball(574), 9:parachute(701)
"""
from ..registry import DatasetConfig, register

REMOTE_CACHE_BASE = "s3://visionlab-datasets/slipstream-cache/imagenette"

register(DatasetConfig(
    name="imagenette",
    num_classes=10,
    remote_cache={
        # TODO: build and upload imagenette caches
    },
    metadata={
        "imagenet_label_map": {
            0: 0, 1: 217, 2: 482, 3: 491, 4: 497,
            5: 566, 6: 569, 7: 571, 8: 574, 9: 701,
        },
    },
))
