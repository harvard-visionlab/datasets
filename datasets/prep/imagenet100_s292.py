"""Build the ImageNet-100 slipstream cache directly from raw ImageNet-1K
(``torchvision.datasets.ImageNet``) with s292_l584 preprocessing.

Unlike :mod:`datasets.prep.imagenet_subsets`, which filters an existing
``imagenet1k-s256_l512`` cache, this script reads the raw ImageNet-1K
distribution (``train/`` and ``val/`` directories) and applies a fresh
preprocessing pass: resize shortest edge to 292, center-crop the longest
edge to 584, then re-encode as JPEG quality 100. Only samples belonging
to the canonical IN100 classes are kept. Both standalone JPEG and
standalone YUV420 caches can be built.

Usage::

    # Build val (jpeg)
    python -m datasets.prep.imagenet100_s292 \\
        --root /path/to/imagenet --split val

    # Build train (jpeg, parallel)
    python -m datasets.prep.imagenet100_s292 \\
        --root /path/to/imagenet --split train --num-workers 16

    # Build all splits and formats
    python -m datasets.prep.imagenet100_s292 \\
        --root /path/to/imagenet --all --num-workers 16
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch
from torchvision.datasets import ImageNet
from torchvision.io import decode_image, encode_jpeg
from torchvision.transforms.functional import center_crop, resize

from .._configs.imagenet_subsets import IN100_TO_IN1000, IN1000_TO_IN100


__all__ = ["ImageNet100_s292_l584", "build"]


SHORT_SIZE = 292
LONG_SIZE = 584
PREPROCESSING_TAG = f"s{SHORT_SIZE}_l{LONG_SIZE}"

# Expected post-filter sample counts for the IN100 subset.
EXPECTED_COUNTS = {
    "train": 126_689,
    "val": 5_000,
}


def _resize_short_crop_long(
    img: torch.Tensor,
    short_size: int = SHORT_SIZE,
    long_size: int = LONG_SIZE,
) -> torch.Tensor:
    """Resize shortest edge to ``short_size`` and center-crop longest edge to ``long_size``."""
    img = resize(img, short_size, antialias=True)
    _, h, w = img.shape
    crop_h = min(h, long_size)
    crop_w = min(w, long_size)
    if crop_h < h or crop_w < w:
        img = center_crop(img, [crop_h, crop_w])
    return img


class ImageNet100_s292_l584(ImageNet):
    """ImageNet-100 with s292_l584 preprocessing, built from raw ImageNet-1K.

    Subclasses ``torchvision.datasets.ImageNet`` for label consistency, then
    filters to the canonical IN100 classes. ``__getitem__`` returns a dict
    with the image (preprocessed JPEG bytes), three label fields (``label``,
    ``in100_label``, ``in1000_label``), the relative path, and the new index.

    Args:
        root: Path to ImageNet root (contains ``train/`` and ``val/`` dirs).
        split: ``"train"`` or ``"val"``.
        jpeg_quality: JPEG encoding quality (1-100). Default 100.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "val",
        jpeg_quality: int = 100,
        **kwargs,
    ):
        super().__init__(str(root), split=split, **kwargs)
        self.jpeg_quality = jpeg_quality
        self._root_path = Path(root)

        # Filter to IN100 classes only.
        in1000_classes = set(IN100_TO_IN1000.values())
        filtered = [(p, l) for p, l in self.imgs if l in in1000_classes]
        self.imgs = filtered
        self.samples = filtered
        self.targets = [l for _, l in filtered]

        expected = EXPECTED_COUNTS.get(split)
        if expected is not None and len(self) != expected:
            import warnings
            warnings.warn(
                f"Expected {expected} IN100 samples for {split}, got {len(self)}. "
                f"Check your ImageNet installation at {root}.",
                stacklevel=2,
            )

    @property
    def field_types(self) -> dict[str, str]:
        return {
            "image": "ImageBytes",
            "label": "int",
            "in100_label": "int",
            "in1000_label": "int",
            "index": "int",
            "path": "str",
        }

    @property
    def dataset_hash(self) -> str:
        content = (
            f"imagenet100:{self._root_path}:{self.split}:"
            f"{PREPROCESSING_TAG}:q{self.jpeg_quality}"
        )
        return hashlib.sha256(content.encode()).hexdigest()[:8]

    @property
    def cache_path(self) -> Path:
        from slipstream.utils.cache_dir import get_cache_base
        return Path(get_cache_base()) / f"slipcache-{self.dataset_hash}"

    def __getitem__(self, index: int) -> dict:
        fullpath, in1000_label = self.imgs[index]
        relpath = os.path.join(
            Path(fullpath).parent.name,
            Path(fullpath).name,
        )

        with open(fullpath, "rb") as f:
            raw_bytes = f.read()

        img = decode_image(
            torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8),
            mode="RGB",
        )

        img = _resize_short_crop_long(img, short_size=SHORT_SIZE, long_size=LONG_SIZE)

        jpeg_bytes = encode_jpeg(img, quality=self.jpeg_quality)
        image_bytes = jpeg_bytes.numpy().tobytes()

        in100_label = IN1000_TO_IN100[in1000_label]

        return {
            "image": image_bytes,
            "label": in100_label,
            "in100_label": in100_label,
            "in1000_label": in1000_label,
            "index": index,
            "path": relpath,
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    root={self._root_path},\n"
            f"    split={self.split},\n"
            f"    num_samples={len(self)},\n"
            f"    preprocessing={PREPROCESSING_TAG}_q{self.jpeg_quality},\n"
            f"    fields={self.field_types},\n"
            f")"
        )


def build(
    root: str | Path,
    split: str = "val",
    fmt: str = "jpeg",
    output_base: Path | None = None,
    num_workers: int = 1,
    jpeg_quality: int = 100,
    verbose: bool = True,
) -> Path:
    """Build a standalone imagenet100-s292_l584 cache for one split / format.

    For ``fmt="yuv420"``, slipstream decodes the JPEG bytes emitted by the
    dataset and re-encodes them as YUV420 in the output cache (so the cache's
    ``image`` field is YUV420, not a JPEG sidecar).

    Returns the output cache directory.
    """
    from slipstream.cache import OptimizedCache, write_index

    if fmt not in ("jpeg", "yuv420"):
        raise ValueError(f"fmt must be 'jpeg' or 'yuv420', got {fmt!r}")

    if output_base is None:
        from ..runtime_platform import configure_slipstream_cache
        output_base = Path(configure_slipstream_cache())
    else:
        output_base = Path(output_base)

    output_dir = output_base / f"imagenet100-{PREPROCESSING_TAG}-{fmt}-{split}"

    dataset = ImageNet100_s292_l584(
        root=root, split=split, jpeg_quality=jpeg_quality,
    )

    if verbose:
        print(dataset)
        print(f"Output cache: {output_dir}")

    OptimizedCache.build(
        dataset,
        output_dir=output_dir,
        verbose=verbose,
        num_workers=num_workers,
        image_format=fmt,
    )

    cache = OptimizedCache.load(output_dir, verbose=False)
    index_fields = ["label", "in100_label", "in1000_label"]
    if verbose:
        print(f"Building indexes for: {index_fields}")
    write_index(cache, fields=index_fields)

    if verbose:
        cache = OptimizedCache.load(output_dir, verbose=False)
        print(f"\nDone! Cache written to: {output_dir}")
        print(f"  Samples: {cache.num_samples}")
        print(f"  Fields:  {list(cache.field_types)}")
        label_idx = cache.get_index("label")
        counts = {k: len(v) for k, v in sorted(label_idx.items())}
        min_c, max_c = min(counts.values()), max(counts.values())
        print(f"  Classes: {len(counts)} (samples per class: min={min_c}, max={max_c})")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Build imagenet100-s292_l584 cache from raw ImageNet-1K",
    )
    parser.add_argument(
        "--root", type=Path, required=True,
        help="Path to ImageNet root (containing train/ and val/ directories)",
    )
    parser.add_argument(
        "--split", default="val", choices=["val", "train"],
        help="Dataset split (default: val)",
    )
    parser.add_argument(
        "--fmt", default="jpeg", choices=["jpeg", "yuv420"],
        help="Image format for the output cache (default: jpeg)",
    )
    parser.add_argument(
        "--output-base", type=Path, default=None,
        help="Output base directory (defaults to platform slipstream cache dir)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Parallel workers for cache building (default: 1)",
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=100,
        help="JPEG quality for re-encoded images (default: 100)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Build both val and train splits, in both jpeg and yuv420 formats",
    )
    args = parser.parse_args()

    if args.all:
        targets = [(s, f) for s in ("val", "train") for f in ("jpeg", "yuv420")]
    else:
        targets = [(args.split, args.fmt)]

    for split, fmt in targets:
        print(f"\n{'='*60}")
        print(f"Building imagenet100-{PREPROCESSING_TAG}-{fmt}-{split}")
        print(f"{'='*60}")
        build(
            root=args.root,
            split=split,
            fmt=fmt,
            output_base=args.output_base,
            num_workers=args.num_workers,
            jpeg_quality=args.jpeg_quality,
        )


if __name__ == "__main__":
    main()
