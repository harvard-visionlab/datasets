"""ImageNet-1K dataset preparation for slipstream.

Provides dataset classes that subclass ``torchvision.datasets.ImageNet``
for label consistency with official PyTorch models. Preprocessing is
baked into ``__getitem__`` so that ``OptimizedCache.build()`` receives
ready-to-cache samples.

Usage::

    from visionlab.datasets.prep.imagenet1k import ImageNet1k_s256_l512
    from slipstream.cache import OptimizedCache

    dataset = ImageNet1k_s256_l512("/path/to/imagenet", split="val")
    cache = OptimizedCache.build(dataset, num_workers=8)
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torchvision.datasets import ImageNet
from torchvision.io import decode_image, encode_jpeg
from torchvision.transforms.functional import center_crop, resize


__all__ = ['ImageNet1k_s256_l512']


def _resize_short_crop_long(
    img: torch.Tensor,
    short_size: int = 256,
    long_size: int = 512,
) -> torch.Tensor:
    """Resize shortest edge and center-crop longest edge.

    Args:
        img: CHW uint8 tensor.
        short_size: Target size for the shortest edge.
        long_size: Maximum size for the longest edge.

    Returns:
        CHW uint8 tensor with short edge = short_size and long edge <= long_size.
    """
    img = resize(img, short_size, antialias=True)
    _, h, w = img.shape
    crop_h = min(h, long_size)
    crop_w = min(w, long_size)
    if crop_h < h or crop_w < w:
        img = center_crop(img, [crop_h, crop_w])
    return img


class ImageNet1k_s256_l512(ImageNet):
    """ImageNet-1K with s256_l512 preprocessing for slipstream cache building.

    Subclasses ``torchvision.datasets.ImageNet`` for label consistency with
    official PyTorch models. Each sample is read as raw bytes, decoded via
    torchvision, preprocessed (resize short edge to 256, center-crop long
    edge to 512), and re-encoded as JPEG quality 100.

    Returns dict samples compatible with ``OptimizedCache.build()``.

    Args:
        root: Path to ImageNet root (contains ``train/`` and ``val/`` dirs).
        split: ``"train"`` or ``"val"``.
        jpeg_quality: JPEG encoding quality (1-100). Default 100.
    """

    # Expected sample counts for validation
    META = {
        "train": 1_281_167,
        "val": 50_000,
    }

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

        # Validate sample count
        expected = self.META.get(split)
        if expected is not None and len(self) != expected:
            import warnings
            warnings.warn(
                f"Expected {expected} samples for {split}, got {len(self)}. "
                f"Check your ImageNet installation at {root}.",
                stacklevel=2,
            )

    @property
    def field_types(self) -> dict[str, str]:
        return {
            "image": "ImageBytes",
            "label": "int",
            "index": "int",
            "path": "str",
        }

    @property
    def dataset_hash(self) -> str:
        """Content-based hash incorporating source path, split, and preprocessing."""
        content = f"imagenet1k:{self._root_path}:{self.split}:s256_l512:q{self.jpeg_quality}"
        return hashlib.sha256(content.encode()).hexdigest()[:8]

    @property
    def cache_path(self) -> Path:
        from slipstream.utils.cache_dir import get_cache_base
        return Path(get_cache_base()) / f"slipcache-{self.dataset_hash}"

    def __getitem__(self, index: int) -> dict:
        fullpath, label = self.imgs[index]
        relpath = os.path.join(
            Path(fullpath).parent.name,
            Path(fullpath).name,
        )

        # Read raw bytes and decode
        with open(fullpath, 'rb') as f:
            raw_bytes = f.read()

        img = decode_image(
            torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8),
            mode="RGB",
        )

        # Preprocess: resize short edge to 256, crop long edge to 512
        img = _resize_short_crop_long(img, short_size=256, long_size=512)

        # Encode back to JPEG
        jpeg_bytes = encode_jpeg(img, quality=self.jpeg_quality)
        image_bytes = jpeg_bytes.numpy().tobytes()

        return {
            "image": image_bytes,
            "label": label,
            "index": index,
            "path": relpath,
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    root={self._root_path},\n"
            f"    split={self.split},\n"
            f"    num_samples={len(self)},\n"
            f"    preprocessing=s256_l512_q{self.jpeg_quality},\n"
            f"    fields={self.field_types},\n"
            f")"
        )
