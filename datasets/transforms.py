"""Shared preprocessing transforms for dataset preparation.

These transforms are used during dataset preparation (not training) to
standardize image dimensions before building slipstream caches.
"""
from PIL import Image
from torchvision import transforms


__all__ = ['ConvertToRGB', 'ResizeShortCropLong']


class ConvertToRGB:
    """Convert the given PIL image to RGB format.

    Handles RGBA, grayscale, palette, and other modes.
    """

    def __call__(self, img):
        return img.convert('RGB')

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class ResizeShortCropLong:
    """Resize shortest edge and center-crop longest edge.

    Standard preprocessing for vision datasets: resize so the shortest
    edge equals ``short_size``, then center-crop the longest edge to
    ``long_size`` if it exceeds that limit.

    The default (short_size=256, long_size=512) is the standard s256_l512
    preprocessing used for ImageNet and related datasets.

    Args:
        short_size: Target size for the shortest edge.
        long_size: Maximum size for the longest edge (center-cropped if exceeded).
    """

    def __init__(self, short_size=256, long_size=512):
        self.short_size = short_size
        self.long_size = long_size
        self.resize = transforms.Resize(short_size)
        self.crop_width = transforms.CenterCrop((short_size, long_size))
        self.crop_height = transforms.CenterCrop((long_size, short_size))

    def __call__(self, img):
        img = self.resize(img)
        width, height = img.size

        if width > self.long_size:
            img = self.crop_width(img)
        if height > self.long_size:
            img = self.crop_height(img)

        return img

    def __repr__(self):
        return (
            f'{self.__class__.__name__}'
            f'(short_size={self.short_size}, long_size={self.long_size})'
        )
