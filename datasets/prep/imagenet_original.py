"""Build the ImageNet val *original* caches (``fmt="original"``): the ILSVRC2012 val JPEGs as distributed.

No decode, resize or re-encode: each record's ``image`` is the source file's bytes (CMYK / grayscale JPEGs included),
so a model's own test transform sees exactly what the published accuracies were measured on. Built from the raw
``torchvision.datasets.ImageNet`` layout (``val/<wnid>/ILSVRC2012_val_XXXXXXXX.JPEG`` + ``meta.bin``), the same source
as :mod:`datasets.prep.imagenet100_s292`.

- ``imagenet1k-original-val``: 50,000 records; fields image, label (0-999, torchvision class order), index, path
  (``<wnid>/<file>``), val_id (the number in the file name, 1-50,000 = line of the official ground-truth file).
- ``imagenet100-original-val``: the 5,000 records of the IN100 classes, plus in100_label / in1000_label
  (``label`` = in100_label), same as the other imagenet100 caches.

slipstream stores JPEG ``ImageBytes`` verbatim (it only transcodes non-JPEG files, none of which are in val; the build
refuses them here). ``imagenet_original_verify`` checks every record's sha256 against its source file.

    python -m visionlab.datasets.prep.imagenet_original --root ~/work/DataLocal/imagenet1k/rawdata \\
        --output-base ~/work/DataScratch/cache/slipstream-cache [--subset in1k in100]
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from torchvision.datasets import ImageNet

from .._configs.imagenet_subsets import IN100_TO_IN1000, IN1000_TO_IN100

__all__ = ["ImageNetValOriginal", "build", "CACHE_NAMES", "EXPECTED_COUNTS"]

CACHE_NAMES = {"in1k": "imagenet1k-original-val", "in100": "imagenet100-original-val"}
EXPECTED_COUNTS = {"in1k": 50_000, "in100": 5_000}
JPEG_SOI = b"\xff\xd8\xff"


def val_id(path: str) -> int:
    """``ILSVRC2012_val_00000293.JPEG`` -> 293."""
    return int(Path(path).stem.rsplit("_", 1)[1])


class ImageNetValOriginal(ImageNet):
    """ImageNet val with the original file bytes; ``subset="in100"`` keeps the IN100 classes and adds their labels."""

    def __init__(self, root: str | Path, subset: str = "in1k", **kwargs):
        super().__init__(str(root), split="val", **kwargs)
        if subset not in CACHE_NAMES:
            raise ValueError(f"subset must be one of {list(CACHE_NAMES)}, got {subset!r}")
        self.subset, self._root_path = subset, Path(root)
        if subset == "in100":
            keep = set(IN100_TO_IN1000.values())
            self.imgs = self.samples = [(p, l) for p, l in self.imgs if l in keep]
            self.targets = [l for _, l in self.imgs]
        if len(self) != EXPECTED_COUNTS[subset]:
            raise RuntimeError(f"{subset}: expected {EXPECTED_COUNTS[subset]} val images under {root}, got {len(self)}")

    @property
    def field_types(self) -> dict[str, str]:
        f = {"image": "ImageBytes", "label": "int"}
        if self.subset == "in100":
            f |= {"in100_label": "int", "in1000_label": "int"}
        return f | {"index": "int", "path": "str", "val_id": "int"}

    @property
    def dataset_hash(self) -> str:
        return hashlib.sha256(f"imagenet-val-original:{self.subset}:{self._root_path}".encode()).hexdigest()[:8]

    @property
    def cache_path(self) -> Path:
        from slipstream.utils.cache_dir import get_cache_base
        return Path(get_cache_base()) / f"slipcache-{self.dataset_hash}"

    def __getitem__(self, index: int) -> dict:
        fullpath, in1000 = self.imgs[index]
        with open(fullpath, "rb") as f:
            raw = f.read()
        if not raw.startswith(JPEG_SOI):  # slipstream would transcode it: refuse instead
            raise ValueError(f"not a JPEG: {fullpath}")
        rel = f"{Path(fullpath).parent.name}/{Path(fullpath).name}"
        out = {"image": raw}
        if self.subset == "in100":
            out |= {"label": IN1000_TO_IN100[in1000], "in100_label": IN1000_TO_IN100[in1000], "in1000_label": in1000}
        else:
            out["label"] = in1000
        return out | {"index": index, "path": rel, "val_id": val_id(rel)}


def build(root: str | Path, subset: str, output_base: str | Path, num_workers: int = 1) -> Path:
    from slipstream.cache import OptimizedCache, write_index

    ds = ImageNetValOriginal(root, subset=subset)
    out = Path(output_base) / CACHE_NAMES[subset]
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out} already exists; remove it to rebuild")
    print(f"{subset}: {len(ds)} images from {root} -> {out}", flush=True)
    OptimizedCache.build(ds, output_dir=out, num_workers=num_workers, image_format="jpeg", verbose=True)
    cache = OptimizedCache.load(out, verbose=False)
    write_index(cache, fields=["label"] + (["in100_label", "in1000_label"] if subset == "in100" else []))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, required=True, help="ImageNet root (val/<wnid>/*.JPEG + meta.bin)")
    p.add_argument("--output-base", type=Path, required=True, help="directory to write the caches into")
    p.add_argument("--subset", nargs="+", default=["in1k", "in100"], choices=list(CACHE_NAMES))
    p.add_argument("--num-workers", type=int, default=1)
    a = p.parse_args(argv)
    os.makedirs(a.output_base, exist_ok=True)
    for s in a.subset:
        build(a.root, s, a.output_base, a.num_workers)


if __name__ == "__main__":
    main()
