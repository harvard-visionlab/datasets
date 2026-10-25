"""End-to-end check of the ImageNet val *original* caches: torchvision models reach their published top-1.

Each model uses its own weights' test transform on PIL images decoded from the stored bytes (``.convert("RGB")``),
the protocol behind torchvision's reported numbers. Expected (torchvision model docs, IMAGENET1K_V1): alexnet 56.522,
resnet50 76.130. A resized or re-encoded copy typically misses by 0.1-1 point.

    python -m visionlab.datasets.prep.imagenet_original_accuracy [--cache-dir DIR] [--models alexnet resnet50]

Without ``--cache-dir`` the store comes through ``load("imagenet1k", split="val", fmt="original")`` (S3 or the
local mirror), so this also checks the published copy.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

EXPECTED = {"alexnet": 56.522, "resnet50": 76.130}


def main(argv: list[str] | None = None) -> int:
    import torch
    import torchvision
    from PIL import Image

    from .imagenet_original_verify import Records

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--models", nargs="+", default=list(EXPECTED), choices=list(EXPECTED))
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tolerance", type=float, default=0.05, help="max |top-1 - published| in points")
    a = p.parse_args(argv)

    if a.cache_dir is None:
        from visionlab.datasets import configure_slipstream_cache, load

        ds = load("imagenet1k", split="val", fmt="original")
        a.cache_dir = Path(configure_slipstream_cache()) / "imagenet1k-original-val"
        print(f"identity {ds.identity}")
    print(f"cache {a.cache_dir}", flush=True)

    class Val(torch.utils.data.Dataset):
        def __init__(self, tf):
            self.tf, self.r = tf, None

        def __len__(self):
            return 50_000

        def __getitem__(self, i):
            if self.r is None:  # open the mmaps in the worker
                self.r = Records(a.cache_dir)
                self.labels = self.r.ints("label")
            img = Image.open(io.BytesIO(self.r.image(i))).convert("RGB")
            return self.tf(img), int(self.labels[i])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rc = 0
    for name in a.models:
        w = torchvision.models.get_model_weights(name).IMAGENET1K_V1
        model = torchvision.models.get_model(name, weights=w).eval().to(dev)
        dl = torch.utils.data.DataLoader(Val(w.transforms()), batch_size=a.batch_size, num_workers=a.workers)
        correct = n = 0
        t0 = time.time()
        with torch.inference_mode():
            for x, y in dl:
                correct += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
                n += len(y)
        top1 = 100 * correct / n
        ok = n == 50_000 and abs(top1 - EXPECTED[name]) <= a.tolerance
        rc |= not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: top-1 {top1:.3f} (published {EXPECTED[name]:.3f}, "
              f"{n} images, {time.time() - t0:.0f} s)", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
