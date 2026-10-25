"""Verify the ImageNet val *original* caches (see :mod:`datasets.prep.imagenet_original`).

Every check runs over all records:

1. counts: 50,000 / 5,000 records, 50 per class, ``index`` = 0..n-1, ``val_id`` unique;
2. bytes: sha256 of each record's image == sha256 of its source file (``--root``/val/<path>);
3. labels vs the official devkit (``ILSVRC2012_validation_ground_truth.txt`` + ``meta.mat``, read straight from
   ``ILSVRC2012_devkit_t12.tar.gz``, independent of the val/<wnid>/ folder layout and of torchvision's ``meta.bin``):
   the ground-truth wnid of ``val_id`` == the file's folder, and ``label`` == torchvision's index of that wnid
   (sorted wnids); in100: in100/in1000 labels via the IN100 map, and the same bytes as the in1k record with that path;
4. (``--compare``) labels agree, by path, with an existing cache of the same dataset (e.g. the s256_l512 jpeg val);
5. reports CMYK / grayscale / other JPEG colour modes (kept as stored).

    python -m visionlab.datasets.prep.imagenet_original_verify --root .../rawdata --base .../slipstream-cache \\
        [--compare in1k=.../imagenet1k-s256_l512-jpeg-val in100=.../imagenet100-s256_l512-jpeg-val]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

from .._configs.imagenet_subsets import IN1000_TO_IN100, IN100_TO_IN1000
from .imagenet_original import CACHE_NAMES, EXPECTED_COUNTS, val_id


class Records:
    """Raw field access to a slipstream cache (no decode)."""

    def __init__(self, cache_dir: Path):
        from slipstream.cache import OptimizedCache

        self.c = OptimizedCache.load(cache_dir, verbose=False)
        self.n = self.c.num_samples

    def ints(self, name: str) -> np.ndarray:
        return np.asarray(self.c.fields[name]._data[: self.n]).astype(np.int64)

    def strs(self, name: str) -> list[str]:
        s = self.c.fields[name]
        return [bytes(s._data_mmap[int(o): int(o) + int(l)]).decode() for o, l in s._offsets[: self.n]]

    def image(self, i: int) -> bytes:
        s = self.c.fields["image"]
        m = s._metadata[i]
        p, z = int(m["data_ptr"]), int(m["data_size"])
        return bytes(s._data_mmap[p: p + z])


def devkit_truth(devkit: Path) -> dict[int, str]:
    """val_id -> wnid from the official devkit tarball."""
    from scipy.io import loadmat

    with tarfile.open(devkit) as t:
        def member(suffix):
            return t.extractfile(next(m for m in t.getmembers() if m.name.endswith(suffix))).read()
        meta = loadmat(io.BytesIO(member("data/meta.mat")), squeeze_me=True)["synsets"]
        gt = member("data/ILSVRC2012_validation_ground_truth.txt").decode().split()
    id_to_wnid = {int(s[0]): str(s[1]) for s in meta if int(s[4]) == 0}  # num_children == 0: the 1000 leaf classes
    assert len(id_to_wnid) == 1000 and len(gt) == 50_000
    return {i + 1: id_to_wnid[int(g)] for i, g in enumerate(gt)}


def color_mode(b: bytes) -> str:
    from PIL import Image

    return Image.open(io.BytesIO(b)).mode


def check(subset: str, cache: Path, root: Path, truth: dict[int, str], in1k_sha: dict[str, str] | None,
          compare: Path | None) -> tuple[list[str], dict[str, str]]:
    probs: list[str] = []
    r = Records(cache)
    n, want = r.n, EXPECTED_COUNTS[subset]
    labels, index, vids, paths = r.ints("label"), r.ints("index"), r.ints("val_id"), r.strs("path")
    if n != want:
        probs.append(f"{n} records, expected {want}")
    if not np.array_equal(index, np.arange(n)):
        probs.append("index is not 0..n-1")
    if len(set(vids.tolist())) != n:
        probs.append("val_id not unique")
    per_class = Counter(labels.tolist())
    if len(per_class) != (1000 if subset == "in1k" else 100) or set(per_class.values()) != {50}:
        probs.append(f"class counts: {len(per_class)} classes, sizes {sorted(set(per_class.values()))}")

    wnids = sorted(set(truth.values()))  # torchvision class order
    wnid_idx = {w: i for i, w in enumerate(wnids)}
    if subset == "in100":
        in100, in1000 = r.ints("in100_label"), r.ints("in1000_label")

    sha: dict[str, str] = {}
    modes: Counter = Counter()
    for i in range(n):
        p, v = paths[i], int(vids[i])
        wnid = p.split("/")[0]
        if val_id(p) != v:
            probs.append(f"{i}: val_id {v} != file name {p}")
        if truth[v] != wnid:
            probs.append(f"{i}: devkit says {truth[v]} for val_id {v}, folder is {wnid}")
        exp1000 = wnid_idx[truth[v]]
        if subset == "in1k":
            if labels[i] != exp1000:
                probs.append(f"{i}: label {labels[i]} != {exp1000}")
        else:
            if not (in1000[i] == exp1000 and in100[i] == labels[i] == IN1000_TO_IN100.get(exp1000, -1)):
                probs.append(f"{i}: labels {labels[i]}/{in100[i]}/{in1000[i]} != in1000 {exp1000}")
        b = r.image(i)
        sha[p] = h = hashlib.sha256(b).hexdigest()
        if h != hashlib.sha256((root / "val" / p).read_bytes()).hexdigest():
            probs.append(f"{i}: bytes differ from {root / 'val' / p}")
        if in1k_sha is not None and in1k_sha.get(p) != h:
            probs.append(f"{i}: bytes differ from the in1k record {p}")
        modes[color_mode(b)] += 1
        if (i + 1) % 10_000 == 0:
            print(f"  {subset}: {i + 1}/{n}", flush=True)
        if len(probs) > 50:
            probs.append("... stopping after 50 problems")
            break
    if subset == "in100" and set(in1000.tolist()) != set(IN100_TO_IN1000.values()):
        probs.append("in1000 labels are not the IN100 classes")

    if compare is not None:
        o = Records(compare)
        other = dict(zip(o.strs("path"), o.ints("label").tolist()))
        mism = [p for p, l in zip(paths, labels.tolist()) if other.get(p) != l]
        if len(other) != n or mism:
            probs.append(f"vs {compare.name}: {len(other)} records, {len(mism)} label/path mismatches (e.g. {mism[:3]})")
        else:
            print(f"  {subset}: labels match {compare.name} for all {n} paths")

    print(f"  {subset}: colour modes {dict(modes)}; {sum(len(r.image(i)) for i in range(n)) / 1e9:.2f} GB of JPEG")
    return probs, sha


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, required=True, help="raw ImageNet root (val/, ILSVRC2012_devkit_t12.tar.gz)")
    p.add_argument("--base", type=Path, required=True, help="directory holding the original caches")
    p.add_argument("--devkit", type=Path, default=None, help="default: ROOT/ILSVRC2012_devkit_t12.tar.gz")
    p.add_argument("--compare", nargs="*", default=[], help="SUBSET=CACHE_DIR: labels must agree by path")
    a = p.parse_args(argv)
    truth = devkit_truth(a.devkit or a.root / "ILSVRC2012_devkit_t12.tar.gz")
    cmp = dict(c.split("=", 1) for c in a.compare)
    rc, in1k_sha = 0, None
    for subset in ("in1k", "in100"):
        cache = a.base / CACHE_NAMES[subset]
        print(f"== {cache}", flush=True)
        probs, sha = check(subset, cache, a.root, truth, in1k_sha if subset == "in100" else None,
                           Path(cmp[subset]) if subset in cmp else None)
        if subset == "in1k":
            in1k_sha = sha
        print(("PASS" if not probs else "FAIL") + f" {CACHE_NAMES[subset]}")
        for x in probs:
            print("  ✗", x)
        rc |= bool(probs)
    return rc


if __name__ == "__main__":
    sys.exit(main())
