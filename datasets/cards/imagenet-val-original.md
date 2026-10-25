# ImageNet val originals (`fmt="original"`)

The ImageNet (ILSVRC2012) validation images exactly as distributed: the original JPEG files, byte for byte, in a
slipstream store. Use them to evaluate a classifier with its **own** test transform, so the result is comparable with
published top-1 accuracies. The lab's other ImageNet caches (`s256_l512`) are resized and re-encoded, which shifts
accuracy by a fraction of a point.

```python
from visionlab.datasets import load

ds = load("imagenet1k", split="val", fmt="original")   # 50,000 images, 1,000 classes
ds = load("imagenet100", split="val", fmt="original")  # 5,000 images, the IN100 classes
ds.identity  # "sha256:..." — pin this in an eval spec (see Identity)
```

## What's in it

| dataset | store | records | classes | size |
|---|---|---:|---:|---:|
| `imagenet1k` | `imagenet1k-original-val` | 50,000 | 1,000 × 50 | 6.71 GB |
| `imagenet100` | `imagenet100-original-val` | 5,000 | 100 × 50 | 0.69 GB |

Fields per record:

| field | type | meaning |
|---|---|---|
| `image` | JPEG bytes | the source file, unchanged |
| `label` | int | class index in torchvision's order (sorted WordNet ids), what pretrained torchvision/timm models predict; imagenet100: 0–99 |
| `in100_label`, `in1000_label` | int | imagenet100 only: IN100 index (= `label`) and the original ImageNet-1K index |
| `index` | int | record number, 0..n−1 (sorted by class, then file name) |
| `path` | str | `<wnid>/ILSVRC2012_val_XXXXXXXX.JPEG` |
| `val_id` | int | the number in the file name (1–50,000) = line of the official `ILSVRC2012_validation_ground_truth.txt` |

The images are not all RGB: 899 are grayscale (`L`) and 1 is CMYK (none CMYK in imagenet100: 88 grayscale). They are
stored as they are; decode with `PIL.Image.open(...).convert("RGB")`, as torchvision's evaluation does.
slipstream's own `DecodeOnly`/`supervised_val` pipelines are built for the resized RGB caches; for published-number
comparisons use the model's transform on PIL images.

## Verified

- Every record's sha256 equals its source file's (`prep/imagenet_original_verify.py`).
- Labels match the official devkit (ground-truth file + `meta.mat`, read independently of the folder layout and of
  torchvision), and agree path by path with the existing `s256_l512` jpeg val caches.
- End to end (`prep/imagenet_original_accuracy.py`, H100, published copy fetched through `load()`), torchvision
  IMAGENET1K_V1 weights with their own transforms:

| model | top-1 here | published |
|---|---:|---:|
| alexnet | 56.552 | 56.522 |
| resnet50 | 76.152 | 76.130 |

  The remaining ~0.02–0.03 points come from library versions (PIL / torchvision resize), not the data.

## Identity

`ds.identity` (or `visionlab.datasets.cache_identity(cache_dir)`) is `"sha256:"` over the store's sample count and the
sha256 of each of its files, as recorded in the manifest. It changes if and only if a stored byte changes, and does not
depend on the local path or the visionlab.datasets version. Built 2026-10-25:

- `imagenet1k-original-val`: `sha256:9c96d093e246346a468d2a25b773d96442f172c2f6f79d3eae86cd8b47acb1d9`
- `imagenet100-original-val`: `sha256:f358feb3953248ad50e67eac646d24f1ee14c1a010b9c0043c32aed600dee29b`

## Source and copies

Built from the lab's raw ImageNet copy on machina (`DataLocal/imagenet1k/rawdata`: torchvision layout, `val/<wnid>/`,
`meta.bin`, `ILSVRC2012_devkit_t12.tar.gz`) with `prep/imagenet_original.py` (visionlab-datasets 0.19.0, slipstream
0.11.2). Copies: S3 `s3://visionlab-datasets/slipstream-cache/imagenet{1k,100}/…-original-val/` (private:
ImageNet's terms allow non-commercial research use only), FASRC lab_storage master and netscratch shared cache,
machina DataScratch cache. All copies deep-checked against the manifest hashes.

## Citation

```bibtex
@article{russakovsky2015imagenet,
  title   = {{ImageNet} Large Scale Visual Recognition Challenge},
  author  = {Russakovsky, Olga and Deng, Jia and Su, Hao and Krause, Jonathan and Satheesh, Sanjeev and Ma, Sean and
             Huang, Zhiheng and Karpathy, Andrej and Khosla, Aditya and Bernstein, Michael and Berg, Alexander C. and
             Fei-Fei, Li},
  journal = {International Journal of Computer Vision},
  volume  = {115}, number = {3}, pages = {211--252}, year = {2015}
}
```
