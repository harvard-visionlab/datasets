# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ImageNet-100 — Load and Visualize
#
# Load pre-built ImageNet-100 validation set (100-class subset of ImageNet-1K)
# and verify multi-label fields.
#
# The cache is automatically downloaded from S3 on first use.

# %% [markdown]
# ## Load the dataset

# %%
from visionlab.datasets import load

dataset = load("imagenet100", split="val")
print(f"Samples: {len(dataset):,}")
print(f"Fields: {dataset.field_types}")

# %% [markdown]
# ## View a single sample (multi-label fields)

# %%
from PIL import Image
import io

sample = dataset[0]
print(f"Keys: {list(sample.keys())}")
print(f"Image: {len(sample['image']):,} bytes")
print(f"label (native):  {sample['label']}")
print(f"in100_label:     {sample['in100_label']}")
print(f"in1000_label:    {sample['in1000_label']}")
print(f"Path: {sample['path']}")

img = Image.open(io.BytesIO(sample['image']))
print(f"Size: {img.size}")
img

# %% [markdown]
# ## Verify label consistency

# %%
import numpy as np
from datasets._configs.imagenet_subsets import IN100_TO_IN1000

from slipstream import SlipstreamLoader

loader = SlipstreamLoader(dataset, batch_size=500, shuffle=False)

all_labels = []
all_in100 = []
all_in1000 = []

for batch in loader:
    all_labels.append(batch['label'])
    all_in100.append(batch['in100_label'])
    all_in1000.append(batch['in1000_label'])

all_labels = np.concatenate(all_labels)
all_in100 = np.concatenate(all_in100)
all_in1000 = np.concatenate(all_in1000)

# label == in100_label
assert np.array_equal(all_labels, all_in100), "label != in100_label"
print("label == in100_label: OK")

# Cross-check: in1000_label == IN100_TO_IN1000[in100_label]
expected_in1000 = np.array([IN100_TO_IN1000[l] for l in all_in100])
assert np.array_equal(all_in1000, expected_in1000), "in1000_label mismatch"
print("in1000_label == IN100_TO_IN1000[in100_label]: OK")

# Class distribution: 100 classes, 50 per class
unique, counts = np.unique(all_labels, return_counts=True)
print(f"\nClasses: {len(unique)}")
print(f"Samples per class: min={counts.min()}, max={counts.max()}")
assert len(unique) == 100
print(f"\nTotal: {len(all_labels):,} samples")

loader.shutdown()

# %% [markdown]
# ## Visualize a batch

# %%
from slipstream import SlipstreamLoader, show_batch
from slipstream.pipelines import supervised_val

pipelines = supervised_val(224)
loader = SlipstreamLoader(dataset, batch_size=16, pipelines=pipelines)

batch = next(iter(loader))
print(f"Images: {batch['image'].shape}")
print(f"Labels (in100): {batch['label'][:16]}")

mean = pipelines['image'][-1].mean.numpy()
std = pipelines['image'][-1].std.numpy()
show_batch(batch['image'], batch['label'], n_cols=16, mean=mean, std=std)

# %%
loader.shutdown()
