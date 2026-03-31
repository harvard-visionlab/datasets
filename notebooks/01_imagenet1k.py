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
# # ImageNet-1K — Load and Visualize
#
# Load pre-built ImageNet-1K validation set from S3 remote cache
# and visualize batches using `SlipstreamLoader`.
#
# The cache is automatically downloaded on first use.

# %% [markdown]
# ## Load the dataset

# %%
from visionlab.datasets import load

# Auto-downloads from S3 if not present locally
dataset = load("imagenet1k", split="val")
print(f"Samples: {len(dataset):,}")
print(f"Fields: {dataset.field_types}")

# %% [markdown]
# ## View a single sample

# %%
from PIL import Image
import io

sample = dataset[0]
print(f"Keys: {list(sample.keys())}")
print(f"Image: {len(sample['image']):,} bytes")
print(f"Label: {sample['label']}")
print(f"Path: {sample['path']}")

img = Image.open(io.BytesIO(sample['image']))
print(f"Size: {img.size}")
img

# %% [markdown]
# ## Load batches with SlipstreamLoader

# %%
from slipstream import SlipstreamLoader
from slipstream.pipelines import supervised_val

pipelines = supervised_val(224)
loader = SlipstreamLoader(
    dataset,
    batch_size=16,
    pipelines=pipelines,
)

batch = next(iter(loader))
print(f"Images: {batch['image'].shape}, dtype={batch['image'].dtype}")
print(f"Labels: {batch['label'].shape}")

# %%
from slipstream import show_batch

mean = pipelines['image'][-1].mean.numpy()
std = pipelines['image'][-1].std.numpy()

show_batch(batch['image'], batch['label'], n_cols=16, mean=mean, std=std)

# %% [markdown]
# ## YUV420 format

# %%
yuv_dataset = load("imagenet1k", split="val", fmt="yuv420")
print(f"YUV420 samples: {len(yuv_dataset):,}")

yuv_loader = SlipstreamLoader(
    yuv_dataset,
    batch_size=16,
    pipelines=supervised_val(224),
)

batch_yuv = next(iter(yuv_loader))
show_batch(batch_yuv['image'], batch_yuv['label'], n_cols=16, mean=mean, std=std)

# %%
loader.shutdown()
yuv_loader.shutdown()
