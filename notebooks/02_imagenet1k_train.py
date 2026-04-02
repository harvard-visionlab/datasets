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
# # ImageNet-1K Train — Load and Visualize
#
# Load pre-built ImageNet-1K training set from S3 remote cache
# and visualize batches. The cache is automatically downloaded on first use.

# %% [markdown]
# ## Load JPEG train set

# %%
from visionlab.datasets import load

dataset = load("imagenet1k", split="train", fmt="jpeg")
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
# ## Training loader (RandomResizedCrop)

# %%
from slipstream import SlipstreamLoader, show_batch
from slipstream.decoders import DecodeRandomResizedCrop, DecodeCenterCrop

loader = SlipstreamLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    pipelines={"image": [DecodeRandomResizedCrop(224)]},
    exclude_fields=["path"],
)

batch = next(iter(loader))
print(f"Images: {batch['image'].shape}")
show_batch(batch['image'], batch['label'], n_cols=16)

# %% [markdown]
# ## Validation-style loader (CenterCrop)

# %%
val_loader = SlipstreamLoader(
    dataset,
    batch_size=16,
    pipelines={"image": [DecodeCenterCrop(224)]},
    exclude_fields=["path"],
)

batch_val = next(iter(val_loader))
show_batch(batch_val['image'], batch_val['label'], n_cols=16)

# %%
loader.shutdown()
val_loader.shutdown()

# %% [markdown]
# ## YUV420 train set

# %%
yuv_dataset = load("imagenet1k", split="train", fmt="yuv420")
print(f"YUV420 samples: {len(yuv_dataset):,}")

yuv_loader = SlipstreamLoader(
    yuv_dataset,
    batch_size=16,
    shuffle=True,
    pipelines={"image": [DecodeRandomResizedCrop(224)]},
    exclude_fields=["path"],
)

batch_yuv = next(iter(yuv_loader))
print(f"Images: {batch_yuv['image'].shape}")
show_batch(batch_yuv['image'], batch_yuv['label'], n_cols=16)

# %%
yuv_loader.shutdown()
