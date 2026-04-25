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
# # ImageNet-100 (s292_l584) — Load and Visualize
#
# Loads the pre-built `imagenet100_s292` validation set in both JPEG and
# YUV420 formats, prints image sizes, and visualizes a batch from each.
# Caches are auto-downloaded from S3 on first use.

# %% [markdown]
# ## JPEG val set

# %%
from visionlab.datasets import load

dataset_jpeg = load("imagenet100_s292", split="val", fmt="jpeg")
print(f"Samples: {len(dataset_jpeg):,}")
print(f"Fields:  {dataset_jpeg.field_types}")
print(f"Stats:   mean={dataset_jpeg.stats['mean']} std={dataset_jpeg.stats['std']}")

# %% [markdown]
# ### Single sample

# %%
import io
from PIL import Image

sample = dataset_jpeg[0]
print(f"Keys:         {sorted(sample.keys())}")
print(f"label:        {sample['label']}")
print(f"in100_label:  {sample['in100_label']}")
print(f"in1000_label: {sample['in1000_label']}")
print(f"path:         {sample['path']}")
print(f"image bytes:  {len(sample['image']):,}")

img = Image.open(io.BytesIO(sample['image']))
print(f"Image size:   {img.size} (W, H)  short_edge={min(img.size)}  long_edge={max(img.size)}")
img

# %% [markdown]
# ### Visualize a batch

# %%
from slipstream import SlipstreamLoader, show_batch
from slipstream.pipelines import supervised_val

pipelines = supervised_val(224)
loader_jpeg = SlipstreamLoader(
    dataset_jpeg,
    batch_size=16,
    pipelines=pipelines,
)

batch = next(iter(loader_jpeg))
print(f"Images: {batch['image'].shape}, dtype={batch['image'].dtype}")
print(f"Labels: {batch['label'][:16].tolist()}")

mean = pipelines['image'][-1].mean.numpy()
std = pipelines['image'][-1].std.numpy()
show_batch(batch['image'], batch['label'], n_cols=16, mean=mean, std=std)

# %%
loader_jpeg.shutdown()

# %% [markdown]
# ## YUV420 val set

# %%
dataset_yuv = load("imagenet100_s292", split="val", fmt="yuv420")
print(f"Samples: {len(dataset_yuv):,}")
print(f"Fields:  {dataset_yuv.field_types}")
print(f"Stats:   mean={dataset_yuv.stats['mean']} std={dataset_yuv.stats['std']}")

# %% [markdown]
# ### Single sample
#
# YUV420 samples come back as a dict with the raw planes plus dimensions
# (the decoders in the loader pipeline turn this into RGB or YUV tensors).

# %%
sample = dataset_yuv[0]
print(f"Keys:         {sorted(sample.keys())}")
print(f"label:        {sample['label']}")
print(f"in100_label:  {sample['in100_label']}")
print(f"in1000_label: {sample['in1000_label']}")
print(f"path:         {sample['path']}")

img_dict = sample['image']
print(f"Image dict keys: {list(img_dict.keys())}")
print(
    f"Image size:   ({img_dict['width']}, {img_dict['height']}) (W, H)  "
    f"short_edge={min(img_dict['width'], img_dict['height'])}  "
    f"long_edge={max(img_dict['width'], img_dict['height'])}"
)
print(f"YUV bytes:    {len(img_dict['bytes']):,} (expected H*W*1.5 = {int(img_dict['height']*img_dict['width']*1.5):,})")

# Sanity check: same underlying sample as the JPEG cache
assert dataset_jpeg[0]['path'] == sample['path'], "path mismatch jpeg vs yuv420"
assert dataset_jpeg[0]['label'] == sample['label'], "label mismatch jpeg vs yuv420"
print("\njpeg/yuv420 caches encode the same sample at index 0: OK")

# %% [markdown]
# ### Visualize a batch (RGB-decoded)

# %%
loader_yuv = SlipstreamLoader(
    dataset_yuv,
    batch_size=16,
    pipelines=supervised_val(224),
)

batch_yuv = next(iter(loader_yuv))
print(f"Images: {batch_yuv['image'].shape}, dtype={batch_yuv['image'].dtype}")
print(f"Labels: {batch_yuv['label'][:16].tolist()}")

show_batch(batch_yuv['image'], batch_yuv['label'], n_cols=16, mean=mean, std=std)

# %%
loader_yuv.shutdown()
