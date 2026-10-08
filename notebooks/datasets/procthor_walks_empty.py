# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # procthor-walks-empty
#
# Egocentric random walks through ProcTHOR-10k houses with the furniture removed, rendered at 160×120 in AI2-THOR by Rupert
# Tawiah-Quashie (Harvard Vision Lab). One sample = one 1,000-step walk: the RGB frames (stored as an mp4), the agent's
# (x, y) position in metres and its heading in radians at every step. Card: `datasets/cards/procthor-walks.md`.
#
# This notebook uses the **val** split (2,000 walks, the smallest download). It
# 1. shows what would be downloaded, how big it is and where it goes, **before** downloading;
# 2. loads the split and walks through the record fields;
# 3. decodes a training window and plots frames, the path and the egomotion;
# 4. runs the training-style loader (61-frame windows, as the student's world models were trained);
# 5. offers to delete the local copy.
#
# macOS: torchcodec needs FFmpeg's shared libraries (`brew install ffmpeg`, then start Jupyter with
# `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`). FASRC: `. lab_env.sh` provides them.

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")          # decoder threads, not torch's intra-op pool
import numpy as np
import torch
torch.set_num_threads(1)
import matplotlib.pyplot as plt

from visionlab.datasets import load
from visionlab.datasets.demo import download_plan, remove_local_copy

DATASET, SPLIT = "procthor-walks-empty", "val"

# %% [markdown]
# ## 1. What will be downloaded, and where
#
# Each split lives in its own store (train ≈ 48 GB, val and test ≈ 5 GB each). Nothing is downloaded by this cell.

# %%
plan = download_plan(DATASET, split=SPLIT)

# %% [markdown]
# Set `DOWNLOAD = True` to fetch it (skipped automatically when the store is already local, e.g. on the cluster).

# %%
DOWNLOAD = False
if plan.local_path is None and not DOWNLOAD:
    raise SystemExit(f"Not downloading {plan.size_gb or 0:.1f} GB. Set DOWNLOAD = True in the cell above to continue.")
ds = load(DATASET, split=SPLIT)
print(ds)
print("store directory:", ds.store_dir)

# %% [markdown]
# ## 2. Records and fields
#
# `ds.clips` has one row per walk of the split: `record_idx` (row in the store), `clip_id`
# (`<folder>/house_XXXXX/seed_XXXX`, the same id in `procthor-walks-objects` for the paired walk) and the generator's
# per-walk metadata.

# %%
cols = ["record_idx", "clip_id", "house", "seed", "n_rooms", "n_objects", "time_of_day", "skybox", "mean_brightness"]
ds.clips[cols].head()

# %%
i0 = np.array([int(ds.indices[0])])
for name, field in ds.cache.fields.items():
    out = field.load_batch(i0)
    if "sizes" in out:
        print(f"{name:11s} bytes, {int(out['sizes'][0]):,} B")
    else:
        v = np.asarray(out["data"][0])
        print(f"{name:11s} {v.dtype} {v.shape}  {v.ravel()[:4]}")

# %% [markdown]
# Fields of one record:
#
# | field | contents |
# | --- | --- |
# | `video` | mp4 bytes: 1,000 frames, 160×120, h264 4:4:4, keyframe every 60 frames (30 fps is a convention; the walk has no time axis) |
# | `positions` | float32 (1000, 2): agent (x, y) in metres, one row per frame |
# | `headings` | float32 (1000,): heading in radians, one per frame |
# | `fps`, `num_frames`, `duration_s` | 30, 1000, 33.3 |

# %% [markdown]
# ## 3. Decode a window and look at it
#
# A training window is 61 consecutive frames starting at a multiple of 60 (16 windows per walk). Decoding by frame
# index with torchcodec:

# %%
from torchcodec.decoders import VideoDecoder

i = int(ds.indices[0])
cache = ds.cache
vb = cache.fields["video"].load_batch(np.array([i]), parallel=False)
video = bytes(vb["data"][0][: int(vb["sizes"][0])])
pos = np.asarray(cache.fields["positions"].load_batch(np.array([i]))["data"][0])
head = np.asarray(cache.fields["headings"].load_batch(np.array([i]))["data"][0])
start = 300
frames = VideoDecoder(video, dimension_order="NHWC").get_frames_in_range(start, start + 61).data.numpy()
print("frames", frames.shape, frames.dtype, "| mp4 size", f"{len(video) / 1e6:.2f} MB")

fig, axes = plt.subplots(2, 6, figsize=(15, 4))
for ax, t in zip(axes.flat, range(0, 61, 5)):
    ax.imshow(frames[t]); ax.set_title(f"frame {start + t}"); ax.axis("off")
plt.tight_layout(); plt.show()

# %%
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(pos[:, 0], pos[:, 1], lw=0.6, color="0.6", label="whole walk")
w = slice(start, start + 61)
ax.plot(pos[w, 0], pos[w, 1], lw=2, label=f"window {start}-{start + 60}")
q = slice(start, start + 61, 10)
# facing direction = (-sin h, -cos h) (generator convention, see the card); y axis down so the view is not mirrored
ax.quiver(pos[q, 0], pos[q, 1], -np.sin(head[q]), -np.cos(head[q]), angles="xy", scale=15, width=0.006)
ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m, down)"); ax.legend(); plt.show()

# %% [markdown]
# **Egomotion** (the world model's movement input): the step between frames t and t+1 rotated into the agent's frame at
# t, plus the wrapped heading change. Normalised with `ds.stats`-style constants in training (see the card).

# %%
step = pos[1:] - pos[:-1]
c, s = np.cos(head[:-1]), np.sin(head[:-1])
turn = (np.diff(head) + np.pi) % (2 * np.pi) - np.pi
ego = np.stack([step[:, 0] * c - step[:, 1] * s, step[:, 0] * s + step[:, 1] * c, turn], -1)
print("egomotion (999, 3): mean", ego.mean(0).round(4), "std", ego.std(0).round(4))

# %% [markdown]
# ## 4. Training-style loader
#
# 16 windows per walk at starts 0, 60, …, 900. slipstream's `DecodeVideoWindow` decodes them in a thread pool; it takes
# a start time, so `t0 = (start + 0.5) / 30` (mid-frame, so rounding never picks the neighbouring frame).
# Positions and headings arrive for the whole walk and are sliced to the window.

# %%
from slipstream.decoders import DecodeVideoWindow
from slipstream.loader import SlipstreamLoader

starts = np.arange(0, 901, 60)
recs = np.repeat(ds.indices, len(starts))
st = np.tile(starts, len(ds.indices))
stage = DecodeVideoWindow(T=61, rate_hz=30, t0_key="t0", num_workers=os.cpu_count(), num_ffmpeg_threads=1)
loader = SlipstreamLoader(ds, batch_size=32, shuffle=True, seed=0, indices=recs,
                          sample_data={"t0": (st + 0.5) / 30, "start": st}, image_field="video",
                          pipelines={"video": [stage]}, verbose=False)
batch = next(iter(loader))
idx = batch["start"].long()[:, None] + torch.arange(61)[None]
positions = torch.gather(batch["positions"], 1, idx[..., None].expand(-1, -1, 2))
headings = torch.gather(batch["headings"], 1, idx)
print("video", tuple(batch["video"].shape), batch["video"].dtype, "| positions", tuple(positions.shape),
      "| headings", tuple(headings.shape), f"| {len(recs):,} windows per epoch")
loader.shutdown()

# %% [markdown]
# ## 5. Remove the local copy
#
# Deletes only a personal download (shared lab copies on the cluster or the QNAP are refused). Dry run first.

# %%
remove_local_copy(ds);                # what would be deleted (dry run)
# remove_local_copy(ds, confirm=True) # uncomment to delete
