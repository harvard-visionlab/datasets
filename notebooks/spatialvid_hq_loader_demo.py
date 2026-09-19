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
# # SpatialVID-HQ loader demo: frames + interpolated camera poses + ego-motion deltas
#
# What this shows, end to end, on the h265 video store:
#
# 1. `load("spatialvid-hq", ...)` → a `VideoDataset` (store choice by `rate_hz`/`fps`, split, population subset).
# 2. Seeded window anchors `(record, t0)` and a slipstream loader whose `DecodeVideoWindow` stage returns
#    `[B, T, 3, H, W]` uint8 frames **and the true frame times**.
# 3. `ds.poses_at(rec, t_sec)` → `[B, T, 7]` world→camera poses interpolated to those frame times (linear position,
#    slerp rotation) from the ~5 Hz annotations.
# 4. `ds.ego_motion_at(rec, t_sec)` → the same poses **plus** `[B, T-1, 6]` frame-to-frame deltas
#    `[dx dy dz rx ry rz]` in the previous frame's camera axes — the `ego_motion` input of a CNN+RNN
#    (frame → CNN embedding; embedding + ego-motion → RNN → predicted next embedding).
# 5. Seed reproducibility, cold vs warm epoch throughput.
#
# Run inside the lab container (machina) with the store staged in `$SLIPSTREAM_CACHE_DIR`; the QNAP mount or an S3
# download also work (slower first epoch). `OMP_NUM_THREADS=1` before importing torch matters (see the design doc §3).

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import time
import numpy as np
import torch
torch.set_num_threads(1)
import matplotlib.pyplot as plt

from visionlab.datasets import load
from visionlab.datasets.video import camera_centers, ego_motion, relative_motion

# %% [markdown]
# ## 1. Dataset: store, split, population

# %%
t = time.perf_counter()
ds = load("spatialvid-hq", split="val", rate_hz=15)      # default res 456x256, split version v3, subset person_carried_v0
print(ds, f"\nstore_dir = {ds.store_dir}\nload took {time.perf_counter() - t:.1f} s")
print(f"store fps = {ds.fps} (None = native)   clips = {len(ds):,}")
ds.clips.head(3)

# %% [markdown]
# The split table labels every store clip; `load()` applies the population subset by default. `subset="all"` gives
# the whole store's members of a split, `where=` any pandas query over the joined columns (carrier, scene, ...):

# %%
print("val, population      :", len(load("spatialvid-hq", split="val", rate_hz=15)))
print("val, whole store     :", len(load("spatialvid-hq", split="val", rate_hz=15, subset="all")))
print("val, walking only    :", len(load("spatialvid-hq", split="val", rate_hz=15, where="carrier == 'walk'")))
print("val, native fps store:", load("spatialvid-hq", split="val", fps="native").store_dir.name)

# %% [markdown]
# ## 2. Windows: seeded anchors → slipstream loader with `DecodeVideoWindow`
#
# A window is `T` frames at `rate_hz` starting at `t0` seconds into a clip. `ds.window_sampler` drops clips shorter
# than the window and draws one `t0` per clip per epoch, reproducibly from `(seed, epoch)`.

# %%
from slipstream.dataset import SlipstreamDataset
from slipstream.decoders import DecodeVideoWindow
from slipstream.loader import SlipstreamLoader

T, RATE, B = 120, 15.0, 8                         # 8 s windows at 15 Hz
sampler = ds.window_sampler(window_s=T / RATE, seed=0)
recs, t0 = sampler.sample(epoch=0)
print(f"{len(sampler):,} eligible clips; first anchors: rec={recs[:4]} t0={np.round(t0[:4], 2)}")

def make_loader(recs, t0, n=512, workers=None, seed=0):
    """Loader over the first n anchors. The stage reads t0 from `sample_data` and returns the true frame times."""
    stage = DecodeVideoWindow(T=T, rate_hz=RATE, seed=seed, t0_key="t0", device="cpu", num_workers=workers, resize=224)
    return SlipstreamLoader(SlipstreamDataset(local_dir=str(ds.store_dir)), batch_size=B, shuffle=True, seed=seed,
                            drop_last=True, indices=recs[:n], sample_data={"t0": t0[:n]}, batches_ahead=8,
                            image_field="video", pipelines={"video": [stage]}, verbose=False)

loader = make_loader(recs, t0, n=64, workers=16)
batch = next(iter(loader))
frames, t_sec, rec = batch["video"], batch["video_t_sec"], batch["video_rec"]
print("frames", tuple(frames.shape), frames.dtype, "| t_sec", tuple(t_sec.shape), "| rec", tuple(rec.shape))
print("frame times of sample 0 (s):", np.round(t_sec[0, :6].numpy(), 3), "...", np.round(t_sec[0, -2:].numpy(), 3))

# %%
fig, axes = plt.subplots(1, 6, figsize=(18, 2.6))
for ax, j in zip(axes, np.linspace(0, T - 1, 6).astype(int)):
    ax.imshow(frames[0, j].permute(1, 2, 0).numpy()); ax.set_title(f"t = {t_sec[0, j]:.2f} s"); ax.axis("off")
plt.suptitle(f"record {int(rec[0])}: 6 of the {T} frames of one 8 s window"); plt.show()

# %% [markdown]
# ## 3. Interpolated camera poses at the frame times
#
# Poses are world→camera `[tx ty tz qx qy qz qw]` (OpenCV axes, non-metric scale, ~5 Hz annotations). `poses_at`
# interpolates them to the **true** frame times the decoder returned, so pose and pixel agree even for VFR sources.

# %%
poses = ds.poses_at(rec, t_sec)                    # [B, T, 7]
print("poses", poses.shape, poses.dtype)
print("sample 0, first 3 frames:\n", np.round(poses[0, :3], 4))

centers = camera_centers(poses)                    # [B, T, 3] camera positions in world axes
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, b in zip(axes, range(4)):
    c = centers[b]; ax.plot(c[:, 0], c[:, 2], ".-", ms=3); ax.plot(c[0, 0], c[0, 2], "go"); ax.plot(c[-1, 0], c[-1, 2], "rs")
    ax.set_title(f"rec {int(rec[b])}: camera path (x, z), start ● end ■"); ax.set_aspect("equal"); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Ego-motion deltas: change in pose since the last frame
#
# `ds.ego_motion_at` returns the poses and, per consecutive frame pair, the relative motion **expressed in the
# previous frame's camera axes**: `[dx dy dz rx ry rz]` — translation (x right, y down, z forward; pose units) and
# rotation as a rotation vector (radians). Delta `t` describes the move from frame `t` to frame `t+1`, so for a
# CNN+RNN it pairs with frame `t+1` (the RNN sees "where the camera went since the last embedding"). Walking forward
# is a steady positive `dz`; a turn shows in `ry`.

# %%
poses, deltas = ds.ego_motion_at(rec, t_sec)       # [B, T, 7], [B, T-1, 6]
print("poses", poses.shape, "deltas", deltas.shape)
print("sample 0, first 3 deltas [dx dy dz rx ry rz]:\n", np.round(deltas[0, :3], 4))

# identical to the pairwise reference implementation used by the authors' get_instructions.py
ref = np.stack([relative_motion(poses[0, i], poses[0, i + 1]) for i in range(T - 1)])
print("matches relative_motion pairwise:", np.allclose(deltas[0], ref, atol=1e-5))

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
tt = t_sec[0, 1:].numpy()
for k, name in enumerate(["dx", "dy", "dz"]):
    axes[0].plot(tt, deltas[0, :, k], label=name)
for k, name in enumerate(["rx", "ry", "rz"]):
    axes[1].plot(tt, np.degrees(deltas[0, :, 3 + k]), label=name)
axes[0].set_ylabel("translation / frame (pose units)"); axes[1].set_ylabel("rotation / frame (deg)"); axes[1].set_xlabel("t (s)")
for ax in axes: ax.legend(ncol=3); ax.grid(alpha=.3)
plt.suptitle(f"ego-motion of record {int(rec[0])} at {RATE:g} Hz"); plt.show()

# %% [markdown]
# Assembling the RNN input for one batch: frame `t` → CNN embedding; the RNN step at `t` receives
# `[embedding_t, ego_motion_{t-1→t}]` and predicts `embedding_{t+1}`. The first frame has no delta: pad with zeros
# (or drop frame 0).

# %%
ego = torch.from_numpy(deltas)                                    # [B, T-1, 6]
ego = torch.cat([torch.zeros(B, 1, 6), ego], dim=1)               # [B, T, 6]: delta into frame t (0 for frame 0)
x_frames = frames.float().div_(255)                               # [B, T, 3, H, W] → CNN, e.g. x_frames.flatten(0, 1)
print("cnn input", tuple(x_frames.shape), "| rnn side input", tuple(ego.shape), "| target = embeddings shifted by one frame")
print("per-batch ego-motion scale (std):", np.round(ego.std(dim=(0, 1)).numpy(), 4), " ← normalise per dataset before the RNN")

# %% [markdown]
# ## 5. Seed reproducibility and throughput (cold disk → warm page cache)

# %%
def first_batch_signature(seed):
    l = make_loader(recs, t0, n=64, workers=16, seed=seed)
    b = next(iter(l)); l.shutdown()
    return b["video_rec"].numpy().copy(), b["video_t_sec"][:, 0].numpy().copy(), b["video"][:, 0].float().mean().item()

a1, a2, a3 = first_batch_signature(0), first_batch_signature(0), first_batch_signature(1)
print("same seed → same records, t0, pixels:", np.array_equal(a1[0], a2[0]), np.allclose(a1[1], a2[1]), a1[2] == a2[2])
print("other seed → different batch     :", not np.array_equal(a1[0], a3[0]))

# %%
N = 1024
loader = make_loader(recs, t0, n=N, workers=os.cpu_count())
print(f"page-cache residency before: {loader.page_cache_residency():.2f}")
for ep in range(2):
    t = time.perf_counter(); n = 0
    for b in loader:
        n += b["video"].shape[0]
    dt = time.perf_counter() - t
    print(f"epoch {ep + 1}: {n / dt:,.1f} windows/s  ({n * T / dt:,.0f} frames/s, {n} windows, residency now {loader.page_cache_residency():.2f})")
loader.shutdown()

# %% [markdown]
# On machina (64 CPU decoders, 15 fps store on NVMe) this reaches ~160 windows/s ≈ 19k frames/s at 224×398; off a
# network mount the first epoch is disk-bound and the second matches (page cache). Full table: design doc §3.
