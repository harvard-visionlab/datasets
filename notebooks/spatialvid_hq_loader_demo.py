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
# 2. `SlipstreamLoader(ds, ...)` with a `DecodeVideoWindow` stage that returns `[B, T, 3, H, W]` uint8 frames **and the
#    true frame times**, drawing a random window start per clip per epoch.
# 3. `batch["poses"]` → `[B, T, 7]` world→camera poses interpolated to those frame times (linear position,
#    slerp rotation) from the ~5 Hz annotations, added by the `ds.ego_motion_transform()` after-batch transform.
# 4. `batch["ego"]` → `[B, T-1, 6]` frame-to-frame deltas
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
# ## 2. Windows: `SlipstreamLoader(ds, ...)` with `DecodeVideoWindow` + the ego-motion after-batch transform
#
# A window is `T` frames at `rate_hz` starting at `t0` seconds into a clip. `DecodeVideoWindow` draws `t0`
# uniformly inside each clip (fresh draw every epoch, seeded); `ds.window_sampler(window_s).recs` just drops the
# clips too short for the window. `ds.ego_motion_transform()` adds `poses` [B, T, 7] and `ego` [B, T-1, 6] to
# every batch from the decoder's true frame times (vectorised over the batch, ~0.05 ms per window).

# %%
from slipstream.decoders import DecodeVideoWindow
from slipstream.loader import SlipstreamLoader

T, RATE, B = 120, 15.0, 8                         # 8 s windows at 15 Hz
clips = ds.window_sampler(window_s=T / RATE).recs
print(f"{len(clips):,} clips long enough for an {T / RATE:g} s window")

def make_loader(clips, workers=None, seed=0, n=None):
    stage = DecodeVideoWindow(T=T, rate_hz=RATE, seed=seed, device="cpu", num_workers=workers, resize=224)
    return SlipstreamLoader(ds, indices=clips if n is None else clips[:n], batch_size=B, shuffle=True, seed=seed, drop_last=True,
                            batches_ahead=8, image_field="video", pipelines={"video": [stage]},
                            after_batch_transforms=[ds.ego_motion_transform()], verbose=False)

loader = make_loader(clips, workers=16, n=64)
batch = next(iter(loader))
frames, t_sec, rec = batch["video"], batch["video_t_sec"], batch["video_rec"]
poses, deltas = batch["poses"].numpy(), batch["ego"].numpy()
print("frames", tuple(frames.shape), frames.dtype, "| t_sec", tuple(t_sec.shape), "| poses", poses.shape, "| ego", deltas.shape)
print("frame times of sample 0 (s):", np.round(t_sec[0, :6].numpy(), 3), "...", np.round(t_sec[0, -2:].numpy(), 3))

# %%
fig, axes = plt.subplots(1, 6, figsize=(18, 2.6))
for ax, j in zip(axes, np.linspace(0, T - 1, 6).astype(int)):
    ax.imshow(frames[0, j].permute(1, 2, 0).numpy()); ax.set_title(f"t = {t_sec[0, j]:.2f} s"); ax.axis("off")
plt.suptitle(f"record {int(rec[0])}: 6 of the {T} frames of one 8 s window"); plt.show()

# %% [markdown]
# ## 3. Interpolated camera poses at the frame times
#
# Poses are world→camera `[tx ty tz qx qy qz qw]` (OpenCV axes, non-metric scale, ~5 Hz annotations), interpolated
# (linear position, slerp rotation) to the **true** frame times the decoder returned, so pose and pixel agree even
# for VFR sources. `batch["poses"]` is exactly `ds.poses_at(rec, t_sec)`:

# %%
print("matches ds.poses_at:", np.allclose(poses, ds.poses_at(rec, t_sec)))
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
# `batch["ego"]` holds, per consecutive frame pair, the relative motion **expressed in the previous frame's camera
# axes**: `[dx dy dz rx ry rz]` — translation (x right, y down, z forward; pose units) and rotation as a rotation
# vector (radians). Delta `t` describes the move from frame `t` to frame `t+1`, so for a CNN+RNN it pairs with
# frame `t+1` (the RNN sees "where the camera went since the last embedding"). Walking forward is a steady
# positive `dz`; a turn shows in `ry`. `ds.ego_motion_transform(pad_first=True)` returns `[B, T, 6]` with a zero
# row for frame 0 instead.

# %%
# identical to the pairwise reference implementation used by the authors' get_instructions.py
ref = np.stack([relative_motion(poses[0, i], poses[0, i + 1]) for i in range(T - 1)])
print("matches relative_motion pairwise:", np.allclose(deltas[0], ref, atol=1e-5))
print("sample 0, first 3 deltas [dx dy dz rx ry rz]:\n", np.round(deltas[0, :3], 4))

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
ego = torch.cat([torch.zeros(B, 1, 6), batch["ego"]], dim=1)      # [B, T, 6]: delta into frame t (0 for frame 0)
x_frames = frames.float().div_(255)                               # [B, T, 3, H, W] → CNN, e.g. x_frames.flatten(0, 1)
print("cnn input", tuple(x_frames.shape), "| rnn side input", tuple(ego.shape), "| target = embeddings shifted by one frame")
print("per-batch ego-motion scale (std):", np.round(ego.std(dim=(0, 1)).numpy(), 4), " ← normalise per dataset before the RNN")

# %% [markdown]
# ## 5. Seed reproducibility and throughput (disk → page cache)
#
# (Earlier cells already touched part of these 1,024 clips, so the first residency is not 0; the standalone benchmark
# `benchmarks/bench_spatialvid_window.py --drop-cache` evicts the pages first.)

# %%
def first_batch_signature(seed):
    l = make_loader(clips, workers=16, seed=seed, n=64)
    b = next(iter(l)); l.shutdown()
    return b["video_rec"].numpy().copy(), b["video_t0"].numpy().copy(), b["video"][:, 0].float().mean().item()

a1, a2, a3 = first_batch_signature(0), first_batch_signature(0), first_batch_signature(1)
print("same seed → same records, t0, pixels:", np.array_equal(a1[0], a2[0]), np.allclose(a1[1], a2[1]), a1[2] == a2[2])
print("other seed → different batch     :", not np.array_equal(a1[0], a3[0]))

# %%
N = 1024
loader = make_loader(clips, workers=os.cpu_count(), n=N)
print(f"page-cache residency before: {loader.page_cache_residency():.2f}")
for ep in range(2):
    t = time.perf_counter(); n = 0
    for b in loader:                                   # each pass = a new epoch: new order, new t0 draws
        n += b["video"].shape[0]
    dt = time.perf_counter() - t
    print(f"epoch {ep + 1}: {n / dt:,.1f} windows/s  ({n * T / dt:,.0f} frames/s, {n} windows, residency now {loader.page_cache_residency():.2f})")
loader.shutdown()

# %% [markdown]
# On machina (64 CPU decoders, 15 fps store on NVMe) this reaches ~160 windows/s ≈ 19k frames/s at 224×398; off a
# network mount the first epoch is disk-bound and the second matches (page cache). Full table: design doc §3.
