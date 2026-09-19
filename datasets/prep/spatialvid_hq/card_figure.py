"""Card figure: one 8 s window of a person-walking clip — frames, the interpolated camera path, and ego-motion traces.

    python -m datasets.prep.spatialvid_hq.card_figure --out datasets/cards/figures/spatialvid-hq-window.png [--seed 0]

Picks, among a few seeded val windows, the one with the clearest turn (largest net yaw under 120°), so the
frame content, the top-down path and the `ry` trace can be checked against each other by eye.
"""
import argparse, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch; torch.set_num_threads(1)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visionlab.datasets import load
from visionlab.datasets.video import camera_centers
from slipstream.decoders import DecodeVideoWindow
from slipstream.loader import SlipstreamLoader


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=48, help="candidate windows to draw from")
    a = ap.parse_args()
    T, RATE = 120, 15.0
    ds = load("spatialvid-hq", split="val", rate_hz=15, where="carrier == 'walk'")
    clips = ds.window_sampler(window_s=T / RATE, seed=a.seed).recs
    rng = np.random.default_rng(a.seed); clips = rng.choice(clips, a.n, replace=False)
    stage = DecodeVideoWindow(T=T, rate_hz=RATE, seed=a.seed, resize=224, device="cpu", num_workers=16)
    loader = SlipstreamLoader(ds, indices=clips, batch_size=a.n, shuffle=False, drop_last=False, image_field="video",
                              pipelines={"video": [stage]}, after_batch_transforms=[ds.ego_motion_transform()], verbose=False)
    b = next(iter(loader)); loader.shutdown()
    frames, t, poses, ego = b["video"], b["video_t_sec"].numpy(), b["poses"].numpy(), b["ego"].numpy()
    yaw = np.degrees(ego[:, :, 4].sum(1)); yaw = np.where(np.abs(yaw) > 120, 0, yaw)
    k = int(np.argmax(np.abs(yaw))); rec = int(b["video_rec"][k])
    row = ds.clips.set_index("record_idx").loc[rec]
    print(f"record {rec} clip {row.clip_id} channel {row.channel_title!r} net yaw {yaw[k]:.0f} deg, t0 {t[k, 0]:.2f} s")

    c = camera_centers(poses[k]); tt = t[k] - t[k, 0]; show = np.linspace(0, T - 1, 6).astype(int)
    fig = plt.figure(figsize=(16, 9)); gs = fig.add_gridspec(3, 6, height_ratios=[1.3, 1.7, 1.0], hspace=0.55, wspace=0.55)
    for col, j in enumerate(show):
        ax = fig.add_subplot(gs[0, col]); ax.imshow(frames[k, j].permute(1, 2, 0).numpy()); ax.axis("off")
        ax.set_title(f"frame {j}   t = {tt[j]:.1f} s", fontsize=10)
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(c[:, 0], c[:, 2], "-", color="0.6", lw=1); sc = ax.scatter(c[:, 0], c[:, 2], c=tt, cmap="viridis", s=16, zorder=3)
    for j in show:
        ax.annotate(f"frame {j}", (c[j, 0], c[j, 2]), textcoords="offset points", xytext=(6, -3), fontsize=8)
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.3); ax.set_xlabel("x (right)"); ax.set_ylabel("z (forward at t = 0)")
    ax.set_title("camera position, top-down\n(one dot per frame, coloured by time in s; per-clip units, not metres)", fontsize=10)
    fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03, ticks=[0, 2, 4, 6, 8])
    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(tt[1:], ego[k, :, 2], label="dz  forward"); ax.plot(tt[1:], ego[k, :, 0], label="dx  sideways"); ax.plot(tt[1:], ego[k, :, 1], label="dy  vertical")
    ax.grid(alpha=.3); ax.legend(fontsize=9, loc="center right"); ax.set_xlabel("t (s)"); ax.set_ylabel("pose units / frame")
    ax.set_title("ego-motion translation per frame, in the previous frame's axes\n(steps: source poses are ~5 Hz, interpolated linearly between keyframes)", fontsize=10)
    ax = fig.add_subplot(gs[2, :])
    ax.plot(tt[1:], np.degrees(ego[k, :, 4]), label="ry  yaw (negative = turning right)"); ax.plot(tt[1:], np.degrees(ego[k, :, 3]), label="rx  pitch")
    ax.plot(tt[1:], np.degrees(ego[k, :, 5]), label="rz  roll")
    for j in show:
        ax.axvline(tt[j], color="0.8", lw=0.8, zorder=0)
    ax.grid(alpha=.3); ax.legend(fontsize=9, ncol=3, loc="lower right"); ax.set_xlabel("t (s)"); ax.set_ylabel("deg / frame")
    ax.set_title(f"ego-motion rotation per frame (net yaw over the window {yaw[k]:+.0f}°); grey lines mark the frames shown above", fontsize=10)
    fig.suptitle(f"SpatialVID-HQ: one 8 s window at 15 Hz  (val split, walking, channel \u201c{row.channel_title}\u201d)", fontsize=12, y=0.93)
    fig.savefig(a.out, dpi=110, bbox_inches="tight"); print("wrote", a.out)


if __name__ == "__main__":
    main()
