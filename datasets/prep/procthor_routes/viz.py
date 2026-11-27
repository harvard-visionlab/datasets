"""Pilot figures: per-house trajectory sheets (PNG) and per-route GIFs (objects | empty | top-down map). Render venv
(+ matplotlib, pillow).

    cd datasets/prep && python -m procthor_routes.viz --pilot <dir> --out <dir> [--gif-house 42]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import plan as P

COL = {"goal": "#1f77b4", "turn": "#d62728", "outback": "#2ca02c", "loop": "#9467bd", "brownian": "#7f7f7f"}


def sheet(hd: Path, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = json.loads((hd / "house.json").read_text()); reach = np.load(hd / "reach.npy")
    lengths = sorted({w["T"] for w in meta["windows"]})
    fig, axs = plt.subplots(len(lengths), len(P.TYPES), figsize=(3.2 * len(P.TYPES), 3.3 * len(lengths)), squeeze=False)
    for r, T in enumerate(lengths):
        for c, kind in enumerate(P.TYPES):
            ax = axs[r, c]; ax.scatter(reach[:, 0], reach[:, 1], s=1.5, c="#cccccc", lw=0)
            ws = [w for w in meta["windows"] if w["kind"] == kind and w["T"] == T]
            for w in ws:
                z = np.load(hd / f"{w['name']}.npz"); p = z["pos_u"][:T]
                ax.plot(p[:P.WARMUP + 1, 0], p[:P.WARMUP + 1, 1], color="#999999", lw=0.8, ls="--")
                ax.plot(p[P.WARMUP:, 0], p[P.WARMUP:, 1], color=COL[kind], lw=1.0)
                ax.plot(*p[P.WARMUP], "o", ms=3, color="k"); ax.plot(*p[-1], "x", ms=4, color=COL[kind])
            a = meta["attempts"].get(f"{kind}_{T}", {})
            ax.set_title(f"{kind} T={T}  kept {a.get('kept', len(ws))}/{a.get('tries', '?')} tries", fontsize=8)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"val house {meta['house_idx']}: {meta['n_rooms']} rooms, {meta['walkable_m2']:.0f} m² walkable, "
                 f"{meta['n_objects']} objects  (grey dashed = 20-frame warm-up, dot = rollout start A, x = target frame)", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def gif(hd: Path, name: str, out: Path, scale=2):
    from PIL import Image, ImageDraw
    z = np.load(hd / f"{name}.npz"); reach = np.load(hd / "reach.npy")
    meta = next(w for w in json.loads((hd / "house.json").read_text())["windows"] if w["name"] == name)
    T = meta["T"]; p = z["pos_u"][:T]; psi = z["psi_u"][:T]
    H, W = 120 * scale, 160 * scale; M = H                      # map panel M x M
    lo = reach.min(0) - 0.5; span = (reach.max(0) + 0.5 - lo).max()
    def xy(q):
        return ((q[..., 0] - lo[0]) / span * (M - 1), (M - 1) - (q[..., 1] - lo[1]) / span * (M - 1))
    base = Image.new("RGB", (M, M), (250, 250, 250)); d = ImageDraw.Draw(base)
    rx, ry = xy(reach)
    for a, b in zip(rx, ry, strict=True):
        d.point((a, b), fill=(200, 200, 200))
    px, py = xy(p)
    d.line(list(zip(px[:P.WARMUP + 1], py[:P.WARMUP + 1], strict=True)), fill=(160, 160, 160), width=1)
    col = tuple(int(COL[meta["kind"]][i:i + 2], 16) for i in (1, 3, 5))
    d.line(list(zip(px[P.WARMUP:], py[P.WARMUP:], strict=True)), fill=col, width=2)
    frames = []
    for t in range(T):
        im = Image.new("RGB", (2 * W + M + 8, H), (255, 255, 255))
        im.paste(Image.fromarray(z["objects"][t]).resize((W, H), Image.NEAREST), (0, 0))
        im.paste(Image.fromarray(z["empty"][t]).resize((W, H), Image.NEAREST), (W + 4, 0))
        mp = base.copy(); md = ImageDraw.Draw(mp); cx, cy = px[t], py[t]
        md.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(0, 0, 0))
        hx, hy = cx + 12 * np.sin(psi[t]), cy - 12 * np.cos(psi[t])
        md.line((cx, cy, hx, hy), fill=(0, 0, 0), width=2)
        md.text((4, 4), f"{meta['kind']} T={T}  f{t:3d} {'warm-up' if t < P.WARMUP else 'rollout'}", fill=(0, 0, 0))
        im.paste(mp, (2 * W + 8, 0)); frames.append(im)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=66, loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gif-house", type=int, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    houses = sorted(a.pilot.glob("house_*"))
    for hd in houses:
        sheet(hd, a.out / f"sheet_{hd.name}.png")
        print("sheet", hd.name, flush=True)
    hd = a.pilot / f"house_{a.gif_house:05d}" if a.gif_house is not None else houses[0]
    ws = json.loads((hd / "house.json").read_text())["windows"]
    for kind in P.TYPES:
        for T in sorted({w["T"] for w in ws}):
            w = next((w for w in ws if w["kind"] == kind and w["T"] == T), None)
            if w and (hd / f"{w['name']}.npz").exists() and "objects" in np.load(hd / f"{w['name']}.npz").files:
                gif(hd, w["name"], a.out / f"{hd.name}_{w['name']}.gif")
                print("gif", w["name"], flush=True)


if __name__ == "__main__":
    main()
