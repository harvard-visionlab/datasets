"""Walking head motion ("gait") added to a planned camera path: preview stage, not in the released stores.

The path (x, z, heading) is unchanged; gait adds per-frame offsets driven by distance walked, so the camera bobs only
while it moves (amplitude scales with speed / V, none when turning in place):
    phase  phi = 2 pi s / L + phi0           s = metres walked, L = stride (two steps), per clip U(1.2, 1.4) m
    height y   = CAM_H - A_v cos(2 phi)       vertical bob, twice per stride (once per step)
    sideways   = A_l sin(phi)                 lateral sway, once per stride (perpendicular to heading)
    roll       = A_r sin(phi)  deg            tilt with the sway
    pitch      = A_p sin(2 phi) deg           nod, once per step
    yaw        = heading + A_y sin(phi) deg   small head turn, once per stride
Per clip: phi0 ~ U(0, 2 pi), amplitudes x U(0.85, 1.15). At V = 0.115 m/frame one step is ~5.6 frames.

Preview (render env, A100; cd datasets/prep):
    python -m procthor_routes.gait --houses-root <src> --render <R>/val --houses 42,7 --out <dir>
re-renders the released goal_directed clips (k = 0, both lengths, furnished) at each LEVEL and writes side-by-side
GIFs (steady | levels) plus the gait poses.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import thor
V = 0.115                    # plan.V, walking pace (m/frame); plan imports scipy

LEVELS = {   # A_v (m), A_l (m), A_r, A_p, A_y (deg)
    "subtle": dict(A_v=0.015, A_l=0.010, A_r=0.5, A_p=0.5, A_y=0.5),
    "medium": dict(A_v=0.025, A_l=0.020, A_r=1.0, A_p=1.0, A_y=1.5),
    "strong": dict(A_v=0.040, A_l=0.030, A_r=2.0, A_p=1.5, A_y=2.5),
}


def gait(pos_u: np.ndarray, psi_u: np.ndarray, level: dict, rng) -> dict:
    pos_u, psi_u = np.asarray(pos_u, float), np.asarray(psi_u, float)
    step = np.r_[0.0, np.linalg.norm(np.diff(pos_u, axis=0), axis=1)]
    s = np.cumsum(step)
    speed = np.convolve(np.r_[step[1:2], step[1:]], np.ones(3) / 3, mode="same")     # smoothed m/frame
    amp = np.clip(speed / V, 0.0, 1.0)
    L = rng.uniform(1.2, 1.4); phi = 2 * np.pi * s / L + rng.uniform(0, 2 * np.pi)
    k = {n: a * rng.uniform(0.85, 1.15) for n, a in level.items()}
    right = np.stack([np.cos(psi_u), -np.sin(psi_u)], 1)
    return dict(
        pos_u=pos_u + right * (amp * k["A_l"] * np.sin(phi))[:, None],
        y=thor.CAM_H - amp * k["A_v"] * np.cos(2 * phi),
        roll_deg=amp * k["A_r"] * np.sin(phi),
        pitch_deg=amp * k["A_p"] * np.sin(2 * phi),
        yaw_deg=np.degrees(psi_u) + amp * k["A_y"] * np.sin(phi),
        stride_m=L, amps=k)


def side_by_side(cols: list[np.ndarray], labels: list[str], out: Path, scale=2, fps=15):
    from PIL import Image, ImageDraw
    T, h, w = cols[0].shape[:3]; bar = 18
    frames = []
    for t in range(T):
        im = Image.new("RGB", (w * scale * len(cols), h * scale + bar), (20, 20, 20)); d = ImageDraw.Draw(im)
        for j, (c, lab) in enumerate(zip(cols, labels)):
            im.paste(Image.fromarray(c[t]).resize((w * scale, h * scale), Image.NEAREST), (j * w * scale, bar))
            d.text((j * w * scale + 6, 3), lab if j else f"{lab}  {t:3d}", fill=(235, 235, 235))
        frames.append(im)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses-root", type=Path, required=True)
    ap.add_argument("--render", type=Path, required=True, help="released render dir (<R>/val)")
    ap.add_argument("--houses", required=True)
    ap.add_argument("--lengths", default="61,121")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    val = thor.load_split(a.houses_root, "val"); a.out.mkdir(parents=True, exist_ok=True)
    c = None; meta = []
    for hi in [int(x) for x in a.houses.split(",")]:
        house = val[hi]
        if c is None:
            c = thor.controller(house)
        thor.reset(c, house); thor.assert_agent_invisible(c, hi, "objects"); cam = thor.Camera(c)
        for T in [int(x) for x in a.lengths.split(",")]:
            src = np.load(a.render / f"house_{hi:05d}" / f"goal_{T}_0.npz")
            steady = src["objects"][:T]
            again = np.stack([cam.shot(x, z, math.degrees(p)) for (x, z), p in zip(src["pos_u"][:T], src["psi_u"][:T])])
            cols, labels, rec = [steady], ["steady"], dict(house=hi, T=T, redraw_mean_absdiff=float(np.abs(again.astype(int) - steady).mean()))
            for i, (name, lev) in enumerate(LEVELS.items()):
                g = gait(src["pos_u"][:T], src["psi_u"][:T], lev, np.random.default_rng([hi, T, i]))
                cols.append(np.stack([cam.shot(x, z, yw, y, pt, rl) for (x, z), yw, y, pt, rl in
                                      zip(g["pos_u"], g["yaw_deg"], g["y"], g["pitch_deg"], g["roll_deg"])]))
                labels.append(name)
                np.savez_compressed(a.out / f"house_{hi:05d}_goal_gait_{T}_{name}.npz", frames=cols[-1],
                                    **{k: v for k, v in g.items() if isinstance(v, np.ndarray)})
                rec[name] = dict(stride_m=g["stride_m"], amps=g["amps"], y_range=[float(g["y"].min()), float(g["y"].max())])
            side_by_side(cols, labels, a.out / f"house_{hi:05d}_goal_gait_{T}.gif")
            meta.append(rec); print(json.dumps(rec), flush=True)
    (a.out / "preview.json").write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
