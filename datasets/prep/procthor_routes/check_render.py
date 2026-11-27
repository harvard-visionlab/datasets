"""Render-equivalence gate: re-render stored val_unseen frames at their stored poses and diff against the lossless
stores (byte-identical to Rupert's H5). Two steps, two venvs:

    # 1. datasets env: dump reference frames + poses from the local lossless stores (v0.25.0)
    python -m visionlab.datasets.prep.procthor_routes.check_render dump --store-root <dir> --out ref.npz
    # 2. render venv (ai2thor): re-render and compare
    python -m visionlab.datasets.prep.procthor_routes.check_render compare --ref ref.npz --houses-root <procthor10k dir>

Also prints the training-data step / turn distribution measured from the dumped walks' full pose tracks.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CELLS = {"objects": "procthor-walks-objects", "empty": "procthor-walks-empty"}
FRAMES = [0, 250, 500, 750, 999]


def dump(store_root: Path, out: Path, walks: int):
    import pandas as pd
    from slipstream.cache import OptimizedCache
    from torchcodec.decoders import VideoDecoder
    res = {}
    for cell, name in CELLS.items():
        store = store_root / f"{name}-lossless-160x120-val_unseen"
        rec = pd.read_parquet(store / "records.parquet")
        cache = OptimizedCache.load(store, verbose=False)
        fv, fp, fh = cache.fields["video"], cache.fields["positions"], cache.fields["headings"]
        sel = np.linspace(0, len(rec) - 1, walks).round().astype(int)
        fr, po, he, ids = [], [], [], []
        for i in sel:
            v = fv.load_batch(np.array([i]), parallel=False)
            blob = bytes(np.asarray(v["data"][0])[: int(v["sizes"][0])])
            dec = VideoDecoder(blob, dimension_order="NHWC")
            fr.append(np.stack([dec[k].numpy() for k in FRAMES]))
            po.append(np.asarray(fp.load_batch(np.array([i]), parallel=False)["data"][0]).reshape(-1, 2))
            he.append(np.asarray(fh.load_batch(np.array([i]), parallel=False)["data"][0]).reshape(-1))
            ids.append(rec["clip_id"].iloc[i])
        res[f"{cell}_frames"] = np.stack(fr); res[f"{cell}_pos"] = np.stack(po); res[f"{cell}_head"] = np.stack(he)
        res[f"{cell}_ids"] = np.array(ids)
    np.savez(out, frames_idx=np.array(FRAMES), **res)
    print(f"wrote {out}: {walks} walks per cell, frames {FRAMES}", flush=True)


def training_motion_stats(pos: np.ndarray, head: np.ndarray) -> dict:
    """Per-step |translation| (m) and |turn| (rad) percentiles from stored walks [n, 1000, 2] / [n, 1000]."""
    step = np.linalg.norm(np.diff(pos, axis=1), axis=-1).ravel()
    turn = np.abs(np.angle(np.exp(1j * np.diff(head, axis=1)))).ravel()
    q = [50, 68, 90, 95, 99, 99.9]
    return {"step_m": {f"p{p}": round(float(np.percentile(step, p)), 4) for p in q} | {"mean": round(float(step.mean()), 4),
            "frac_zero": round(float((step < 1e-6).mean()), 4)},
            "turn_rad": {f"p{p}": round(float(np.percentile(turn, p)), 4) for p in q},
            "pct_turn_le": {str(t): round(float((turn <= t).mean() * 100), 1) for t in (0.10, 0.15, 0.16, 0.20, 0.30)},
            "pct_step_le": {str(t): round(float((step <= t).mean() * 100), 1) for t in (0.10, 0.115, 0.13, 0.20, 0.30)},
            "n_steps": int(step.size)}


def compare(ref: Path, houses_root: Path, out_json: Path | None):
    from . import thor
    z = np.load(ref)
    val = thor.load_split(houses_root, "val")
    rep = {"training_motion": training_motion_stats(z["objects_pos"], z["objects_head"]), "frames": []}
    print(json.dumps(rep["training_motion"], indent=1), flush=True)
    c = None; worst = 0
    for cell in ("objects", "empty"):
        for w, cid in enumerate(z[f"{cell}_ids"]):
            hi = int(str(cid).split("house_")[1].split("/")[0])
            scene = val[hi] if cell == "objects" else thor.empty_house(val[hi])
            t = time.time()
            if c is None:
                c = thor.controller(scene)
            else:
                thor.reset(c, scene)
            thor.assert_agent_invisible(c, hi, cell)
            cam = thor.Camera(c)
            for k, fi in enumerate(z["frames_idx"]):
                x, zz, yaw = thor.from_habitat(*z[f"{cell}_pos"][w, fi], z[f"{cell}_head"][w, fi])
                img = cam.shot(x, zz, yaw)
                d = np.abs(img.astype(int) - z[f"{cell}_frames"][w, k].astype(int))
                worst = max(worst, int(d.max()))
                rep["frames"].append(dict(cell=cell, clip=str(cid), frame=int(fi), max=int(d.max()), mean=float(d.mean()),
                                          frac_diff=float((d.max(-1) > 0).mean())))
                print(f"{cell:7s} {cid} frame {fi:3d}: max|diff| {int(d.max()):3d} mean {d.mean():.4f} "
                      f"px differing {(d.max(-1) > 0).mean():.4%}  ({time.time() - t:.1f}s)", flush=True)
    rep["worst"] = worst
    print("RENDER CHECK", "PASS (pixel-identical)" if worst == 0 else f"max diff {worst}", flush=True)
    if out_json:
        out_json.write_text(json.dumps(rep, indent=1))
    c.stop()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("--store-root", type=Path, required=True); d.add_argument("--out", type=Path, required=True)
    d.add_argument("--walks", type=int, default=6)
    c = sub.add_parser("compare"); c.add_argument("--ref", type=Path, required=True); c.add_argument("--houses-root", type=Path, required=True)
    c.add_argument("--out-json", type=Path)
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a.store_root, a.out, a.walks)
    else:
        compare(a.ref, a.houses_root, a.out_json)


if __name__ == "__main__":
    main()
