"""Pilot: plan + render route windows for a few val houses, both cells on identical paths. Render venv.

    cd datasets/prep && python -m procthor_routes.pilot --houses-root <procthor10k dir> --out <dir> \
        [--houses 3,17,...] [--per 5] [--lengths 61,121] [--plan-only]

Per house: <out>/house_XXXXX/<type>_<T>_<k>.npz (objects, empty uint8 [T+4,120,160,3]; pos_h float64 [T+4,2];
head_h float64 [T+4]; pos_u, psi_u) and house.json (windows: info + metrics; attempts per (type, T); timings).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from . import plan as P
from . import thor

MAX_TRIES = 40


def plan_house(h: P.House, hi: int, lengths, per: int):
    wins, attempts = [], {}
    for kind in P.TYPES:
        for T in lengths:
            got, tries, fails = 0, 0, {"plan": 0, "event": 0}
            while got < per and tries < MAX_TRIES:
                rng = np.random.default_rng([hi, P.TYPES.index(kind), T, got, tries])
                tries += 1
                w = P.plan(h, kind, T, rng)
                if w is None:
                    fails["plan"] += 1; continue
                w["metrics"] = P.metrics(w)
                if not w["metrics"]["event_ok"]:
                    fails["event"] += 1; continue
                w["k"] = got; w["seed"] = [hi, P.TYPES.index(kind), T, got, tries - 1]
                wins.append(w); got += 1
            attempts[f"{kind}_{T}"] = {"kept": got, "tries": tries} | fails
    return wins, attempts


def to_h(w):
    hb = [thor.to_habitat(x, z, math.degrees(p)) for (x, z), p in zip(w["pos_u"], w["psi_u"], strict=True)]
    a = np.array(hb, dtype=np.float64)
    return a[:, :2], a[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--houses", default="")
    ap.add_argument("--per", type=int, default=5)
    ap.add_argument("--lengths", default="61,121")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    val = thor.load_split(a.houses_root, "val")
    lengths = [int(x) for x in a.lengths.split(",")]
    houses = [int(x) for x in a.houses.split(",")] if a.houses else pick_houses(val)
    print("houses", houses, flush=True)
    c = None; summary = []
    for hi in houses:
        house = val[hi]; hd = a.out / f"house_{hi:05d}"; hd.mkdir(parents=True, exist_ok=True)
        t = time.time()
        if c is None:
            c = thor.controller(house)
        else:
            thor.reset(c, house)
        reach = thor.reachable_positions(c); t_load = time.time() - t
        np.save(hd / "reach.npy", reach)
        t = time.time(); h = P.House(reach); t_graph = time.time() - t
        t = time.time(); wins, attempts = plan_house(h, hi, lengths, a.per); t_plan = time.time() - t
        meta = {"house_idx": hi, "n_rooms": len(house["rooms"]), "room_types": [r["roomType"] for r in house["rooms"]],
                "n_objects": len(house["objects"]), "time_of_day": thor.time_of_day(house), "n_reachable": len(reach),
                "walkable_m2": len(reach) * P.GRID ** 2, "attempts": attempts,
                "timing_s": {"load": t_load, "graph": t_graph, "plan": t_plan}}
        print(f"house {hi}: rooms {meta['n_rooms']} reach {len(reach)} load {t_load:.1f}s graph {t_graph:.1f}s "
              f"plan {t_plan:.1f}s  " + " ".join(f"{k}:{v['kept']}/{v['tries']}" for k, v in attempts.items()), flush=True)
        frames = {}
        if not a.plan_only:
            for cell in ("objects", "empty"):
                scene = house if cell == "objects" else thor.empty_house(house)
                t = time.time(); thor.reset(c, scene); diff = thor.assert_agent_invisible(c, hi, cell); cam = thor.Camera(c)
                t_reset = time.time() - t
                t = time.time(); n = 0
                for w in wins:
                    frames[(cell, id(w))] = np.stack([cam.shot(x, z, math.degrees(p)) for (x, z), p in zip(w["pos_u"], w["psi_u"], strict=True)])
                    n += len(w["pos_u"])
                dt = time.time() - t
                w0 = wins[0]; again = cam.shot(*w0["pos_u"][0], math.degrees(w0["psi_u"][0]))
                det = int(np.abs(again.astype(int) - frames[(cell, id(w0))][0].astype(int)).max())
                meta["timing_s"][f"render_{cell}"] = {"reset": t_reset, "frames": n, "seconds": dt, "fps": n / dt,
                                                      "redraw_maxdiff": det, "agent_check_maxdiff": diff}
                print(f"  {cell:7s} reset {t_reset:.1f}s  {n} frames in {dt:.1f}s = {n / dt:.0f} fps  redraw diff {det}", flush=True)
        out_w = []
        for w in wins:
            pos_h, head_h = to_h(w)
            name = f"{w['kind']}_{w['T']}_{w['k']}"
            if not a.plan_only:
                np.savez_compressed(hd / f"{name}.npz", objects=frames[("objects", id(w))], empty=frames[("empty", id(w))],
                                    pos_h=pos_h, head_h=head_h, pos_u=w["pos_u"], psi_u=w["psi_u"])
            else:
                np.savez_compressed(hd / f"{name}.npz", pos_h=pos_h, head_h=head_h, pos_u=w["pos_u"], psi_u=w["psi_u"])
            out_w.append({"name": name, "kind": w["kind"], "T": w["T"], "seed": w["seed"], "info": w["info"], "metrics": w["metrics"]})
        meta["windows"] = out_w
        (hd / "house.json").write_text(json.dumps(meta, indent=1, default=float))
        summary.append(meta)
    if c is not None:
        c.stop()
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


def pick_houses(val, k=5):
    """k houses from val_unseen (0-199) spanning room counts: the 10th..90th percentile by number of rooms."""
    rooms = np.array([len(val[i]["rooms"]) for i in range(200)])
    order = np.argsort(rooms, kind="stable")
    return sorted(int(order[int(q * 199)]) for q in np.linspace(0.05, 0.95, k))


if __name__ == "__main__":
    main()
