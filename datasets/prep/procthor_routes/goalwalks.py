"""procthor-goalwalks: 1000-frame goal-directed walks (procthor-walks layout), planned in advance, checked per frame.

Plan-only stage (render env, A100: AI2-THOR is needed for the reachable grids; no frames are rendered):
    cd datasets/prep && python -m procthor_routes.goalwalks plan --houses-root <src> --split train --houses 0-9999:20 \
        --out <dir> [--shard k/n]
Per house: <out>/<split>/house_XXXXX/plan.npz (pos_u, psi_u [2, 1000]; pos_h, head_h) and plan.json (checks, stats).

Walk (one per seed 1000, 1001, as procthor-walks) = the procthor-routes goal_directed generator (plan.plan "goal" +
_walk_tour), unchanged, run to 1000 frames: start on a node with >= 0.5 m clearance, heading toward a first goal; legs to
goals 2-5 m geodesic away (first one within 100 deg ahead, then never straight back), smoothed shortest paths on the
FURNISHED house's 0.25 m reachable grid (weighted to keep ~0.75 m from walls/furniture), pure pursuit at 0.115 m/frame,
turn <= 0.15 rad/frame, slowing to 0.3x while the heading error is large; at a goal the next leg starts at once. Steps
up to 0.20 m off the grid are pulled back to 0.12 m (counted, as in routes). The one extension: a leg with a step
farther off (where the 61/121 generator discarded the whole window) is rolled back to its start and a new goal drawn
(a "replan"); 30 failed legs in a row restart the walk (next try, counted). Seeded [split, house, seed, try].

Checks (every frame / step; a walk is kept only if all pass):
  furnished   position on the furnished reachable set (within 0.13 m of a grid point, or inside a cell with 4 reachable
              corners: the generator's tolerance), also at 1/4, 1/2, 3/4 of every step (no corner cutting)
  empty       the same against the EMPTY house's own reachable grid (poses are identical in both cells)
  objects     camera point (x, 1.454 m, z) outside every furnished object's axis-aligned box grown by 0.10 m
              (near clip); min distance reported
  kinematics  step <= 0.115 m + 1e-9, |turn| <= 0.15 rad + 1e-9, finite
Reachable grid = AI2-THOR GetReachablePositions: the agent capsule fits there, i.e. clear of walls, doors frames and
furniture at all heights the capsule covers.

Degeneracy metrics per walk (windows = frames 0, 60, ..., 900, 61 frames each, as training samples them):
  replans        failed legs (rolled back and redrawn)
  frac_slow      steps < 0.9 V (slow-downs at sharp turns)
  straight_min   min over windows of net displacement / path length (0 = returns to start, 1 = straight)
  disp_min_m     min over windows of net displacement
  frac_near_wall frames whose nearest grid node has < 0.35 m clearance (the walkable floor's edge)
  uturns         goals whose next leg starts > 135 deg from the arrival heading
Proposed REJECT rule, report-only for now (`would_reject` per walk; not applied): replans > 10, straight_min < 0.25,
frac_near_wall > 0.15, frac_slow > 0.35.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import plan as P
from . import thor

N = 1000
SEEDS = (1000, 1001)
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
MAX_LEG_FAILS, MAX_TRIES = 30, 20
OBJ_MARGIN = 0.10
REJECT = {"replans": 10, "straight_min": 0.25, "frac_near_wall": 0.15, "frac_slow": 0.35}


class Grid:
    """Validity test against a reachable set (House.valid without the path graph)."""

    def __init__(self, reach):
        self.reach = reach; self.tree = cKDTree(reach); self.origin = reach.min(0)
        self.keys = set(map(tuple, np.round((reach - self.origin) / P.GRID).astype(int)))

    valid = P.House.valid


def walk(h: P.House, rng) -> tuple[dict | None, dict]:
    st = {"legs": 0, "goals": 0, "replans": 0, "uturns": 0}
    i0 = int(rng.choice(np.where(h.clear >= 0.5)[0] if (h.clear >= 0.5).any() else np.arange(len(h.reach))))
    g0 = P._goal_from(h, i0, rng)
    if g0 is None:
        return None, st | {"fail": "no_first_goal"}
    sp = P.smooth(h.path(i0, g0)); wk = P.Walker(h, h.reach[i0], math.atan2(*(sp[min(3, len(sp) - 1)] - sp[0])))
    cur, last, fails, goals_at = h.node(wk.p), None, 0, []
    while wk.n < N:
        g = P._goal_from(h, cur, rng, away_from=last, ahead=(wk.p, wk.psi, math.radians(100)) if st["goals"] == 0 and st["legs"] == 0 else None)
        if g is None:
            g = P._goal_from(h, cur, rng, away_from=last)
        if g is None:
            return None, st | {"fail": "no_goal"}
        save = (len(wk.P), wk.p.copy(), wk.psi, wk.fixes)
        path = h.path(h.node(wk.p), g)
        k = wk.pursue(path, N - wk.n)
        if not wk.ok or k == 0:
            n0, p0, psi0, f0 = save
            wk.P, wk.Psi = wk.P[:n0], wk.Psi[:n0]; wk.p, wk.psi, wk.fixes, wk.ok = p0, psi0, f0, True
            st["replans"] += 1; fails += 1
            if fails >= MAX_LEG_FAILS:
                return None, st | {"fail": "stuck"}
            continue
        if st["legs"]:
            sp = P.smooth(path); ang = math.atan2(*(sp[min(3, len(sp) - 1)] - sp[0]))
            st["uturns"] += int(abs(P.wrap(ang - save[2])) > math.radians(135))
        fails = 0; st["legs"] += 1
        if np.linalg.norm(wk.p - h.reach[g]) < 0.3:
            st["goals"] += 1; goals_at.append(wk.n - 1)
        last, cur = cur, g
    st["fixes"] = wk.fixes
    return {"pos_u": np.array(wk.P[:N]), "psi_u": np.array(wk.Psi[:N]), "goal_frames": goals_at}, st


def boxes(c) -> np.ndarray:
    """Furnished objects' AABBs spanning camera height, grown by OBJ_MARGIN: [n, 4] (xmin, xmax, zmin, zmax)."""
    out = []
    for o in c.last_event.metadata["objects"]:
        b = o.get("axisAlignedBoundingBox") or {}
        cen, sz = b.get("center"), b.get("size")
        if not cen or not sz:
            continue
        if cen["y"] - sz["y"] / 2 - OBJ_MARGIN <= thor.CAM_H <= cen["y"] + sz["y"] / 2 + OBJ_MARGIN:
            out.append([cen["x"] - sz["x"] / 2, cen["x"] + sz["x"] / 2, cen["z"] - sz["z"] / 2, cen["z"] + sz["z"] / 2])
    return np.array(out).reshape(-1, 4)


def box_dist(p: np.ndarray, bx: np.ndarray) -> np.ndarray:
    """Min distance from each point to any box (0 inside)."""
    if not len(bx):
        return np.full(len(p), np.inf)
    dx = np.maximum(np.maximum(bx[None, :, 0] - p[:, None, 0], p[:, None, 0] - bx[None, :, 1]), 0)
    dz = np.maximum(np.maximum(bx[None, :, 2] - p[:, None, 1], p[:, None, 1] - bx[None, :, 3]), 0)
    return np.hypot(dx, dz).min(1)


def check(w: dict, h: P.House, empty: Grid, bx: np.ndarray) -> dict:
    p, psi = w["pos_u"], w["psi_u"]
    sub = np.vstack([p] + [p[:-1] + f * (p[1:] - p[:-1]) for f in (0.25, 0.5, 0.75)])
    step = np.linalg.norm(np.diff(p, axis=0), axis=1); turn = np.abs(P.wrap(np.diff(psi)))
    od = box_dist(p, bx)
    c = {"furnished_bad": int((~h.valid(sub)).sum()), "empty_bad": int((~empty.valid(sub)).sum()),
         "object_bad": int((od < OBJ_MARGIN).sum()), "min_object_dist_m": float(od.min()),
         "step_bad": int((step > P.V + 1e-9).sum()), "turn_bad": int((turn > P.W_MAX + 1e-9).sum()),
         "finite": bool(np.isfinite(p).all() and np.isfinite(psi).all()), "n": int(len(p))}
    c["ok"] = c["n"] == N and c["finite"] and not any(c[k] for k in ("furnished_bad", "empty_bad", "object_bad", "step_bad", "turn_bad"))
    win = [(s0, s0 + 60) for s0 in range(0, N - 60, 60)]
    net = np.array([np.linalg.norm(p[b] - p[a]) for a, b in win]); pl = np.array([step[a:b].sum() for a, b in win])
    c |= {"path_m": float(step.sum()), "frac_slow": float((step < 0.9 * P.V).mean()),
          "mean_abs_turn": float(turn.mean()), "max_disp_m": float(np.linalg.norm(p - p[0], axis=1).max()),
          "straight_min": float((net / np.maximum(pl, 1e-9)).min()), "straight_median": float(np.median(net / np.maximum(pl, 1e-9))),
          "disp_min_m": float(net.min()), "frac_near_wall": float((h.clear[h.tree.query(p)[1]] < 0.35).mean())}
    return c


def degenerate(ck: dict, st: dict) -> list[str]:
    v = {"replans": st["replans"], "straight_min": ck["straight_min"], "frac_near_wall": ck["frac_near_wall"], "frac_slow": ck["frac_slow"]}
    return [k for k, lim in REJECT.items() if (v[k] < lim if k == "straight_min" else v[k] > lim)]


def parse_houses(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        rng, _, stride = part.partition(":")
        a, _, b = rng.partition("-")
        out += list(range(int(a), int(b or a) + 1, int(stride or 1)))
    return out


def cmd_plan(a) -> int:
    houses = thor.load_split(a.houses_root, a.split)
    ids = parse_houses(a.houses)
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/")); ids = ids[k::n]
    c = None
    for hi in ids:
        hd = a.out / a.split / f"house_{hi:05d}"
        if (hd / "plan.json").exists():
            continue
        hd.mkdir(parents=True, exist_ok=True); t = time.time()  # noqa: E702
        house = houses[hi]
        if c is None:
            c = thor.controller(house)
        else:
            thor.reset(c, house)
        reach = thor.reachable_positions(c); bx = boxes(c)
        thor.reset(c, thor.empty_house(house)); empty = Grid(thor.reachable_positions(c))
        t_load = time.time() - t; t = time.time()
        np.save(hd / "reach.npy", reach); np.save(hd / "reach_empty.npy", empty.reach)
        h = P.House(reach); t_graph = time.time() - t
        walks, meta = [], {"house_idx": hi, "split": a.split, "n_rooms": len(house["rooms"]), "n_objects": len(house["objects"]),
                           "n_reachable": len(reach), "n_reachable_empty": len(empty.reach), "n_boxes": len(bx), "walks": []}
        t = time.time()
        for seed in SEEDS:
            tries, rec = 0, {"seed": seed, "fails": []}
            while tries < MAX_TRIES:
                rng = np.random.default_rng([SPLIT_CODE[a.split], hi, seed, tries]); tries += 1
                w, st = walk(h, rng)
                if w is None:
                    rec["fails"].append(st); continue
                ck = check(w, h, empty, bx)
                if not ck["ok"]:
                    rec["fails"].append(st | {"fail": "check", "check": ck}); continue
                rec["would_reject"] = degenerate(ck, st)
                rec |= {"tries": tries, "stats": st, "check": ck}; walks.append(w); break
            else:
                rec["tries"] = tries
            rec["ok"] = "check" in rec and rec["check"]["ok"]
            meta["walks"].append(rec)
        meta["timing_s"] = {"load": t_load, "graph": t_graph, "plan": time.time() - t}
        if walks:
            hb = [[thor.to_habitat(x, z, math.degrees(q)) for (x, z), q in zip(w["pos_u"], w["psi_u"])] for w in walks]
            hb = np.array(hb)
            np.savez_compressed(hd / "plan.npz", pos_u=np.stack([w["pos_u"] for w in walks]),
                                psi_u=np.stack([w["psi_u"] for w in walks]), pos_h=hb[..., :2], head_h=hb[..., 2],
                                seeds=np.array([r["seed"] for r in meta["walks"] if r["ok"]]),
                                goal_frames=np.array([np.pad(np.array(w["goal_frames"], int), (0, 200 - len(w["goal_frames"])), constant_values=-1) for w in walks]))
        tmp = hd / "plan.json.tmp"; tmp.write_text(json.dumps(meta, indent=1, default=float)); tmp.replace(hd / "plan.json")
        print(f"house {hi}: rooms {meta['n_rooms']} reach {len(reach)} " +
              " ".join(f"s{r['seed']}:{'ok' if r['ok'] else 'FAIL'}/{r['tries']}" for r in meta["walks"]) +
              f"  load {t_load:.1f}s graph {t_graph:.1f}s plan {meta['timing_s']['plan']:.1f}s", flush=True)
    return 0


def _q(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "p5": float(np.quantile(x, .05)), "p50": float(np.median(x)), "p95": float(np.quantile(x, .95)),
            "min": float(x.min()), "max": float(x.max())}


def draw(hd: Path, wi: int, title: str, out: Path, px=40):
    from PIL import Image, ImageDraw
    reach = np.load(hd / "reach.npy"); z = np.load(hd / "plan.npz"); p = z["pos_u"][wi]; gf = z["goal_frames"][wi]
    lo = reach.min(0) - 0.5; hi = reach.max(0) + 0.5; W, H = ((hi - lo) * px).astype(int) + 1
    im = Image.new("RGB", (W, H + 22), (255, 255, 255)); d = ImageDraw.Draw(im)
    xy = lambda q: ((q[0] - lo[0]) * px, H - (q[1] - lo[1]) * px + 22)  # noqa: E731
    for q in reach:
        x, y = xy(q); d.rectangle([x - px * .125, y - px * .125, x + px * .125, y + px * .125], fill=(225, 228, 225))
    n = len(p)
    for i in range(n - 1):
        f = i / n; d.line([xy(p[i]), xy(p[i + 1])], fill=(int(40 + 200 * f), 60, int(220 - 180 * f)), width=2)
    for g in gf[gf >= 0]:
        x, y = xy(p[g]); d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(0, 0, 0))
    x, y = xy(p[0]); d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(0, 150, 0))
    d.text((4, 4), title, fill=(0, 0, 0)); im.save(out)


def cmd_report(a) -> int:
    root = a.out / a.split
    rows = [json.loads(f.read_text()) for f in sorted(root.glob("house_*/plan.json"))]
    W = [(m, w) for m in rows for w in m["walks"]]
    ok = [(m, w) for m, w in W if w["ok"]]
    fails = {}
    for _, w in W:
        for f in w["fails"]:
            fails[f["fail"]] = fails.get(f["fail"], 0) + 1
    ck = lambda k: [w["check"][k] for _, w in ok]  # noqa: E731
    st = lambda k: [w["stats"][k] for _, w in ok]  # noqa: E731
    rep = {"houses": len(rows), "walks": len(W), "walks_ok": len(ok),
           "houses_all_ok": sum(all(w["ok"] for w in m["walks"]) for m in rows),
           "one_room_houses": sum(m["n_rooms"] == 1 for m in rows),
           "tries_hist": np.bincount([w["tries"] for _, w in ok]).tolist() if ok else [],
           "failed_tries_by_reason": fails,
           "would_reject": {"any": sum(bool(w["would_reject"]) for _, w in ok),
                            **{k: sum(k in w["would_reject"] for _, w in ok) for k in REJECT}},
           "kept_walks": {k: _q(ck(k)) for k in ("path_m", "frac_slow", "straight_min", "straight_median", "disp_min_m",
                                                  "frac_near_wall", "min_object_dist_m", "max_disp_m", "mean_abs_turn")}
           | {k: _q(st(k)) for k in ("replans", "legs", "goals", "uturns", "fixes")},
           "plan_s_median": float(np.median([m["timing_s"]["plan"] for m in rows])),
           "load_s_median": float(np.median([m["timing_s"]["load"] for m in rows])),
           "failed_walks": [(m["house_idx"], w["seed"], m["n_rooms"], m["n_reachable"], [f["fail"] for f in w["fails"]][-3:])
                            for m, w in W if not w["ok"]]}
    (a.out / f"report_{a.split}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items()}, indent=1))
    figs = a.out / f"figs_{a.split}"; figs.mkdir(exist_ok=True)
    def wi(m, w): return [r["seed"] for r in m["walks"] if r["ok"]].index(w["seed"])  # noqa: E301
    picks = {}
    for key, rev in (("straight_min", False), ("replans", True), ("frac_near_wall", True), ("frac_slow", True)):
        srt = sorted(ok, key=lambda mw: (mw[1]["stats"] if key == "replans" else mw[1]["check"])[key], reverse=rev)
        picks[f"worst_{key}"] = srt[:2]
    rng = np.random.default_rng(0)
    picks["random"] = [ok[i] for i in rng.choice(len(ok), min(6, len(ok)), replace=False)]
    for tag, lst in picks.items():
        for m, w in lst:
            c, s_ = w["check"], w["stats"]
            title = (f"{tag}  house {m['house_idx']} seed {w['seed']} rooms {m['n_rooms']}  straight_min {c['straight_min']:.2f} "
                     f"replans {s_['replans']} near_wall {c['frac_near_wall']:.2f} slow {c['frac_slow']:.2f}")
            draw(root / f"house_{m['house_idx']:05d}", wi(m, w), title, figs / f"{tag}_h{m['house_idx']:05d}_s{w['seed']}.png")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--houses-root", type=Path, required=True); p.add_argument("--split", required=True)
    p.add_argument("--houses", required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--shard", default="")
    r = sub.add_parser("report"); r.add_argument("--out", type=Path, required=True); r.add_argument("--split", required=True)
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
