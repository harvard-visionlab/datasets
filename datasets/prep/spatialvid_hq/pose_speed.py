"""Per-clip camera speed of the SpatialVID-HQ person_carried_v0 population, from the MegaSaM poses (step 1 of the pose
scale question: are the poses already near-metric, i.e. do walking clips agree on a speed?).

MegaSaM reconstructs every clip on its own from monocular video, so translation is in per-clip units unless its
metric-depth alignment made them near-metric. If it did, person-carried clips should cluster near walking speed
(~1.0-1.5 m/s) with a narrow spread; if not, the spread is wide and a per-clip scale is needed. Rotation is
scale-free and is reported as a control (turning rate should not depend on the translation scale).

Per clip, from the ~5 Hz annotated poses (times = annot_frame_idx / source fps):
  path_speed     path length / annotated duration (units/s)
  step_p50       median per-step speed (units/s; robust to a pause or a glitch)
  fwd_p50        median forward (+z, camera axes) speed (units/s)
  net_speed      |end - start| / duration (straight-line)
  vert_frac      mean |dy| / step length (bob / stairs share)
  fwd_frac       mean dz / step length (1 = straight ahead)
  turn_dps       median |rotation| per second (deg/s; scale-free control)
  turn_p90       90th percentile of |rotation| per second (deg/s; fast pans / spins)
  moving_frac    share of the clip's duration with step speed >= MOVING_UPS (not standing still)
  spin_frac      share of the clip's duration with rotation >= SPIN_DPS (fast pan / spin)
  meta_speed     the authors' move_dist / duration_s (what make_subset caps at 0.5)

    uv run --no-sync --group video python -m datasets.prep.spatialvid_hq.pose_speed --out <tree> --workers 16
writes <tree>/index/pose_speed_v0.parquet and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["path_speed", "step_p50", "fwd_p50", "net_speed", "vert_frac", "fwd_frac", "turn_dps", "turn_p90", "moving_frac",
        "spin_frac"]
MOVING_UPS = 0.04      # pose units/s: ~p15 of clip speeds, a quarter of the forward-walker median (0.164)
SPIN_DPS = 30.0        # deg/s: a fast pan (walking turns are ~1-10 deg/s)


def clip_speeds(poses: np.ndarray, t: np.ndarray) -> dict:
    from visionlab.datasets.video import camera_centers, ego_motion

    c = camera_centers(poses).astype(np.float64)
    ego = ego_motion(poses).astype(np.float64)                 # (n-1, 6) camera-t axes: x right, y down, z forward
    dt = np.diff(t)
    ok = dt > 0
    if ok.sum() < 2:
        return dict.fromkeys(COLS, np.nan)
    step = np.linalg.norm(ego[:, :3], axis=1)
    dur = t[-1] - t[0]
    moving = step > 1e-9
    v, w = step[ok] / dt[ok], np.degrees(np.linalg.norm(ego[ok, 3:], axis=1) / dt[ok])
    return dict(
        path_speed=step.sum() / dur,
        step_p50=float(np.median(step[ok] / dt[ok])),
        fwd_p50=float(np.median(ego[ok, 2] / dt[ok])),
        net_speed=float(np.linalg.norm(c[-1] - c[0]) / dur),
        vert_frac=float(np.mean(np.abs(ego[moving, 1]) / step[moving])) if moving.any() else np.nan,
        fwd_frac=float(np.mean(ego[moving, 2] / step[moving])) if moving.any() else np.nan,
        turn_dps=float(np.median(w)),
        turn_p90=float(np.quantile(w, 0.9)),
        moving_frac=float(dt[ok][v >= MOVING_UPS].sum() / dt[ok].sum()),
        spin_frac=float(dt[ok][w >= SPIN_DPS].sum() / dt[ok].sum()),
    )


def group_speeds(args: tuple[str, set[str], dict[str, float]]) -> list[dict]:
    path, keep, fps = args
    a = pd.read_parquet(path, columns=["clip_id", "annot_ok", "annot_frame_idx", "poses"])
    out = []
    for r in a.itertuples(index=False):
        if r.clip_id not in keep or not r.annot_ok:
            continue
        poses = np.asarray(r.poses, np.float32).reshape(-1, 7)
        t = np.asarray(r.annot_frame_idx, np.float64) / fps[r.clip_id]
        out.append({"clip_id": r.clip_id, "n_annot": len(poses), **clip_speeds(poses, t)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="SpatialVID-HQ-slipstream tree")
    ap.add_argument("--subset", default="person_carried_v0")
    ap.add_argument("--split-version", default="v3")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    tree = Path(a.out)
    t0 = time.time()
    sub = pd.read_parquet(tree / "subsets" / f"{a.subset}.parquet")
    clips = pd.read_parquet(tree / "index" / "clips.parquet", columns=["clip_id", "fps", "move_dist", "duration_s"])
    split = pd.read_parquet(tree / "splits" / f"{a.split_version}.parquet", columns=["clip_id", "split"])
    df = sub[["clip_id", "channel_id", "source_id", "carrier", "record_idx"]].merge(clips, on="clip_id").merge(split, on="clip_id")
    keep, fps = set(df.clip_id), dict(zip(df.clip_id, df.fps))
    groups = sorted((tree / "index" / "annotations").glob("group_*.parquet"))
    rows: list[dict] = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(group_speeds, [(str(g), keep, fps) for g in groups])):
            rows += r
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(groups)} groups, {len(rows):,} clips, {time.time() - t0:.0f} s", flush=True)
    out = df.merge(pd.DataFrame(rows), on="clip_id")
    out["meta_speed"] = out.move_dist / out.duration_s
    dest = tree / "index" / f"pose_speed_{a.subset.split('_')[-1]}.parquet"
    out.to_parquet(dest, index=False)
    q = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    summary = {k: {"by_carrier": {c: out.loc[out.carrier == c, k].quantile(q).round(4).tolist() for c in ("walk", "rig")},
                   "all": out[k].quantile(q).round(4).tolist()} for k in COLS + ["meta_speed"]}
    print(json.dumps({"clips": len(out), "quantiles": q, "summary": summary}, indent=1))
    print(f"wrote {dest} ({len(out):,} clips) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
