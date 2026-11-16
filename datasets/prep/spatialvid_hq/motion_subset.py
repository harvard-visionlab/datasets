"""A motion-filtered subset of SpatialVID-HQ: drop clips where the camera mostly stands still or spins in place.

Reads the per-clip motion index `index/pose_speed_v0.parquet` (`pose_speed.py`) and a base subset, keeps clips with
`moving_frac >= --min-moving` and `spin_frac < --max-spin`, and writes `subsets/<name>.parquet` (base subset columns +
the motion index columns, so `load(..., subset=<name>, where="...")` can tighten further) and `<name>.report.md`.

  moving_frac  share of the clip's duration with camera speed >= 0.04 pose units/s (~0.3 m/s; walking ~0.16 u/s)
  spin_frac    share of the clip's duration rotating >= 30 deg/s (a fast pan or spin)

Defaults (2026-11-16): the 5 "low" review clips (p5-p15 speed; standing, panning, one in-place spin) have moving_frac
0.08-0.48; every "mid"/"high" clip >= 0.75. A clip that walks half the time and stands half the time is kept.
Loosen or tighten by building another subset with other flags (or `where=` on this one).

    uv run --no-sync python -m datasets.prep.spatialvid_hq.motion_subset --out <tree> [--name person_walking_v0 \
        --base person_carried_v0 --min-moving 0.5 --max-spin 0.1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MOTION_COLS = ["step_p50", "fwd_p50", "fwd_frac", "vert_frac", "turn_dps", "turn_p90", "moving_frac", "spin_frac"]


def build(tree: Path, name: str, base: str, min_moving: float, max_spin: float, index: str = "pose_speed_v0") -> pd.DataFrame:
    sub = pd.read_parquet(tree / "subsets" / f"{base}.parquet")
    mot = pd.read_parquet(tree / "index" / f"{index}.parquet", columns=["clip_id", "split"] + MOTION_COLS)
    df = sub.merge(mot, on="clip_id", how="left")
    if df.moving_frac.isna().any():
        raise SystemExit(f"{df.moving_frac.isna().sum()} {base} clips have no motion index; rerun pose_speed")
    stationary = df.moving_frac < min_moving
    spinning = ~stationary & (df.spin_frac >= max_spin)
    keep = df[~stationary & ~spinning].drop(columns=["split"])
    keep.to_parquet(tree / "subsets" / f"{name}.parquet", index=False)
    _report(tree, name, base, index, min_moving, max_spin, df, stationary, spinning, keep)
    return keep


def _report(tree, name, base, index, min_moving, max_spin, df, stationary, spinning, keep) -> None:
    def row(label, m):
        return (f"| {label} | {m.sum():,} | {m.mean() * 100:.1f} | {df.duration_s[m].sum() / 3600:,.0f} | "
                + " | ".join(f"{(m & (df.split == s)).sum():,}" for s in ("train", "val", "test")) + " |")
    L = [f"# subset {name}", "",
         f"definition: {base} clips with moving_frac >= {min_moving} and spin_frac < {max_spin} (per-clip motion index "
         f"`index/{index}.parquet`, built by `prep/spatialvid_hq/pose_speed.py`; this file by `motion_subset.py`).", "",
         "moving_frac = share of the clip's duration with camera speed >= 0.04 pose units/s (~0.3 m/s); spin_frac = share "
         "rotating >= 30 deg/s. Columns carried for `where=`: " + ", ".join(f"`{c}`" for c in MOTION_COLS) + ".", "",
         "| | clips | % of base | hours | train | val | test |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
         row(f"{base} (base)", pd.Series(True, index=df.index)),
         row("dropped: stationary", stationary), row("dropped: spinning", spinning),
         row(f"**{name}**", df.clip_id.isin(keep.clip_id)), "",
         "## kept clips by carrier", "", "| carrier | clips |", "| --- | ---: |"]
    L += [f"| {c} | {n:,} |" for c, n in keep.carrier.value_counts().items()]
    q = keep.step_p50.quantile([0.1, 0.5, 0.9])
    L += ["", f"kept clip speed (step_p50, pose units/s): p10 {q.iloc[0]:.3f}, median {q.iloc[1]:.3f}, p90 {q.iloc[2]:.3f}"]
    (tree / "subsets" / f"{name}.report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="SpatialVID-HQ-slipstream tree")
    ap.add_argument("--name", default="person_walking_v0")
    ap.add_argument("--base", default="person_carried_v0")
    ap.add_argument("--min-moving", type=float, default=0.5)
    ap.add_argument("--max-spin", type=float, default=0.1)
    a = ap.parse_args(argv)
    build(a.out, a.name, a.base, a.min_moving, a.max_spin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
