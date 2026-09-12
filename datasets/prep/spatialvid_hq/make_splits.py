"""Stage 2: source-level stratified train/val split -> splits/<version>.parquet + <version>.report.md

Unit of assignment is the source video (YouTube id): a source is entirely train or entirely val, so clips
from one recording never straddle the split. Sources are assigned greedily, largest strata first, to match
the marginal distributions of the stratification dimensions between val and the whole dataset, subject to:
  * val target size in clips (--val-clips), reached with whole sources
  * prefer sources with <= --max-source-clips clips for val (long recordings stay in train)
  * every "main" stratum (>= --main-cell-frac of clips) gets >= --min-cell-sources val sources where possible

    uv run python -m datasets.prep.spatialvid_hq.make_splits --out <work dir> --version v1 --val-clips 12000 \
        [--dims scene_l1,tod,weather,crowd,motion] [--filter "annot_ok"] [--seed 0]

Extra stratification columns (e.g. `carrier` once the VLM pass exists) can be merged into index/clips.parquet
and named in --dims. The output has one row per clip: clip_id, source_id, split, strata key.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Layout


def derive_dims(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scene_l1"] = df["scene_l1"].where(df["scene_l1"].isin(["Urban", "Natural Landscape", "Rural", "Interior", "Waterfront"]), "Other")
    tod = df["time_of_day"].fillna("Unknown")
    df["tod"] = np.select([tod.str.startswith("Daytime"), tod.eq("Night"), tod.str.startswith(("Dawn", "Dusk"))], ["day", "night", "dawn_dusk"], "unknown")
    w = df["weather"].fillna("Unknown"); df["weather_c"] = w.where(w.isin(["Sunny", "Cloudy", "Rainy", "Snowy"]), "other")
    c = df["crowd_density"].fillna("Unknown"); df["crowd"] = c.where(c.isin(["Deserted", "Sparse", "Moderate", "Crowded"]), "other")
    mt = df["motion_tags"].fillna("")
    df["motion"] = np.select([mt.eq("stationary"), mt.eq("forward"), mt.str.contains("left|right")], ["stationary", "forward_only", "turning"], "other")
    df["dur_bucket"] = pd.cut(df["duration_s"], [0, 5, 10, 16], labels=["short", "mid", "long"]).astype(str)
    return df


def assign(df: pd.DataFrame, dims: list[str], val_clips: int, max_source_clips: int, main_cell_frac: float,
           min_cell_sources: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    df = df.copy(); df["cell"] = df[dims].astype(str).agg("|".join, axis=1)
    n = len(df); target_frac = val_clips / n
    cell_total = df["cell"].value_counts()
    main_cells = set(cell_total[cell_total >= main_cell_frac * n].index)
    # per-source summary: size, dominant cell, cell histogram
    src = df.groupby("source_id").agg(n=("clip_id", "size"))
    src_cells = df.groupby(["source_id", "cell"]).size()
    eligible = src[src.n <= max_source_clips].index.to_numpy(); rng.shuffle(eligible)
    val_sources: set[str] = set(); val_cell = Counter(); val_n = 0; val_src_per_cell = Counter()
    # pass 1: guarantee coverage of main cells with >= min_cell_sources sources each
    for cell in sorted(main_cells, key=lambda c: -cell_total[c]):
        cands = src_cells.xs(cell, level="cell")
        cands = cands[cands.index.isin(eligible) & ~cands.index.isin(val_sources)].sort_values(ascending=False)
        for sid in cands.index[: max(0, min_cell_sources - val_src_per_cell[cell])]:
            val_sources.add(sid); val_n += int(src.n[sid])
            for c2, k in src_cells.xs(sid, level="source_id").items(): val_cell[c2] += int(k); val_src_per_cell[c2] += 1
    # pass 2: greedy marginal matching until the clip target is reached
    deficit = lambda c: cell_total[c] * target_frac - val_cell[c]
    for sid in eligible:
        if val_n >= val_clips: break
        if sid in val_sources: continue
        cells = src_cells.xs(sid, level="source_id")
        gain = sum(min(k, max(0.0, deficit(c))) for c, k in cells.items()) / int(src.n[sid])
        if gain < 0.5: continue          # source would mostly overfill already-satisfied cells
        val_sources.add(sid); val_n += int(src.n[sid])
        for c, k in cells.items(): val_cell[c] += int(k)
    return df["source_id"].isin(val_sources).map({True: "val", False: "train"})


def report(df: pd.DataFrame, dims: list[str], path: Path, version: str) -> None:
    lines = [f"# split {version}", "", f"clips: {len(df):,}  sources: {df.source_id.nunique():,}", ""]
    vc = df["split"].value_counts(); lines.append("| split | clips | sources |"); lines.append("| --- | ---: | ---: |")
    for s in ("train", "val"):
        lines.append(f"| {s} | {vc.get(s, 0):,} | {df.loc[df.split == s, 'source_id'].nunique():,} |")
    for d in dims + ["dur_bucket"]:
        t = pd.crosstab(df[d], df["split"], normalize="columns") * 100
        lines += ["", f"## {d} (% of split)", "", "| value | train | val |", "| --- | ---: | ---: |"]
        for v, r in t.sort_values("train", ascending=False).head(25).iterrows():
            lines.append(f"| {v} | {r.get('train', 0):.1f} | {r.get('val', 0):.1f} |")
    val_src = df[df.split == "val"].groupby("source_id").size()
    lines += ["", f"val clips per source: median {val_src.median():.0f}, max {val_src.max()}; "
                  f"largest source share of val {val_src.max() / val_src.sum() * 100:.1f} %"]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path); ap.add_argument("--version", default="v1")
    ap.add_argument("--val-clips", type=int, default=12000); ap.add_argument("--max-source-clips", type=int, default=40)
    ap.add_argument("--main-cell-frac", type=float, default=0.005); ap.add_argument("--min-cell-sources", type=int, default=5)
    ap.add_argument("--dims", default="scene_l1,tod,weather_c,crowd,motion")
    ap.add_argument("--filter", default="annot_ok", help="pandas query applied before splitting (excluded clips get split='excluded')")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    lay = Layout(Path("/nonexistent"), a.out); lay.splits_dir.mkdir(parents=True, exist_ok=True)
    clips = derive_dims(pd.read_parquet(lay.index_dir / "clips.parquet"))
    dims = a.dims.split(",")
    keep = clips.query(a.filter) if a.filter else clips
    split = assign(keep, dims, a.val_clips, a.max_source_clips, a.main_cell_frac, a.min_cell_sources, a.seed)
    clips["split"] = "excluded"; clips.loc[keep.index, "split"] = split.values
    out = clips[["clip_id", "source_id", "group_id", "split"] + dims + ["dur_bucket"]]
    out.to_parquet(lay.splits_dir / f"{a.version}.parquet", index=False)
    report(out[out.split != "excluded"], dims, lay.splits_dir / f"{a.version}.report.md", a.version)
    print(out["split"].value_counts().to_string()); print(f"wrote {lay.splits_dir / a.version}.parquet and .report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
