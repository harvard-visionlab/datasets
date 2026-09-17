"""Stage 2: source-level stratified train/val split -> splits/<version>.parquet + <version>.report.md

Unit of assignment is the source video (YouTube id): a source is entirely train or entirely val, so clips
from one recording never straddle the split. Sources are assigned greedily, largest strata first, to match
the marginal distributions of the stratification dimensions between val and the whole dataset, subject to:
  * val target size in clips (--val-clips), reached with whole sources
  * prefer sources with <= --max-source-clips clips for val (long recordings stay in train)
  * every "main" stratum (>= --main-cell-frac of clips) gets >= --min-cell-sources val sources where possible

    uv run python -m datasets.prep.spatialvid_hq.make_splits --out <work dir> --version v1 --val-clips 12000 \
        [--dims scene_l1,tod,weather,crowd,motion] [--filter "annot_ok"] [--seed 0]

v2 additions: `--carrier index/carrier_v1.parquet` merges the clip carrier (usable in --dims); `--subset
subsets/<name>.parquet` makes the stratification targets and the report apply to that population (clips outside
it follow their source / channel); `--val-unit channel` holds out WHOLE CHANNELS (leave-channel-out val) instead of
whole videos, with eligibility `--max-unit-frac` (share of the population one unit may hold) and `--min-unit-clips`.

    uv run python -m datasets.prep.spatialvid_hq.make_splits --out O --version v2a --subset subsets/person_carried_v0.parquet \
        --dims scene_l1,tod,weather_c,crowd,motion,carrier --val-clips 21000 --val-unit channel --max-unit-frac 0.03
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
           min_cell_sources: int, seed: int, unit: str = "source_id", min_unit_clips: int = 1) -> tuple[pd.Series, set]:
    """Greedy assignment of whole `unit`s (videos or channels) to val, matching joint strata-cell counts.
    Returns (split per clip, set of val units). `max_source_clips` is the per-unit clip cap for val eligibility."""
    rng = np.random.default_rng(seed)
    df = df.copy(); df["cell"] = df[dims].astype(str).agg("|".join, axis=1)
    n = len(df); target_frac = val_clips / n
    cell_total = df["cell"].value_counts()
    main_cells = set(cell_total[cell_total >= main_cell_frac * n].index)
    # per-unit summary: size and cell histogram
    src = df.groupby(unit).agg(n=("clip_id", "size"))
    src_cells = df.groupby([unit, "cell"]).size()
    eligible = src[(src.n <= max_source_clips) & (src.n >= min_unit_clips)].index.to_numpy(); rng.shuffle(eligible)
    val_sources: set[str] = set(); val_cell = Counter(); val_n = 0; val_src_per_cell = Counter()
    # pass 1: guarantee coverage of main cells with >= min_cell_sources sources each
    for cell in sorted(main_cells, key=lambda c: -cell_total[c]):
        cands = src_cells.xs(cell, level="cell")
        cands = cands[cands.index.isin(eligible) & ~cands.index.isin(val_sources)].sort_values(ascending=False)
        for sid in cands.index[: max(0, min_cell_sources - val_src_per_cell[cell])]:
            val_sources.add(sid); val_n += int(src.n[sid])
            for c2, k in src_cells.xs(sid, level=unit).items(): val_cell[c2] += int(k); val_src_per_cell[c2] += 1
    # pass 2: greedy marginal matching until the clip target is reached
    deficit = lambda c: cell_total[c] * target_frac - val_cell[c]
    def gain_of(sid):
        cells = src_cells.xs(sid, level=unit)
        return sum(min(k, max(0.0, deficit(c))) for c, k in cells.items()) / int(src.n[sid]), cells
    if unit == "source_id":                       # many small units: first-come in random order is fine and fast
        for sid in eligible:
            if val_n >= val_clips: break
            if sid in val_sources: continue
            gain, cells = gain_of(sid)
            if gain < 0.5: continue               # unit would mostly overfill already-satisfied cells
            val_sources.add(sid); val_n += int(src.n[sid])
            for c, k in cells.items(): val_cell[c] += int(k)
    else:                                         # few large units (channels): best-first, never overshoot the target by > 15 %
        pool = [sid for sid in eligible if sid not in val_sources]
        while pool and val_n < val_clips:
            scored = sorted(((gain_of(sid)[0], sid) for sid in pool if val_n + int(src.n[sid]) <= val_clips * 1.15), reverse=True)
            if not scored or scored[0][0] < 0.3: break
            sid = scored[0][1]; pool.remove(sid)
            val_sources.add(sid); val_n += int(src.n[sid])
            for c, k in src_cells.xs(sid, level=unit).items(): val_cell[c] += int(k)
    return df[unit].isin(val_sources).map({True: "val", False: "train"}), val_sources


def report(df: pd.DataFrame, dims: list[str], path: Path, version: str) -> None:
    lines = [f"# split {version}", "", f"clips: {len(df):,}  sources: {df.source_id.nunique():,}" + (f"  channels: {df.channel_id.nunique()}" if "channel_id" in df else ""), ""]
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
    if "channel_id" in df:
        vc = df[df.split == "val"].groupby("channel_id").size().sort_values(ascending=False)
        tc = df[df.split == "train"].groupby("channel_id").size().sort_values(ascending=False)
        lines += ["", f"channels: val {len(vc)}, train {len(tc)}, in both {len(set(vc.index) & set(tc.index))}; "
                      f"largest channel share of val {vc.iloc[0] / vc.sum() * 100:.1f} %, of train {tc.iloc[0] / tc.sum() * 100:.1f} %"]
        titles = df.drop_duplicates("channel_id").set_index("channel_id").get("channel_title")
        if titles is not None:
            lines += ["", "val channels (clips): " + ", ".join(f"{titles.get(c, c)} {n:,}" for c, n in vc.head(15).items())]
        # channel concentration inside each stratum value, val vs train
        lines += ["", "## top-channel share inside each stratum value (val | train)", "", "| dim | value | val top channel % | train top channel % |", "| --- | --- | ---: | ---: |"]
        for d in dims:
            for v in df[d].value_counts().head(6).index:
                sub = df[df[d] == v]
                tv = sub[sub.split == "val"].groupby("channel_id").size(); tt = sub[sub.split == "train"].groupby("channel_id").size()
                lines.append(f"| {d} | {v} | {tv.max() / tv.sum() * 100 if len(tv) else 0:.0f} | {tt.max() / tt.sum() * 100 if len(tt) else 0:.0f} |")
    if "val_kind" in df and (df.val_kind != "").any():
        vk = df[df.split == "val"].val_kind.value_counts()
        lines += ["", "val composition: " + ", ".join(f"{k}-holdout {n:,} clips" for k, n in vk.items())]
    if "carrier" in df:
        lines += ["", "## val fraction per carrier", "", "| carrier | clips | val % |", "| --- | ---: | ---: |"]
        for c, g in df.groupby("carrier"):
            lines.append(f"| {c} | {len(g):,} | {(g.split == 'val').mean() * 100:.1f} |")
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path); ap.add_argument("--version", default="v1")
    ap.add_argument("--val-clips", type=int, default=12000); ap.add_argument("--max-source-clips", type=int, default=40)
    ap.add_argument("--main-cell-frac", type=float, default=0.005); ap.add_argument("--min-cell-sources", type=int, default=5)
    ap.add_argument("--dims", default="scene_l1,tod,weather_c,crowd,motion")
    ap.add_argument("--filter", default="annot_ok", help="pandas query applied before splitting (excluded clips get split='excluded')")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--carrier", type=Path, default=None, help="index/carrier_v1.parquet (default if present): adds carrier + channel_id")
    ap.add_argument("--subset", type=Path, default=None, help="subsets/<name>.parquet: stratify and report on this population")
    ap.add_argument("--val-unit", choices=["source", "channel", "both"], default="source",
                    help="hold out whole videos, whole channels, or (both) a channel-holdout core of --val-channel-clips filled to --val-clips with whole videos")
    ap.add_argument("--val-channel-clips", type=int, default=10000, help="both: clips to hold out as whole channels before the video-level fill")
    ap.add_argument("--max-unit-frac", type=float, default=0.03, help="channel mode: max share of the population one val channel may hold")
    ap.add_argument("--min-unit-clips", type=int, default=100, help="channel mode: min clips for a channel to be val-eligible")
    a = ap.parse_args(argv)
    lay = Layout(Path("/nonexistent"), a.out); lay.splits_dir.mkdir(parents=True, exist_ok=True)
    clips = pd.read_parquet(lay.index_dir / "clips.parquet")
    carrier_path = a.carrier or (lay.index_dir / "carrier_v1.parquet")
    if carrier_path.exists():
        car = pd.read_parquet(carrier_path, columns=["clip_id", "channel_id", "carrier"])
        clips = clips.merge(car, on="clip_id", how="left"); clips["carrier"] = clips["carrier"].fillna("no_channel")
        ch = pd.read_parquet(lay.index_dir / "channels.parquet", columns=["channel_id", "channel_title"])
        clips = clips.merge(ch, on="channel_id", how="left")
    clips = derive_dims(clips)
    dims = a.dims.split(",")
    keep = clips.query(a.filter) if a.filter else clips
    if a.subset is not None:
        sub_ids = set(pd.read_parquet(a.subset, columns=["clip_id"])["clip_id"]); clips["in_subset"] = clips["clip_id"].isin(sub_ids)
        keep = keep[keep["clip_id"].isin(sub_ids)]
    clips["split"] = "excluded"; clips["val_kind"] = ""
    if a.val_unit == "source":
        unit = "source_id"
        split, val_units = assign(keep, dims, a.val_clips, a.max_source_clips, a.main_cell_frac, a.min_cell_sources, a.seed)
        clips.loc[keep.index, "split"] = split.values; clips.loc[keep.index[split.values == "val"], "val_kind"] = "video"
    else:
        keep = keep[keep["channel_id"].notna()]
        cap = int(a.max_unit_frac * len(keep)); n_ch = a.val_clips if a.val_unit == "channel" else a.val_channel_clips
        # whole channels: one unit per main cell is enough coverage (5 would force far too many channels in)
        split_c, val_channels = assign(keep, dims, n_ch, cap, a.main_cell_frac, min(a.min_cell_sources, 1), a.seed, unit="channel_id", min_unit_clips=a.min_unit_clips)
        clips.loc[keep.index, "split"] = split_c.values; clips.loc[keep.index[split_c.values == "val"], "val_kind"] = "channel"
        val_units = set(val_channels); unit = "channel_id"
        if a.val_unit == "both":
            rest = keep[split_c.values == "train"]
            # video-level fill: targets are the population marginals, minus what the channel core already holds
            split_s, val_sources = assign(rest, dims, a.val_clips - int((split_c == "val").sum()), a.max_source_clips, a.main_cell_frac, a.min_cell_sources, a.seed)
            clips.loc[rest.index[split_s.values == "val"], ["split", "val_kind"]] = ["val", "video"]
            val_units_src = set(val_sources)
    # clips outside the population (or filtered) follow their unit so no video/channel straddles the split
    rest_idx = clips.index.difference(keep.index)
    ok = clips.loc[rest_idx, "annot_ok"] if "annot_ok" in clips else pd.Series(True, index=rest_idx)
    ri = rest_idx[ok.values]
    is_val = clips.loc[ri, unit].isin(val_units)
    if a.val_unit == "both":
        is_val = is_val | clips.loc[ri, "source_id"].isin(val_units_src)
    clips.loc[ri, "split"] = np.where(is_val, "val", "train")
    # clips outside the population (or filtered) follow their unit so no video/channel straddles the split
    cols = ["clip_id", "source_id", "group_id", "split", "val_kind"] + [c for c in ("channel_id", "channel_title", "carrier", "in_subset") if c in clips] + dims + ["dur_bucket"]
    out = clips[list(dict.fromkeys(cols))]
    out.to_parquet(lay.splits_dir / f"{a.version}.parquet", index=False)
    pop = out.loc[keep.index] if a.subset is not None else out[out.split != "excluded"]
    report(pop, dims, lay.splits_dir / f"{a.version}.report.md", a.version)
    if a.subset is not None:
        print("population (subset):", pop.split.value_counts().to_dict(), "| all clips:", out.split.value_counts().to_dict())
    if a.val_unit != "source":
        print(f"val channels ({len(val_units)}):", ", ".join(sorted(str(clips.loc[clips.channel_id == u, 'channel_title'].iloc[0]) for u in val_units)))
        print("val_kind within population:", pop.val_kind.value_counts().to_dict())
    print(out["split"].value_counts().to_string()); print(f"wrote {lay.splits_dir / a.version}.parquet and .report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
