"""Stage 2b: clip-level carrier assignment + named training subsets.

    python -m datasets.prep.spatialvid_hq.make_subset --out O                       # carrier_v1 + person_carried_v0
    python -m datasets.prep.spatialvid_hq.make_subset --out O --name walk_v0 --carriers walk

Carrier per clip (`index/carrier_v1.parquet`), from the channel review (`index/channels.parquet`, written by
`channel_review import`): a video takes the decision of the first matching per-keyword group of its channel
(`<keyword>:title` if the keyword is in the video title, else `<keyword>:tags` if only in its tags), otherwise the
channel's carrier. Videos of unreviewed channels -> "unlabelled"; videos with no YouTube metadata -> "no_channel".
Column `how` records which rule fired.

Subset (`subsets/<name>.parquet` + `.report.md`): carrier in --carriers, no "stationary" motion tag, speed
(move_dist / duration_s, scene-relative units) <= --max-speed. Duration floors are NOT part of a subset: the
window length chosen at load time implies them (see docs/plans/spatialvid-hq-subsets-and-loader.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .channel_review import KEYWORDS, GENRE_KEYWORDS, keyword_hits
from .common import Layout

# Vehicle keywords first: a video titled "Driving ..." on a walking channel is a drive video even if it also says "walk".
KEYWORD_PRIORITY = ["drive", "drone", "train", "boat", "bike", "walk", "house"]
SUBSET_COLUMNS = ["clip_id", "record_idx", "source_id", "channel_id", "carrier", "how", "duration_s", "num_frames", "fps",
                  "speed", "motion_tags"]


def assign_carriers(lay: Layout) -> pd.DataFrame:
    ch = pd.read_parquet(lay.index_dir / "channels.parquet")
    src = pd.read_parquet(lay.index_dir / "sources.parquet", columns=["source_id", "channel_id", "title", "tags", "found"])
    src = keyword_hits(src)
    chan_carrier = dict(zip(ch["channel_id"], ch["carrier"]))
    chan_groups = {cid: json.loads(g or "{}") for cid, g in zip(ch["channel_id"], ch["groups"])}

    def decide(r) -> tuple[str, str]:
        cid = r.channel_id
        if not isinstance(cid, str) or not cid:
            return "no_channel", "no_channel"
        base = chan_carrier.get(cid)
        if not isinstance(base, str) or not base:
            return "unlabelled", "unlabelled"
        groups = chan_groups.get(cid, {})
        for k in KEYWORD_PRIORITY:
            if getattr(r, f"{k}__title") and f"{k}:title" in groups:
                return groups[f"{k}:title"], f"title:{k}"
        for k in KEYWORD_PRIORITY:
            if getattr(r, f"{k}__tags") and f"{k}:tags" in groups:
                return groups[f"{k}:tags"], f"tags:{k}"
        return base, "channel"

    dec = [decide(r) for r in src.itertuples()]
    src["carrier"] = [d[0] for d in dec]; src["how"] = [d[1] for d in dec]
    clips = pd.read_parquet(lay.index_dir / "clips.parquet",
                            columns=["clip_id", "source_id", "duration_s", "num_frames", "fps", "move_dist", "motion_tags", "annot_ok"])
    out = clips.merge(src[["source_id", "channel_id", "carrier", "how"]], on="source_id", how="left")
    out["carrier"] = out["carrier"].fillna("no_channel"); out["how"] = out["how"].fillna("no_channel")
    out["speed"] = out["move_dist"] / out["duration_s"]
    path = lay.index_dir / "carrier_v1.parquet"
    out[["clip_id", "source_id", "channel_id", "carrier", "how", "speed"]].to_parquet(path, index=False)
    print(f"wrote {path}: {len(out):,} clips"); print(out["carrier"].value_counts().to_string()); print(out["how"].value_counts().to_string())
    return out


def build_subset(lay: Layout, carriers: pd.DataFrame, name: str, keep: list[str], max_speed: float, drop_stationary: bool) -> pd.DataFrame:
    df = carriers[carriers["annot_ok"] & carriers["carrier"].isin(keep)].copy()
    if drop_stationary:
        df = df[~df["motion_tags"].fillna("").str.contains("stationary")]
    df = df[df["speed"] <= max_speed]
    records = pd.read_parquet(lay.store_dir("456x256") / "records.parquet", columns=["record_idx", "clip_id"])
    df = df.merge(records, on="clip_id", how="inner")                       # only clips present in the stores
    df = df[SUBSET_COLUMNS].sort_values("record_idx").reset_index(drop=True)
    lay.out.joinpath("subsets").mkdir(exist_ok=True)
    out = lay.out / "subsets" / f"{name}.parquet"
    prev = pd.read_parquet(out, columns=["clip_id"]) if out.exists() else None
    df.to_parquet(out, index=False)
    report(lay, df, name, keep, max_speed, drop_stationary, prev)
    return df


def _md(series: pd.Series, col: str) -> str:
    """Two-column markdown table from a Series (no tabulate dependency)."""
    return "\n".join(["| value | " + col + " |", "| --- | ---: |"] + [f"| {k} | {v:,.1f} |" if isinstance(v, float) else f"| {k} | {v:,} |" for k, v in series.items()])


def report(lay: Layout, df: pd.DataFrame, name: str, keep: list[str], max_speed: float, drop_stationary: bool, prev: pd.DataFrame | None) -> None:
    n_all = len(pd.read_parquet(lay.index_dir / "clips.parquet", columns=["clip_id"]))
    L = [f"# subset {name}", "",
         f"definition: carrier in {keep}; {'no stationary motion tag; ' if drop_stationary else ''}speed = move_dist/duration_s <= {max_speed}; "
         f"clips present in the h265 stores. Carrier from `index/carrier_v1.parquet` (channel review + per-keyword video groups).", "",
         f"clips: {len(df):,} ({len(df) / n_all * 100:.1f} % of HQ)  hours: {df.duration_s.sum() / 3600:,.0f}  "
         f"frames: {df.num_frames.sum() / 1e6:.1f} M  videos: {df.source_id.nunique():,}  channels: {df.channel_id.nunique()}", ""]
    if prev is not None:
        a, b = set(prev.clip_id), set(df.clip_id)
        L += [f"vs previous file: {len(a & b):,} shared, {len(b - a):,} added, {len(a - b):,} removed", ""]
    L += ["## carrier", "", _md(df.carrier.value_counts(), "clips"), "",
          "## how the carrier was assigned", "", _md(df.how.value_counts(), "clips"), "",
          "## duration floors (window length implies them)", "", "| >= s | clips | % clips | % frames |", "| ---: | ---: | ---: | ---: |"]
    for d in (2, 4, 6, 8, 10):
        m = df.duration_s >= d
        L.append(f"| {d} | {m.sum():,} | {m.mean() * 100:.0f} | {df.num_frames[m].sum() / df.num_frames.sum() * 100:.0f} |")
    L += ["", "## fps", "", _md((df.fps.round().value_counts(normalize=True) * 100).round(1), "% clips"), ""]
    top = df.groupby("channel_id").size().sort_values(ascending=False)
    ch = pd.read_parquet(lay.index_dir / "channels.parquet", columns=["channel_id", "channel_title"]).set_index("channel_id")["channel_title"]
    L += ["## channel concentration", "", f"top channel {top.iloc[0] / len(df) * 100:.1f} % of clips, top 5 {top.iloc[:5].sum() / len(df) * 100:.1f} %", "",
          "| channel | clips | % |", "| --- | ---: | ---: |"] + [f"| {ch.get(c, c)} | {n:,} | {n / len(df) * 100:.1f} |" for c, n in top.head(10).items()]
    sp = lay.splits_dir / "v1.parquet"
    if sp.exists():
        s = pd.read_parquet(sp, columns=["clip_id", "split"]); m = df.merge(s, on="clip_id", how="left")
        L += ["", "## split v1 within the subset", "", _md(m.split.fillna("(none)").value_counts(), "clips")]
    (lay.out / "subsets" / f"{name}.report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:6]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default="person_carried_v0")
    ap.add_argument("--carriers", default="walk,rig")
    ap.add_argument("--max-speed", type=float, default=0.5)
    ap.add_argument("--keep-stationary", action="store_true")
    a = ap.parse_args(argv)
    lay = Layout(Path("/nonexistent"), a.out)
    carriers = assign_carriers(lay)
    build_subset(lay, carriers, a.name, a.carriers.split(","), a.max_speed, not a.keep_stationary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
