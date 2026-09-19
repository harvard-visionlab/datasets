"""Markdown count tables for the dataset card: split × (whole store | population), carrier breakdowns.

    uv run --no-sync --group video python -m datasets.prep.spatialvid_hq.card_counts --tree <tree> [--split-version v3] [--subset person_carried_v0]

Reads `splits/<version>.parquet` (every index clip, `in_subset` flag) joined with `index/clips.parquet` (durations, frame
counts) and prints the tables pasted into `datasets/cards/spatialvid-hq.md` §3, so the card can be regenerated rather than retyped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ORDER = ["train", "val", "test", "excluded"]


def split_table(df: pd.DataFrame) -> list[str]:
    lines = ["| split | clips | videos | channels | hours | frames (M) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for sp in ORDER + ["all"]:
        g = df if sp == "all" else df[df.split == sp]
        if len(g) == 0:
            continue
        lines.append(f"| {sp} | {len(g):,} | {g.source_id.nunique():,} | {g.channel_id.nunique():,} | "
                     f"{g.duration_s.sum() / 3600:,.0f} | {g.num_frames.sum() / 1e6:,.1f} |")
    return lines


def carrier_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    vc = df.groupby(["carrier", "split"]).size().unstack(fill_value=0).reindex(columns=cols, fill_value=0)
    vc["total"] = vc.sum(1); vc = vc.sort_values("total", ascending=False)
    lines = ["| carrier | " + " | ".join(vc.columns) + " |", "| --- |" + " ---: |" * len(vc.columns)]
    lines += [f"| {i} | " + " | ".join(f"{int(x):,}" for x in r) + " |" for i, r in vc.iterrows()]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--split-version", default="v3")
    ap.add_argument("--subset", default="person_carried_v0")
    a = ap.parse_args()
    s = pd.read_parquet(a.tree / "splits" / f"{a.split_version}.parquet")
    idx = pd.read_parquet(a.tree / "index" / "clips.parquet", columns=["clip_id", "duration_s", "num_frames"])
    s = s.merge(idx, on="clip_id", how="left")
    pop, rest = s[s.in_subset], s[~s.in_subset]
    out = [f"**Population `{a.subset}`**", ""] + split_table(pop) + ["", *carrier_table(pop, ["train", "val", "test"]), "",
           '**Whole store** (`subset="all"`)', ""] + split_table(s) + ["", "Clips outside the population, by carrier:", ""] + carrier_table(rest, ORDER)
    print("\n".join(out))


if __name__ == "__main__":
    main()
