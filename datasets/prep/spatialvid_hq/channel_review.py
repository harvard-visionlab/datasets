"""Stage 1c: channel-level carrier review -> index/channels_review.json (UI input) / index/channels.parquet (labels)

The 22 k source recordings come from ~136 YouTube channels and most channels post one kind of video, so the
carrier (walk / drive / drone / ...) is labelled per channel by a human, then refined per source. This module
builds the review payload for the labelling page and imports the labels back.

    python -m datasets.prep.spatialvid_hq.channel_review build  --out O [--samples 8] [--no-thumbs]
    python -m datasets.prep.spatialvid_hq.channel_review import --out O --labels labels.json

`build` writes `index/channels_review.json`: one entry per channel with counts, keyword rates, a draft label,
top tags and sample sources (title, duration, clip count, 120x90 YouTube thumbnail as a data URI).
`import` merges `{channel_id: {carrier, confidence, notes, ...}}` into `index/channels.parquet`.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Layout

CARRIERS = ["walk", "house", "drive", "drone", "train", "boat", "bike", "mixed", "other"]
KEYWORDS = {
    "walk": r"walk|stroll|hik|wander|on foot|trek|ramble",
    "house": r"house tour|home tour|apartment|real estate|mansion|for sale|penthouse|villa tour|interior",
    "drive": r"driv|dashcam|dash cam|road trip|highway|scenic byway",
    "drone": r"drone|aerial|dji|fpv|bird.?s.?eye|from above",
    "train": r"train ride|railway|railroad|tram|metro|cab ride|subway|locomotive|rail journey",
    "boat": r"boat|cruise|sail|kayak|ferry|yacht|canal ride",
    "bike": r"bike|cycl|bicycle|motorcycle|scooter|e-bike",
}
THUMB_URL = "https://i.ytimg.com/vi/{sid}/default.jpg"      # 120x90, ~3-5 KB


def keyword_rates(f: pd.DataFrame) -> pd.DataFrame:
    txt = (f["title"].fillna("") + " " + f["tags"].fillna("")).str.lower()
    return pd.DataFrame({k: txt.str.contains(p, regex=True) for k, p in KEYWORDS.items()}, index=f.index)


def draft_label(rates: pd.Series) -> tuple[str, str]:
    top = rates.sort_values(ascending=False)
    if top.iloc[0] >= 0.8 and (len(top) < 2 or top.iloc[1] < 0.3):
        return top.index[0], "clean"
    if top.iloc[0] >= 0.6:
        return top.index[0], "mostly"
    if top.iloc[0] < 0.2:
        return "other", "mixed"
    return "mixed", "mixed"


def fetch_thumb(sid: str) -> str | None:
    try:
        with urllib.request.urlopen(THUMB_URL.format(sid=sid), timeout=20) as r:
            return "data:image/jpeg;base64," + base64.b64encode(r.read()).decode()
    except Exception:
        return None


def build(lay: Layout, n_samples: int, thumbs: bool, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    src = pd.read_parquet(lay.index_dir / "sources.parquet")
    clips = pd.read_parquet(lay.index_dir / "clips.parquet", columns=["clip_id", "source_id"])
    n_clips_total = len(clips)
    per_src = clips.groupby("source_id").size().rename("n_clips")
    f = src[src["found"]].set_index("source_id").join(per_src)
    kw = keyword_rates(f)
    f = f.join(kw)
    channels = []
    for cid, g in f.groupby("channel_id"):
        rates = g[list(KEYWORDS)].mean()
        label, conf = draft_label(rates)
        tags = pd.Series([t.lower() for ts in g["tags"] for t in json.loads(ts or "[]")]).value_counts().head(15)
        # samples: top by clips + random, deduplicated, ordered by clips
        top = g.sort_values("n_clips", ascending=False).index[: n_samples // 2].tolist()
        rest = [s for s in g.index if s not in top]
        rand = list(rng.choice(rest, size=min(n_samples - len(top), len(rest)), replace=False)) if rest else []
        sample_ids = top + rand
        samples = [{"source_id": s, "title": g.at[s, "title"], "duration_s": None if pd.isna(g.at[s, "duration_s"]) else float(g.at[s, "duration_s"]),
                    "n_clips": int(g.at[s, "n_clips"]), "published": (g.at[s, "published_at"] or "")[:10]} for s in sample_ids]
        channels.append({
            "channel_id": cid, "channel_title": g["channel_title"].mode().iat[0],
            "n_sources": int(len(g)), "n_clips": int(g["n_clips"].sum()), "pct_clips": round(g["n_clips"].sum() / n_clips_total * 100, 2),
            "median_duration_min": round(float(g["duration_s"].median()) / 60, 1) if g["duration_s"].notna().any() else None,
            "keyword_rates": {k: round(float(v), 3) for k, v in rates.items()},
            "draft_carrier": label, "draft_confidence": conf,
            "top_tags": [{"tag": t, "n": int(n)} for t, n in tags.items()],
            "category_ids": {str(int(k)): int(v) for k, v in g["category_id"].dropna().value_counts().items()},
            "samples": samples,
        })
    channels.sort(key=lambda c: -c["n_clips"])
    if thumbs:
        ids = [s["source_id"] for c in channels for s in c["samples"]]
        with ThreadPoolExecutor(16) as ex:
            got = dict(zip(ids, ex.map(fetch_thumb, ids)))
        for c in channels:
            for s in c["samples"]:
                s["thumb"] = got.get(s["source_id"])
        print(f"thumbnails: {sum(v is not None for v in got.values()):,}/{len(ids):,}")
    missing = src[~src["found"]]
    payload = {
        "built": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "carriers": CARRIERS, "n_clips_total": n_clips_total, "n_sources_total": int(len(src)),
        "n_sources_missing": int(len(missing)), "n_clips_missing": int(per_src.reindex(missing["source_id"]).fillna(0).sum()),
        "channels": channels,
    }
    out = lay.index_dir / "channels_review.json"
    out.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"wrote {out}: {len(channels)} channels, {out.stat().st_size / 1e6:.1f} MB")
    return out


def import_labels(lay: Layout, labels_path: Path) -> Path:
    review = json.loads((lay.index_dir / "channels_review.json").read_text())
    labels = json.loads(labels_path.read_text())
    rows = []
    for c in review["channels"]:
        lab = labels.get(c["channel_id"], {})
        rows.append({
            "channel_id": c["channel_id"], "channel_title": c["channel_title"], "n_sources": c["n_sources"], "n_clips": c["n_clips"],
            "draft_carrier": c["draft_carrier"], "draft_confidence": c["draft_confidence"],
            "carrier": lab.get("carrier"), "confidence": lab.get("confidence"), "notes": lab.get("notes"),
            "labelled_at": lab.get("updated_at"), "labelled": bool(lab.get("carrier")),
        })
    df = pd.DataFrame(rows)
    out = lay.index_dir / "channels.parquet"
    df.to_parquet(out, index=False)
    done = df[df["labelled"]]
    print(f"wrote {out}: {len(done)}/{len(df)} channels labelled ({done['n_clips'].sum() / review['n_clips_total'] * 100:.1f} % of clips)")
    print(done.groupby("carrier")["n_clips"].sum().sort_values(ascending=False).to_string())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "import"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--no-thumbs", action="store_true")
    ap.add_argument("--labels", type=Path)
    a = ap.parse_args(argv)
    lay = Layout(Path("/nonexistent"), a.out)
    if a.cmd == "build":
        build(lay, a.samples, not a.no_thumbs)
    else:
        if not a.labels:
            sys.exit("--labels required")
        import_labels(lay, a.labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
