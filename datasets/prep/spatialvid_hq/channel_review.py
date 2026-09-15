"""Stage 1c: channel-level carrier review -> index/channels_review.json (UI input) / index/channels.parquet (labels)

The 22 k source recordings come from ~136 YouTube channels and most channels post one kind of video, so the
carrier (walk / drive / drone / ...) is labelled per channel by a human, then refined per source. This module
builds the review payload for the labelling page and imports the labels back.

    python -m datasets.prep.spatialvid_hq.channel_review build  --out O [--samples 8] [--no-thumbs]
    python -m datasets.prep.spatialvid_hq.channel_review sheets --out O [--res 456x256] [--clips 6] [--frames 3]
    python -m datasets.prep.spatialvid_hq.channel_review clips  --out O [--res 456x256] [--seconds 6]
    python -m datasets.prep.spatialvid_hq.channel_review minority --out O      # per minority keyword: clips from hit videos
    python -m datasets.prep.spatialvid_hq.channel_review import --out O --labels labels.json

`build` writes `index/channels_review.json`: one entry per channel with counts, keyword rates, a draft label,
top tags and sample sources (title, duration, clip count, 120x90 YouTube thumbnail as a data URI).
`sheets` decodes `--frames` frames from `--clips` store clips per channel (different sources) into one JPEG grid
per channel, `index/channel_sheets/<channel_id>.jpg` (rows = clips, columns = time), and records the clips in
`channels_review.json` under `sheet_clips`. The clips, not the YouTube videos, are what gets labelled: SpatialVID
keeps only motion-selected 2-15 s segments, so a talking-head channel can still contribute walkthrough b-roll.
`clips` re-encodes the same clips into one looping preview per channel, `index/channel_clips/<channel_id>.mp4`
(`--cols` x N grid, 240x135 tiles, 10 fps, H.264, first `--seconds`), so camera motion is visible in the browser.
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

# Carrier = who/what moves the camera. walk = carried by a walking person (first-person or following someone,
# indoors or out); rig = smooth mechanical/gimbal/crane/slider motion; other = tripod pans, talking head, timelapse.
# "house" (house tours) is a GENRE keyword, not a carrier: kept in KEYWORDS for evidence, never a label.
CARRIERS = ["walk", "rig", "drive", "drone", "train", "boat", "bike", "mixed", "other"]
GENRE_KEYWORDS = {"house"}
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


def keyword_evidence(g: pd.DataFrame, n_examples: int = 4) -> dict:
    """Per keyword: share of videos hit via title vs via tags only, the tags responsible, example titles."""
    title = g["title"].fillna("").str.lower(); tags = g["tags"].fillna("").str.lower()
    out = {}
    for k, p in KEYWORDS.items():
        in_title = title.str.contains(p, regex=True); in_tags = tags.str.contains(p, regex=True) & ~in_title
        if not (in_title.any() or in_tags.any()):
            continue
        hit_tags = pd.Series([t for ts in g.loc[in_tags | in_title, "tags"] for t in json.loads(ts or "[]")
                              if pd.Series([t.lower()]).str.contains(p, regex=True).iat[0]]).value_counts().head(5)
        out[k] = {"title_frac": round(float(in_title.mean()), 3), "tags_only_frac": round(float(in_tags.mean()), 3),
                  "tags": [{"tag": t, "n": int(n)} for t, n in hit_tags.items()],
                  "title_examples": g.loc[in_title, "title"].head(n_examples).tolist(),
                  "tags_only_examples": g.loc[in_tags, "title"].head(n_examples).tolist()}
    return out


def draft_label(rates: pd.Series) -> tuple[str, str]:
    top = rates.drop(labels=[k for k in GENRE_KEYWORDS if k in rates.index]).sort_values(ascending=False)
    if top.iloc[0] < 0.2 and rates.get("house", 0) >= 0.6:
        return "walk", "mostly"                      # house-tour genre: usually a walking operator, sometimes a rig
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
            "keyword_evidence": keyword_evidence(g),
            "draft_carrier": label, "draft_confidence": conf,
            "top_tags": [{"tag": t, "n": int(n)} for t, n in tags.items()],
            "category_ids": {str(int(k)): int(v) for k, v in g["category_id"].dropna().value_counts().items()},
            "samples": samples,
        })
    channels.sort(key=lambda c: -c["n_clips"])
    prev_path = lay.index_dir / "channels_review.json"
    if prev_path.exists():                                   # keep sheet_clips / preview from an earlier sheets/clips run
        prev = {c["channel_id"]: c for c in json.loads(prev_path.read_text())["channels"]}
        for c in channels:
            for k in ("sheet_clips", "sheet", "preview"):
                if k in prev.get(c["channel_id"], {}):
                    c[k] = prev[c["channel_id"]][k]
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


# ----------------------------------------------------------------------------------------------- contact sheets
_STORE = None


def _sheet_worker(args):
    """Decode frames for one channel's clips -> JPEG bytes (rows = clips, cols = frames). Runs in a worker."""
    global _STORE
    store_path, rows, n_frames, tile_w, tile_h = args
    import torch
    import torch.nn.functional as F
    from torchvision.io import encode_jpeg
    from ...video import VideoStore
    if _STORE is None:
        _STORE = VideoStore(store_path, device="cpu")
    tiles = []
    for r in rows:
        try:
            n = int(_STORE.raw(r["record_idx"], "num_frames"))
            idxs = np.linspace(0, max(n - 3, 0), n_frames).round().astype(int).tolist()   # avoid the last 2 frames
            fb = _STORE._decode(r["record_idx"], "get_frames_at", indices=idxs)
            x = fb.data.float()                                                          # [T,3,H,W]
            x = F.interpolate(x, size=(tile_h, tile_w), mode="bilinear", antialias=True, align_corners=False)
            row = torch.cat(list(x), dim=2)                                              # [3,H,T*W]
        except Exception as e:                                                            # keep the grid aligned
            row = torch.zeros(3, tile_h, tile_w * n_frames); r["error"] = f"{type(e).__name__}: {e}"[:120]
        tiles.append(row)
    _STORE.close_decoders()
    grid = torch.cat(tiles, dim=1).clamp(0, 255).to(torch.uint8)
    return bytes(encode_jpeg(grid, quality=72).numpy()), rows


def sheets(lay: Layout, res: str, n_clips: int, n_frames: int, tile_w: int = 160, tile_h: int = 90, workers: int = 16, seed: int = 0) -> None:
    from concurrent.futures import ProcessPoolExecutor
    rng = np.random.default_rng(seed)
    review_path = lay.index_dir / "channels_review.json"
    review = json.loads(review_path.read_text())
    store_path = lay.store_dir(res)
    records = pd.read_parquet(store_path / "records.parquet")                            # record_idx, clip_id
    rec_of = dict(zip(records["clip_id"], records["record_idx"] if "record_idx" in records else records.index))
    clips = pd.read_parquet(lay.index_dir / "clips.parquet", columns=["clip_id", "source_id", "duration_s", "motion_tags", "scene_l1", "scene_type"])
    src = pd.read_parquet(lay.index_dir / "sources.parquet", columns=["source_id", "channel_id", "title"])
    clips = clips[clips["clip_id"].isin(rec_of)].merge(src, on="source_id", how="inner")
    out_dir = lay.index_dir / "channel_sheets"; out_dir.mkdir(exist_ok=True)
    jobs = []
    for ch in review["channels"]:
        g = clips[clips["channel_id"] == ch["channel_id"]]
        # one clip per source where possible, sources chosen at random
        picks = g.groupby("source_id", group_keys=False).apply(lambda x: x.sample(1, random_state=int(rng.integers(1 << 31))))
        if len(picks) < n_clips:
            extra = g.drop(picks.index).sample(min(n_clips - len(picks), len(g) - len(picks)), random_state=int(rng.integers(1 << 31)))
            picks = pd.concat([picks, extra])
        picks = picks.sample(min(n_clips, len(picks)), random_state=int(rng.integers(1 << 31)))
        rows = [{"clip_id": r.clip_id, "record_idx": int(rec_of[r.clip_id]), "source_id": r.source_id, "duration_s": round(float(r.duration_s), 1),
                 "motion_tags": r.motion_tags, "scene_l1": r.scene_l1, "scene_type": r.scene_type, "title": r.title} for r in picks.itertuples()]
        jobs.append((ch, (str(store_path), rows, n_frames, tile_w, tile_h)))
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        for ch, (jpg, rows) in zip([j[0] for j in jobs], ex.map(_sheet_worker, [j[1] for j in jobs])):
            (out_dir / f"{ch['channel_id']}.jpg").write_bytes(jpg)
            ch["sheet_clips"] = rows; ch["sheet"] = {"cols": n_frames, "rows": len(rows), "tile_w": tile_w, "tile_h": tile_h}
            done += 1
            if done % 20 == 0: print(f"  {done}/{len(jobs)} channels")
    review["sheets_res"] = res; review["carriers"] = CARRIERS
    review_path.write_text(json.dumps(review, ensure_ascii=False))
    n_err = sum(1 for ch in review["channels"] for r in ch.get("sheet_clips", []) if r.get("error"))
    print(f"wrote {len(jobs)} sheets to {out_dir} ({sum(f.stat().st_size for f in out_dir.glob('*.jpg')) / 1e6:.1f} MB), {n_err} clip decode errors")


def _clip_worker(args):
    """ffmpeg: 6 store clips (looped) -> one 3x2 grid mp4. Returns (channel_id, ok, message)."""
    import subprocess, tempfile
    store_path, cid, rows, seconds, out_path, tile_w, tile_h, ffmpeg, cols = args
    from ...video import VideoStore
    global _STORE
    if _STORE is None:
        _STORE = VideoStore(store_path, device="cpu")
    with tempfile.TemporaryDirectory() as td:
        ins, filt, names = [], [], []
        for i, r in enumerate(rows):
            f = Path(td) / f"c{i}.mp4"; f.write_bytes(_STORE.video_bytes(r["record_idx"]))
            ins += ["-stream_loop", "-1", "-i", str(f)]
            filt.append(f"[{i}:v]scale={tile_w}:{tile_h},setsar=1,fps=10[v{i}]")
            names.append(f"[v{i}]")
        layout = "|".join(f"{(i % cols) * tile_w}_{(i // cols) * tile_h}" for i in range(len(rows)))
        even = "pad=ceil(iw/2)*2:ceil(ih/2)*2[v]"                       # libx264 needs even dimensions
        if len(rows) == 1:
            graph = filt[0].replace("[v0]", "[g]") + f";[g]{even}"
        else:
            graph = ";".join(filt) + f";{''.join(names)}xstack=inputs={len(rows)}:layout={layout}:fill=black[g];[g]{even}"
        cmd = [ffmpeg, "-y", "-loglevel", "error", *ins, "-filter_complex", graph, "-map", "[v]", "-t", str(seconds),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out_path)]
        p = subprocess.run(cmd, capture_output=True, text=True)
    return cid, p.returncode == 0, p.stderr[-300:]


def clips_previews(lay: Layout, res: str, seconds: float, workers: int = 16, tile_w: int = 240, tile_h: int = 135, ffmpeg: str = "ffmpeg", cols: int = 6) -> None:
    from concurrent.futures import ProcessPoolExecutor
    review_path = lay.index_dir / "channels_review.json"
    review = json.loads(review_path.read_text())
    out_dir = lay.index_dir / "channel_clips"; out_dir.mkdir(exist_ok=True)
    jobs = [(str(lay.store_dir(res)), ch["channel_id"], ch["sheet_clips"], seconds, out_dir / f"{ch['channel_id']}.mp4", tile_w, tile_h, ffmpeg, cols)
            for ch in review["channels"] if ch.get("sheet_clips")]
    n_ok = 0
    with ProcessPoolExecutor(workers) as ex:
        for i, (cid, ok, msg) in enumerate(ex.map(_clip_worker, jobs), 1):
            n_ok += ok
            if not ok: print(f"  FAILED {cid}: {msg}")
            if i % 20 == 0: print(f"  {i}/{len(jobs)}")
    for ch in review["channels"]:
        n = len(ch.get("sheet_clips", []))
        ch["preview"] = {"cols": min(cols, n), "rows": -(-n // cols), "tile_w": tile_w, "tile_h": tile_h, "seconds": seconds} if (out_dir / f"{ch['channel_id']}.mp4").exists() else None
    review["carriers"] = CARRIERS
    review_path.write_text(json.dumps(review, ensure_ascii=False))
    print(f"wrote {n_ok}/{len(jobs)} previews to {out_dir} ({sum(f.stat().st_size for f in out_dir.glob('*.mp4')) / 1e6:.1f} MB)")


def title_exceptions(c: dict, min_frac: float = 0.01) -> tuple[str, list[tuple[str, float]]]:
    """Confidence + per-video exceptions derived from TITLE keyword shares (tags-only hits are uploader boilerplate).
    The dominant title keyword maps to the human's channel carrier; any other keyword with >= min_frac of titles
    becomes a per-video override (video title matches KEYWORDS[k] -> carrier k). Mirrors autoConf() in the page."""
    ev = c.get("keyword_evidence", {})
    ranked = sorted(((k, e["title_frac"]) for k, e in ev.items() if k not in GENRE_KEYWORDS), key=lambda kv: -kv[1])
    dominant = ranked[0][0] if ranked else None
    exc = [(k, f) for k, f in ranked if k != dominant and f >= min_frac]
    level = "clean" if not exc else ("mixed" if any(f >= 0.25 for _, f in exc) else "mostly")
    return level, exc


def minority_previews(lay: Layout, res: str, seconds: float, min_rate: float = 0.05, n_clips: int = 6, workers: int = 16,
                      tile_w: int = 240, tile_h: int = 135, ffmpeg: str = "ffmpeg", seed: int = 1) -> None:
    """For every non-dominant keyword with >= min_rate of a channel's videos: a 6x1 preview of clips drawn only from
    videos whose title/tags hit that keyword -> index/channel_clips/<channel_id>__<keyword>.mp4. Lets the reviewer
    check that e.g. 'walk'-tagged videos of a driving channel are still driving (tag boilerplate) rather than walking."""
    from concurrent.futures import ProcessPoolExecutor
    rng = np.random.default_rng(seed)
    review_path = lay.index_dir / "channels_review.json"
    review = json.loads(review_path.read_text())
    store_path = lay.store_dir(res)
    records = pd.read_parquet(store_path / "records.parquet", columns=["record_idx", "clip_id"])
    rec_of = dict(zip(records["clip_id"], records["record_idx"]))
    clips = pd.read_parquet(lay.index_dir / "clips.parquet", columns=["clip_id", "source_id", "duration_s", "motion_tags", "scene_l1", "scene_type"])
    src = pd.read_parquet(lay.index_dir / "sources.parquet", columns=["source_id", "channel_id", "title", "tags"])
    src = src[src["channel_id"].notna()]
    hits = keyword_rates(src.set_index("source_id")).reset_index()
    clips = clips[clips["clip_id"].isin(rec_of)].merge(src, on="source_id").merge(hits, on="source_id")
    out_dir = lay.index_dir / "channel_clips"; out_dir.mkdir(exist_ok=True)
    jobs, meta = [], []
    for ch in review["channels"]:
        rates = ch["keyword_rates"]; dominant = max(rates, key=rates.get)
        g = clips[clips["channel_id"] == ch["channel_id"]]
        ch["minority_previews"] = {}
        for k, rate in rates.items():
            if k == dominant or rate < min_rate:
                continue
            gk = g[g[k]]
            if gk.empty:
                continue
            picks = gk.groupby("source_id", group_keys=False).apply(lambda x: x.sample(1, random_state=int(rng.integers(1 << 31))))
            picks = picks.sample(min(n_clips, len(picks)), random_state=int(rng.integers(1 << 31)))
            rows = [{"clip_id": r.clip_id, "record_idx": int(rec_of[r.clip_id]), "source_id": r.source_id, "duration_s": round(float(r.duration_s), 1),
                     "motion_tags": r.motion_tags, "scene_l1": r.scene_l1, "title": r.title,
                     "hit": "title" if pd.Series([str(r.title).lower()]).str.contains(KEYWORDS[k], regex=True).iat[0] else "tags"} for r in picks.itertuples()]
            out = out_dir / f"{ch['channel_id']}__{k}.mp4"
            jobs.append((str(store_path), ch["channel_id"], rows, seconds, out, tile_w, tile_h, ffmpeg, n_clips)); meta.append((ch, k, rows))
    n_ok = 0
    with ProcessPoolExecutor(workers) as ex:
        for (ch, k, rows), (cid, ok, msg) in zip(meta, ex.map(_clip_worker, [j[0:9] for j in jobs])):
            n_ok += ok
            if not ok: print(f"  FAILED {cid} {k}: {msg}")
            ch["minority_previews"][k] = {"rate": ch["keyword_rates"][k], "n_sources_hit": int(len({r["source_id"] for r in rows})),
                                          "clips": rows, "cols": len(rows), "rows": 1, "tile_w": tile_w, "tile_h": tile_h, "seconds": seconds, "ok": ok}
    review_path.write_text(json.dumps(review, ensure_ascii=False))
    print(f"wrote {n_ok}/{len(jobs)} minority previews for {sum(1 for ch in review['channels'] if ch['minority_previews'])} channels")


def import_labels(lay: Layout, labels_path: Path) -> Path:
    review = json.loads((lay.index_dir / "channels_review.json").read_text())
    labels = json.loads(labels_path.read_text())
    rows = []
    for c in review["channels"]:
        lab = labels.get(c["channel_id"], {})
        level, exc = title_exceptions(c)
        rows.append({
            "channel_id": c["channel_id"], "channel_title": c["channel_title"], "n_sources": c["n_sources"], "n_clips": c["n_clips"],
            "draft_carrier": c["draft_carrier"], "draft_confidence": c["draft_confidence"],
            "carrier": lab.get("carrier"), "confidence": level, "title_exceptions": json.dumps(exc),
            "notes": lab.get("notes"), "labelled_at": lab.get("updated_at"), "labelled": bool(lab.get("carrier")),
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
    ap.add_argument("cmd", choices=["build", "sheets", "clips", "minority", "import"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--no-thumbs", action="store_true")
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--res", default="456x256"); ap.add_argument("--clips", type=int, default=12); ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--seconds", type=float, default=6.0); ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--cols", type=int, default=6)
    a = ap.parse_args(argv)
    lay = Layout(Path("/nonexistent"), a.out)
    if a.cmd == "build":
        build(lay, a.samples, not a.no_thumbs)
    elif a.cmd == "sheets":
        sheets(lay, a.res, a.clips, a.frames, workers=a.workers)
    elif a.cmd == "clips":
        clips_previews(lay, a.res, a.seconds, workers=a.workers, ffmpeg=a.ffmpeg, cols=a.cols)
    elif a.cmd == "minority":
        minority_previews(lay, a.res, a.seconds, workers=a.workers, ffmpeg=a.ffmpeg)
    else:
        if not a.labels:
            sys.exit("--labels required")
        import_labels(lay, a.labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
