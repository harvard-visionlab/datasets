"""Stage 1b: YouTube metadata for every source recording -> index/sources_raw/*.jsonl + index/sources.parquet

SpatialVID has no carrier label (walk / drive / drone / ...). Titles, descriptions, tags and channel names of
the source videos ("4K walking tour", "dashcam", "drone") label whole recordings at once, which is the split
unit. This fetches them once and archives the raw responses (videos get deleted/privated over time).

    uv run --group video python -m datasets.prep.spatialvid_hq.fetch_sources --out <work dir> \
        [--backend api|ytdlp] [--api-key K | env YOUTUBE_API_KEY] [--limit N] [--derive-only]

Backends
  api    YouTube Data API v3 `videos.list`, 50 ids per call, 1 quota unit per call (~450 units for HQ; the
         default project quota is 10,000/day). Needs an API key (Google Cloud console -> enable "YouTube Data
         API v3" -> Credentials -> API key).
  ytdlp  `yt_dlp` extract_info per id, no key, ~1-2 s per id, may be rate-limited. Fallback only.

Resumable: ids already present in the raw archive are skipped; re-running only derives the parquet.
Raw archive: `index/sources_raw/<backend>.jsonl`, one line per requested id
  {"source_id", "backend", "fetched_at", "found": bool, "item": <verbatim response item> | null}
Derived: `index/sources.parquet`, one row per source_id (see SOURCE_COLUMNS).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .common import Layout

API_URL = "https://www.googleapis.com/youtube/v3/videos"
API_PARTS = "snippet,contentDetails,statistics,status,topicDetails"
API_BATCH = 50

SOURCE_COLUMNS = [
    "source_id", "found", "backend", "fetched_at",
    "title", "description", "tags", "channel_id", "channel_title", "category_id", "topic_categories",
    "default_language", "default_audio_language", "published_at", "duration_s", "definition",
    "view_count", "like_count", "comment_count", "made_for_kids", "privacy_status", "license",
]

_ISO_DUR = re.compile(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_duration_s(s: str | None) -> float | None:
    if not s:
        return None
    m = _ISO_DUR.fullmatch(s)
    if not m:
        return None
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return float(d * 86400 + h * 3600 + mi * 60 + se)


def _to_int(x) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------------------------- raw archive
def raw_path(lay: Layout, backend: str) -> Path:
    return lay.index_dir / "sources_raw" / f"youtube_{backend}.jsonl"


def load_raw(path: Path) -> dict[str, dict]:
    """source_id -> archived line (last one wins)."""
    out: dict[str, dict] = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    out[rec["source_id"]] = rec
    return out


# ------------------------------------------------------------------------------------------------ backends
class QuotaExceeded(RuntimeError):
    pass


def api_fetch_batch(ids: list[str], key: str, retries: int = 6) -> dict[str, dict]:
    """videos.list for <= 50 ids. Returns {id: item} for ids that exist (missing = deleted/private)."""
    q = urllib.parse.urlencode({"part": API_PARTS, "id": ",".join(ids), "maxResults": API_BATCH, "key": key})
    req = urllib.request.Request(f"{API_URL}?{q}", headers={"Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
            return {it["id"]: it for it in data.get("items", [])}
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code == 403 and ("quotaExceeded" in body or "dailyLimitExceeded" in body):
                raise QuotaExceeded(body[:500])
            if e.code in (400, 401, 403):
                raise RuntimeError(f"HTTP {e.code}: {body[:500]}")
            wait = 2 ** attempt
            print(f"  HTTP {e.code}, retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            wait = 2 ** attempt
            print(f"  {type(e).__name__}: {e}, retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"giving up after {retries} attempts on batch starting {ids[0]}")


def api_item_to_row(sid: str, it: dict) -> dict:
    sn, cd, st, stat, tp = (it.get(k, {}) for k in ("snippet", "contentDetails", "status", "statistics", "topicDetails"))
    return {
        "source_id": sid, "title": sn.get("title"), "description": sn.get("description"),
        "tags": json.dumps(sn.get("tags", []), ensure_ascii=False), "channel_id": sn.get("channelId"),
        "channel_title": sn.get("channelTitle"), "category_id": _to_int(sn.get("categoryId")),
        "topic_categories": json.dumps(tp.get("topicCategories", [])), "default_language": sn.get("defaultLanguage"),
        "default_audio_language": sn.get("defaultAudioLanguage"), "published_at": sn.get("publishedAt"),
        "duration_s": iso_duration_s(cd.get("duration")), "definition": cd.get("definition"),
        "view_count": _to_int(stat.get("viewCount")), "like_count": _to_int(stat.get("likeCount")),
        "comment_count": _to_int(stat.get("commentCount")), "made_for_kids": st.get("madeForKids"),
        "privacy_status": st.get("privacyStatus"), "license": st.get("license"),
    }


def ytdlp_fetch_one(sid: str) -> dict | None:
    import yt_dlp  # optional dependency: uv pip install yt-dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True, "socket_timeout": 30}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={sid}", download=False)
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        if any(k in msg for k in ("Private video", "unavailable", "removed", "terminated", "not available")):
            return None
        raise
    keep = ("id", "title", "description", "tags", "categories", "channel", "channel_id", "uploader", "duration",
            "upload_date", "view_count", "like_count", "comment_count", "language", "availability", "license",
            "age_limit", "width", "height", "fps", "was_live", "live_status", "chapters")
    return {k: info.get(k) for k in keep}


def ytdlp_item_to_row(sid: str, it: dict) -> dict:
    up = it.get("upload_date")
    return {
        "source_id": sid, "title": it.get("title"), "description": it.get("description"),
        "tags": json.dumps(it.get("tags") or [], ensure_ascii=False), "channel_id": it.get("channel_id"),
        "channel_title": it.get("channel") or it.get("uploader"), "category_id": None,
        "topic_categories": json.dumps(it.get("categories") or []), "default_language": it.get("language"),
        "default_audio_language": None,
        "published_at": f"{up[:4]}-{up[4:6]}-{up[6:8]}T00:00:00Z" if up and len(up) == 8 else None,
        "duration_s": float(it["duration"]) if it.get("duration") is not None else None,
        "definition": "hd" if (it.get("height") or 0) >= 720 else ("sd" if it.get("height") else None),
        "view_count": _to_int(it.get("view_count")), "like_count": _to_int(it.get("like_count")),
        "comment_count": _to_int(it.get("comment_count")), "made_for_kids": None,
        "privacy_status": it.get("availability"), "license": it.get("license"),
    }


# --------------------------------------------------------------------------------------------------- derive
def derive(lay: Layout, ids: list[str]) -> pd.DataFrame:
    """Merge raw archives (api preferred over ytdlp) into one row per source id."""
    rows: dict[str, dict] = {}
    for backend, to_row in (("ytdlp", ytdlp_item_to_row), ("api", api_item_to_row)):   # api last -> wins
        for sid, rec in load_raw(raw_path(lay, backend)).items():
            if rec.get("found") and rec.get("item"):
                row = to_row(sid, rec["item"])
            elif sid in rows:
                continue                      # keep a found record from the other backend
            else:
                row = {"source_id": sid}
            row.update(found=bool(rec.get("found")), backend=backend, fetched_at=rec.get("fetched_at"))
            rows[sid] = row
    df = pd.DataFrame([rows.get(s, {"source_id": s, "found": False}) for s in ids])
    for c in SOURCE_COLUMNS:
        if c not in df:
            df[c] = None
    df = df[SOURCE_COLUMNS]
    df["found"] = df["found"].fillna(False).astype(bool)
    for c in ("view_count", "like_count", "comment_count", "category_id"):
        df[c] = pd.array(df[c], dtype="Int64")
    for c in SOURCE_COLUMNS:
        if c not in ("found", "duration_s", "made_for_kids", "view_count", "like_count", "comment_count", "category_id"):
            df[c] = df[c].astype("string")
    return df


def summary(df: pd.DataFrame, clips: pd.DataFrame) -> str:
    n_src = clips["source_id"].nunique()
    per_src = clips.groupby("source_id").size().rename("n_clips")
    d = df.set_index("source_id").join(per_src)
    fetched = d[d["fetched_at"].notna()]
    found = d[d["found"]]
    lines = [f"sources: {n_src:,}   fetched: {len(fetched):,}   found: {len(found):,} "
             f"({found['n_clips'].sum():,} clips = {found['n_clips'].sum() / len(clips) * 100:.1f} %)   "
             f"missing (deleted/private): {len(fetched) - len(found):,}"]
    if len(found):
        lines.append(f"duration: median {found['duration_s'].median() / 60:.0f} min, "
                     f"p90 {found['duration_s'].quantile(0.9) / 60:.0f} min; "
                     f"channels: {found['channel_id'].nunique():,}")
        top = found.groupby("channel_title")["n_clips"].agg(["sum", "size"]).sort_values("sum", ascending=False).head(15)
        lines += ["", "top channels (clips, sources):"] + [f"  {c[:50]:50s} {int(r['sum']):7,d} {int(r['size']):5,d}"
                                                            for c, r in top.iterrows()]
        cat = found.groupby("category_id")["n_clips"].sum().sort_values(ascending=False).head(8)
        lines += ["", "category_id -> clips: " + ", ".join(f"{k}:{v:,}" for k, v in cat.items())]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--backend", choices=["api", "ytdlp"], default="api")
    ap.add_argument("--api-key", default=os.environ.get("YOUTUBE_API_KEY"))
    ap.add_argument("--limit", type=int, default=None, help="fetch at most N not-yet-archived ids (smoke test)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between requests")
    ap.add_argument("--derive-only", action="store_true", help="skip fetching; rebuild sources.parquet from the archive")
    a = ap.parse_args(argv)

    lay = Layout(Path("/nonexistent"), a.out)
    clips = pd.read_parquet(lay.index_dir / "clips.parquet", columns=["clip_id", "source_id"])
    ids = sorted(clips["source_id"].dropna().unique().tolist())
    print(f"{len(ids):,} source ids from {len(clips):,} clips")

    if not a.derive_only:
        path = raw_path(lay, a.backend)
        path.parent.mkdir(parents=True, exist_ok=True)
        done = load_raw(path)
        todo = [s for s in ids if s not in done]
        if a.limit:
            todo = todo[: a.limit]
        print(f"archive {path}: {len(done):,} already fetched, {len(todo):,} to fetch via {a.backend}")
        t0 = time.time(); n_found = 0
        with path.open("a") as f:
            def write(sid: str, item: dict | None):
                f.write(json.dumps({"source_id": sid, "backend": a.backend, "fetched_at": _now(),
                                    "found": item is not None, "item": item}, ensure_ascii=False) + "\n")
            try:
                if a.backend == "api":
                    if not a.api_key:
                        sys.exit("need --api-key or YOUTUBE_API_KEY (see module docstring)")
                    for i in range(0, len(todo), API_BATCH):
                        batch = todo[i: i + API_BATCH]
                        items = api_fetch_batch(batch, a.api_key)
                        for sid in batch:
                            write(sid, items.get(sid)); n_found += sid in items
                        f.flush()
                        if (i // API_BATCH) % 20 == 0:
                            print(f"  {i + len(batch):,}/{len(todo):,}  found {n_found:,}  {time.time() - t0:.0f}s")
                        if a.sleep:
                            time.sleep(a.sleep)
                else:
                    for i, sid in enumerate(todo, 1):
                        item = ytdlp_fetch_one(sid)
                        write(sid, item); n_found += item is not None
                        if i % 50 == 0:
                            f.flush(); print(f"  {i:,}/{len(todo):,}  found {n_found:,}  {time.time() - t0:.0f}s")
                        if a.sleep:
                            time.sleep(a.sleep)
            except QuotaExceeded as e:
                print(f"QUOTA EXCEEDED, archive is resumable (re-run tomorrow): {e}", file=sys.stderr)
            except KeyboardInterrupt:
                print("interrupted, archive is resumable", file=sys.stderr)
        print(f"fetched {len(todo):,} ids in {time.time() - t0:.0f}s, found {n_found:,}")

    df = derive(lay, ids)
    out = lay.index_dir / "sources.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df):,} rows)\n")
    print(summary(df, clips))
    return 0


if __name__ == "__main__":
    sys.exit(main())
