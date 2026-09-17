"""Stage 4: merge group shards into one slipstream store per resolution.

    uv run python -m datasets.prep.spatialvid_hq.merge --out <work dir> [--res 640x360,456x256] [--fps 30] [--delete-shards]

Produces stores/spatialvid-hq-h265-<res>/ with slipstream's manifest.json + field files, records.parquet
(record_idx -> clip_id, group, frame count, bytes) and store_manifest.json (encode settings, groups, totals).
Splits and experiment subsets map clip_id -> record_idx through records.parquet. Re-running rebuilds the store
from whatever shards exist (add groups later by re-running encode for them, then merge again).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from .common import DATASET, RES, Layout, axis_name, resolve_res


def merge_res(lay: Layout, res: str, delete_shards: bool, fps: int | None = None) -> dict:
    from slipstream.cache import CACHE_VERSION, MANIFEST_FILE, OptimizedCache, _get_expected_files, _merge_shards
    ax = axis_name(res, fps)
    shard_dirs = sorted(p for p in lay.shard_axis_dir(res, fps).glob("group_*") if (p / "_shard_manifest.json").exists())
    if not shard_dirs:
        print(f"[{ax}] no finished shards"); return {}
    manifests = [json.loads((d / "_shard_manifest.json").read_text()) for d in shard_dirs]
    # field types come from the shards (stores built before 2026-09-17 lack src_fps / src_num_frames)
    field_types = {f: m["type"] for f, m in manifests[0]["fields"].items()}
    ranges, start = [], 0
    for m in manifests:
        if {f: x["type"] for f, x in m["fields"].items()} != field_types:
            raise RuntimeError(f"shard field types differ: {m.get('group')}")
        ranges.append((start, start + m["num_samples"])); start += m["num_samples"]
    num = start; store = lay.store_dir(res, fps=fps); store.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); print(f"[{ax}] merging {len(shard_dirs)} shards, {num:,} records -> {store}")
    field_meta = _merge_shards(store, shard_dirs, ranges, field_types, num, verbose=False)
    file_sizes = {fn: os.path.getsize(store / fn) for f, t in field_types.items() for fn in _get_expected_files(f, t) if (store / fn).exists()}
    (store / MANIFEST_FILE).write_text(json.dumps({"version": CACHE_VERSION, "num_samples": num, "fields": field_meta, "file_sizes": file_sizes}, indent=2))
    recs = []
    for d, (s, _e) in zip(shard_dirs, ranges):
        r = pd.read_parquet(d / "records.parquet"); r["record_idx"] = r["local_idx"] + s; recs.append(r.drop(columns=["local_idx"]))
    records = pd.concat(recs, ignore_index=True)[["record_idx", "clip_id", "group_id", "num_frames", "video_bytes"]]
    records.to_parquet(store / "records.parquet", index=False)
    sm = dict(dataset=DATASET, fmt="h265", res=res, fps=fps, width=RES[res][0], height=RES[res][1], num_records=int(num),
              groups=[m.get("group") for m in manifests], encode=manifests[0].get("encode"),
              video_bytes=int(records.video_bytes.sum()), max_video_bytes=int(records.video_bytes.max()),
              errors=sum(len(m.get("errors", [])) for m in manifests), built=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (store / "store_manifest.json").write_text(json.dumps(sm, indent=1))
    ok, problems = OptimizedCache.check_integrity(store)
    print(f"[{ax}] {num:,} records, video {sm['video_bytes'] / 1e9:.2f} GB, largest clip {sm['max_video_bytes'] / 1e6:.1f} MB, "
          f"integrity {'OK' if ok else problems}, {time.time() - t0:.0f} s")
    if delete_shards and ok:
        for d in shard_dirs: shutil.rmtree(d)
    return sm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path); ap.add_argument("--res", default=",".join(RES))
    ap.add_argument("--fps", type=int, default=None, help="merge the <fps>-fps shards (default: native-fps shards)")
    ap.add_argument("--delete-shards", action="store_true")
    a = ap.parse_args(argv)
    lay = Layout(Path("/nonexistent"), a.out)
    for r in a.res.split(","):
        merge_res(lay, resolve_res(r), a.delete_shards, fps=a.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
