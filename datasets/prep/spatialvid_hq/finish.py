"""Fleet finisher: when every group shard of an axis exists, merge it into the store, verify, and sync the store to S3.

    uv run python -m datasets.prep.spatialvid_hq.finish --out <shared work dir> --res 640x360,456x256 --fps 30,15 [--groups 74] [--s3 s3://...]

Polls every 5 min; exits when every (res, fps) axis has a verified store on S3. Safe to restart (skips merged / synced axes).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .common import Layout, axis_name, resolve_res
from .encode import patch_pending
from .merge import merge_res

S3_BASE = "s3://visionlab-datasets/slipstream-cache/spatialvid-hq"


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--res", default="640x360,456x256"); ap.add_argument("--fps", default="30,15")
    ap.add_argument("--groups", type=int, default=74); ap.add_argument("--s3", default=S3_BASE); ap.add_argument("--poll", type=int, default=300)
    a = ap.parse_args(argv)
    lay = Layout(Path("/nonexistent"), a.out)
    axes = [(resolve_res(r), int(f)) for f in a.fps.split(",") for r in a.res.split(",")]
    pending = set(axes)
    while pending:
        for res, fps in sorted(pending):
            ax = axis_name(res, fps); store = lay.store_dir(res, fps=fps); done_marker = store / ".synced"
            if done_marker.exists():
                pending.discard((res, fps)); continue
            if not (store / "store_manifest.json").exists():
                missing_groups, uncovered = patch_pending(lay, res, fps, a.groups)
                if missing_groups or uncovered:
                    if not missing_groups:
                        log(f"[{ax}] all {a.groups} shards present but {len(uncovered)} groups have uncovered failures {uncovered[:8]} -> waiting for the patch pass")
                    continue
                log(f"[{ax}] {a.groups} shards complete, every failure covered -> merging")
                sm = merge_res(lay, res, delete_shards=False, fps=fps)
                if not sm or sm.get("num_records", 0) == 0:
                    log(f"[{ax}] merge produced nothing; will retry"); continue
                log(f"[{ax}] merged {sm['num_records']:,} records, {sm['video_bytes'] / 1e9:.1f} GB, errors {sm['errors']}")
            remote = f"{a.s3}/{store.name}/"
            log(f"[{ax}] syncing to {remote}")
            r = subprocess.run(["s5cmd", "sync", "--concurrency", "8", f"{store}/", remote], capture_output=True, text=True)
            if r.returncode != 0:
                log(f"[{ax}] s5cmd sync FAILED: {r.stderr[-400:]}"); continue
            done_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n"); pending.discard((res, fps))
            log(f"[{ax}] synced")
        if pending:
            time.sleep(a.poll)
    log("all axes merged and synced"); return 0


if __name__ == "__main__":
    sys.exit(main())
