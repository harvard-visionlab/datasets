"""Progress across all encode workers: finished shards per resolution, per-host last progress line, ETA.

    uv run python -m datasets.prep.spatialvid_hq.status --out <shared work dir>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from .common import RES, Layout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args(argv)
    lay = Layout(Path("/nonexistent"), a.out)
    stats = sorted(lay.shards_dir.glob("group_*.stats.json"))
    done = [json.loads(p.read_text()) for p in stats]
    n_written = sum(d.get("written", 0) for d in done); src = sum(d.get("src_bytes", 0) for d in done)
    secs = sum(d.get("seconds", 0) for d in done)
    print(f"groups finished: {len(done)} / 74   clips: {n_written:,}   wall encode time summed over hosts: {secs/3600:.1f} h")
    for r in RES:
        shards = [p for p in (lay.shards_dir / r).glob("group_*") if (p / "_shard_manifest.json").exists()]
        out_b = sum(d.get("bytes", {}).get(r, 0) for d in done)
        print(f"  {r}: {len(shards)} shards, {out_b/1e9:.1f} GB ({out_b/max(src,1)*100:.1f}% of source)")
    print("\nper host (last progress line):")
    logs = sorted((a.out / "logs").glob("encode_*.log"))
    for lg in logs:
        lines = [l for l in lg.read_text(errors="replace").splitlines() if l.startswith("[group") or l.startswith("ENCODE-EXIT")]
        last = lines[-1] if lines else "(no progress yet)"
        age = (time.time() - lg.stat().st_mtime) / 60
        host = lg.stem.replace("encode_", "")
        m = re.search(r"\[group_(\d+)\] done", last)
        print(f"  {host:9s} {last[:150]}   [log updated {age:.0f} min ago]")
    failed = sum(d.get("failed", 0) for d in done)
    if failed:
        print(f"\nfailed clips so far: {failed} (see shards/group_XXXX.stats.json 'errors')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
