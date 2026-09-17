"""Progress across all encode workers: finished shards per axis, claims in flight, per-host last log line, ETA.

    uv run python -m datasets.prep.spatialvid_hq.status --out <shared work dir> [--fps 30] [--res 640x360,456x256]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from .common import RES, Layout, axis_name, resolve_res

N_GROUPS = 74


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=None); ap.add_argument("--res", default=",".join(RES))
    a = ap.parse_args(argv)
    lay = Layout(Path("/nonexistent"), a.out); res_names = [resolve_res(r) for r in a.res.split(",")]
    ax0 = lay.shard_axis_dir(res_names[0], a.fps)
    stats_paths = sorted((ax0 / "stats").glob("group_????.stats.json")) if (ax0 / "stats").exists() else sorted(lay.shards_dir.glob("group_*.stats.json"))
    patch_stats = [json.loads(p.read_text()) for p in sorted((ax0 / "stats").glob("group_????_patch.stats.json"))] if (ax0 / "stats").exists() else []
    done = [json.loads(p.read_text()) for p in stats_paths]
    n_written = sum(d.get("written", 0) for d in done); src = sum(d.get("src_bytes", 0) for d in done)
    secs = sum(d.get("seconds", 0) for d in done)
    print(f"axis {[axis_name(r, a.fps) for r in res_names]}: groups finished {len(done)} / {N_GROUPS}   clips {n_written:,}   "
          f"encode time summed over hosts {secs / 3600:.1f} h")
    for r in res_names:
        d = lay.shard_axis_dir(r, a.fps)
        shards = [p for p in d.glob("group_????") if (p / "_shard_manifest.json").exists()]
        patches = [p for p in d.glob("group_????_patch") if (p / "_shard_manifest.json").exists()]
        claims = sorted(p.stem for p in d.glob("group_*.claim"))
        finished = {p.name for p in shards + patches}
        in_flight = [c for c in claims if c not in finished]
        out_b = sum(x.get("bytes", {}).get(r, 0) for x in done)
        print(f"  {axis_name(r, a.fps)}: {len(shards)} shards + {len(patches)} patch shards, {out_b / 1e9:.1f} GB ({out_b / max(src, 1) * 100:.1f}% of source); "
              f"claimed but unfinished: {len(in_flight)} {in_flight[:12]}")
    if a.fps:
        dec = {}
        for x in done:
            for k, n in (x.get("decimation") or {}).items(): dec[k] = dec.get(k, 0) + n
        print(f"  decimation factors (clips): {dict(sorted(dec.items()))}")
    by_host = {}
    for x in done:
        h = x.get("host", "?"); by_host.setdefault(h, [0, 0.0]); by_host[h][0] += 1; by_host[h][1] += x.get("seconds", 0)
    if by_host:
        print("  groups per host: " + ", ".join(f"{h} {n} ({s / 3600:.1f} h)" for h, (n, s) in sorted(by_host.items())))
    print("\nper host (last progress line):")
    for lg in sorted(lay.logs_dir.glob("encode_*.log")) if lay.logs_dir.exists() else []:
        lines = [l for l in lg.read_text(errors="replace").splitlines() if l.startswith("[group") or l.startswith("ENCODE-EXIT") or "Error" in l or "Traceback" in l]
        last = lines[-1] if lines else "(no progress yet)"
        age = (time.time() - lg.stat().st_mtime) / 60
        print(f"  {lg.stem.replace('encode_', ''):22s} {last[:140]}   [log mtime {age:.0f} min ago; CIFS clients may lag]")
    tl = sorted((p.stat().st_mtime, p.name.replace(".stats.json", "")) for p in stats_paths)
    if tl:
        print("\nfinished groups (time = stats file mtime): " + ", ".join(f"{g}@{time.strftime('%H:%M', time.localtime(t))}" for t, g in tl[-20:]))
    if len(tl) >= 2:
        span_h = (tl[-1][0] - tl[0][0]) / 3600
        rate = (len(tl) - 1) / span_h if span_h > 0 else float("nan")
        print(f"  fleet rate {rate:.2f} groups/h over the last {span_h:.1f} h -> remaining {N_GROUPS - len(tl)} groups ≈ {(N_GROUPS - len(tl)) / rate:.1f} h")
    failed = sum(d.get("failed", 0) for d in done)
    if failed or patch_stats:
        patched = sum(d.get("written", 0) for d in patch_stats); pfail = sum(d.get("failed", 0) for d in patch_stats)
        nit = sum(len(d.get("not_in_tar", [])) for d in patch_stats)
        print(f"\nfailed clips in the main pass: {failed}; patch shards: {len(patch_stats)} groups, {patched} clips re-encoded, {pfail} failed again, {nit} not found in the tar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
