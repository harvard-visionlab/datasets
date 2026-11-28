"""Publish a verified procthor-routes store: lab_storage master + private S3 (store, split and subset tables).

    python -m datasets.prep.procthor_routes.publish --cell objects --split val --store-root <dir> --tree-root <dir> [--dry-run]

  <store-root>/<store>                 -> /n/lab_storage/alvarez_lab/Lab/datasets/slipstream/<store>/  and  S3 <dataset>/<store>/
  <tree-root>/<dataset>/{splits,subsets} -> lab_storage <dataset>/{splits,subsets}/  and  S3 <dataset>/{splits,subsets}/
Uploads keep the bucket's default object ACL (owner-only). Private data: never pass --acl public-read. The last step
checks that an anonymous GET of the manifest is refused. Refuses a store whose manifest has no passing `verify` block.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .build import CELLS, dataset, store_name

S3_BASE = "s3://visionlab-datasets/slipstream-cache"
LAB_BASE = Path("/n/lab_storage/alvarez_lab/Lab/datasets/slipstream")


def run(cmd: list[str], dry: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=CELLS)
    ap.add_argument("--split", required=True)
    ap.add_argument("--store-root", required=True, type=Path)
    ap.add_argument("--tree-root", required=True, type=Path)
    ap.add_argument("--s5cmd", default="s5cmd")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    ds = dataset(a.cell); store = a.store_root / store_name(a.cell, a.split); tree = a.tree_root / ds
    sm = json.loads((store / "store_manifest.json").read_text())
    if not (store / "manifest.json").exists() or not sm.get("verify", {}).get("ok"):
        sys.exit(f"{store}: not a finished, verified store")
    s3 = f"{S3_BASE}/{ds}"
    run(["rsync", "-a", "--chmod=Dg+rwxs,Fg+rw", f"{store}/", f"{LAB_BASE / store.name}/"], a.dry_run)
    for t in ("splits", "subsets"):
        run(["mkdir", "-p", str(LAB_BASE / ds / t)], a.dry_run)
        run(["rsync", "-a", "--chmod=Dg+rwxs,Fg+rw", f"{tree / t}/", f"{LAB_BASE / ds / t}/"], a.dry_run)
        run([a.s5cmd, "sync", f"{tree / t}/", f"{s3}/{t}/"], a.dry_run)
    run([a.s5cmd, "sync", "--concurrency", "8", f"{store}/", f"{s3}/{store.name}/"], a.dry_run)
    url = f"https://visionlab-datasets.s3.amazonaws.com/slipstream-cache/{ds}/{store.name}/manifest.json"
    if not a.dry_run:
        code = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url], capture_output=True, text=True).stdout
        print(f"anonymous GET {url} -> HTTP {code} ({'private, OK' if code in ('403', '301') else 'CHECK ACLS'})", flush=True)
        if code not in ("403", "301"):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
