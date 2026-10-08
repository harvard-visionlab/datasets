"""Stage 4: publish a verified store: lab_storage master copy + private S3 copy (store and split tables).

    python -m datasets.prep.procthor_walks.publish --dataset procthor-walks-objects --out <work> --store <store dir> [--dry-run]

  <store>                    -> /n/lab_storage/alvarez_lab/Lab/datasets/slipstream/<store name>/   (rsync; FAS cache base)
  <store>                    -> s3://visionlab-datasets/slipstream-cache/<dataset>/<store name>/   (s5cmd sync)
  <work>/<dataset>/splits/*  -> s3://visionlab-datasets/slipstream-cache/<dataset>/splits/  and  <lab_storage>/<dataset>/splits/
Uploads keep the bucket's default object ACL (owner-only). This data is private: never pass --acl public-read. The
last step checks that an anonymous GET of the manifest is refused.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .common import DATASETS

S3_BASE = "s3://visionlab-datasets/slipstream-cache"
LAB_BASE = Path("/n/lab_storage/alvarez_lab/Lab/datasets/slipstream")


def run(cmd: list[str], dry: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--s5cmd", default="s5cmd")
    ap.add_argument("--skip-lab", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not (a.store / "manifest.json").exists() or not (a.store / "records.parquet").exists():
        sys.exit(f"{a.store} is not a finished store (manifest.json + records.parquet)")
    splits = a.out / a.dataset / "splits"
    s3 = f"{S3_BASE}/{a.dataset}"
    if not a.skip_lab:
        run(["rsync", "-a", "--chmod=Dg+rwxs,Fg+rw", f"{a.store}/", f"{LAB_BASE / a.store.name}/"], a.dry_run)
        run(["rsync", "-a", f"{splits}/", f"{LAB_BASE / a.dataset / 'splits'}/"], a.dry_run)
    run([a.s5cmd, "sync", "--concurrency", "8", f"{a.store}/", f"{s3}/{a.store.name}/"], a.dry_run)
    run([a.s5cmd, "sync", f"{splits}/", f"{s3}/splits/"], a.dry_run)
    url = f"https://visionlab-datasets.s3.amazonaws.com/slipstream-cache/{a.dataset}/{a.store.name}/manifest.json"
    if not a.dry_run:
        code = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url], capture_output=True, text=True).stdout
        print(f"anonymous GET {url} -> HTTP {code} ({'private, OK' if code in ('403', '301') else 'CHECK ACLS'})", flush=True)
        if code not in ("403", "301"):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
