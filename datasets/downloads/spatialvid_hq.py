"""Download the raw SpatialVID-HQ release from Hugging Face to lab storage.

Downloads the group archives as-is (``videos/group_XXXX.tar.gz``, ``annotations/group_XXXX.tar.gz``),
the metadata CSV and the dataset card. Depth archives (2.33 TB) are skipped unless ``--types`` includes
``depths``. Nothing is extracted: 365k clips + 6 annotation files each would be ~2.5 M small files on a NAS;
downstream prep scripts stream the tar.gz files directly.

The dataset is gated: accept the terms at https://huggingface.co/datasets/SpatialVID/SpatialVID-HQ once,
then ``hf auth login`` (or export ``HF_TOKEN``) on the downloading machine.

Usage (on mcp, repo cloned, from the repo root)::

    uv run python -m datasets.downloads.spatialvid_hq \\
        --dest /home/jovyan/work/DataRemote/qnap/exactitude/Flash/DataSets/VideoDatasets/SpatialVID-HQ

    # metadata + annotations first (119 GB), videos later
    uv run python -m datasets.downloads.spatialvid_hq --dest ... --types metadata,annotations
    uv run python -m datasets.downloads.spatialvid_hq --dest ... --types videos

    # a few groups only, e.g. for a pilot
    uv run python -m datasets.downloads.spatialvid_hq --dest ... --groups 1-4

    # verify what is on disk against the remote listing (sizes, optional sha256)
    uv run python -m datasets.downloads.spatialvid_hq --dest ... --verify-only [--checksum]

Downloads are resumable: re-running skips complete files. ``--workers`` controls parallel files
(the Hub backend already chunks each file). Writes ``download_manifest.json`` in ``--dest`` when done.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ID = "SpatialVID/SpatialVID-HQ"
REPO_TYPE = "dataset"
TYPES = ("metadata", "annotations", "videos", "depths")
METADATA_FILES = ("README.md", ".gitattributes", "data/train/SpatialVID_HQ_metadata.csv")


def parse_groups(spec: str | None) -> set[int] | None:
    """'1-4,7,10-12' -> {1,2,3,4,7,10,11,12}; None -> all groups."""
    if not spec or spec == "all":
        return None
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def remote_listing(api, revision: str | None):
    info = api.dataset_info(REPO_ID, revision=revision, files_metadata=True)
    files = {}
    for s in info.siblings:
        files[s.rfilename] = {
            "size": s.size,
            "sha256": getattr(getattr(s, "lfs", None), "sha256", None),
        }
    return info.sha, files


def select_files(files: dict, types: set[str], groups: set[int] | None) -> list[str]:
    chosen = []
    for path in sorted(files):
        top = path.split("/")[0]
        if path in METADATA_FILES:
            if "metadata" in types:
                chosen.append(path)
            continue
        if top not in types:
            continue
        # videos/group_0001.tar.gz
        stem = Path(path).name.removesuffix(".tar.gz").removesuffix(".tar")
        try:
            gid = int(stem.split("_")[-1])
        except ValueError:
            continue
        if groups is None or gid in groups:
            chosen.append(path)
    return chosen


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000:
            return f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} PB"


def sha256_file(path: Path, bufsize: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def verify(dest: Path, files: dict, chosen: list[str], checksum: bool) -> tuple[list[str], list[str]]:
    missing, bad = [], []
    for rel in chosen:
        p = dest / rel
        if not p.exists():
            missing.append(rel)
            continue
        exp = files[rel]["size"]
        if exp is not None and p.stat().st_size != exp:
            bad.append(f"{rel}: size {p.stat().st_size} != {exp}")
            continue
        if checksum and files[rel]["sha256"]:
            t0 = time.time()
            got = sha256_file(p)
            if got != files[rel]["sha256"]:
                bad.append(f"{rel}: sha256 mismatch")
            else:
                print(f"  sha256 ok {rel} ({time.time() - t0:.0f}s)", flush=True)
    return missing, bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, type=Path, help="Target directory (created if missing)")
    ap.add_argument("--types", default="metadata,annotations,videos",
                    help=f"Comma list from {TYPES}; default excludes depths")
    ap.add_argument("--groups", default="all", help="Group ids, e.g. '1-4,10' (default all 74)")
    ap.add_argument("--workers", type=int, default=4, help="Parallel file downloads (default 4)")
    ap.add_argument("--revision", default=None, help="Pin a repo revision (commit sha); default main")
    ap.add_argument("--dry-run", action="store_true", help="List what would be downloaded and exit")
    ap.add_argument("--verify-only", action="store_true", help="Only compare on-disk files with the remote listing")
    ap.add_argument("--checksum", action="store_true", help="With --verify-only: also sha256 every file (slow)")
    args = ap.parse_args(argv)

    types = {t.strip() for t in args.types.split(",") if t.strip()}
    unknown = types - set(TYPES)
    if unknown:
        ap.error(f"unknown types {sorted(unknown)}; choose from {TYPES}")
    groups = parse_groups(args.groups)

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("huggingface_hub is required: uv sync (it comes with slipstream)", file=sys.stderr)
        return 2

    api = HfApi()
    try:
        who = api.whoami()
        print(f"Hugging Face user: {who.get('name')}")
    except Exception as e:  # noqa: BLE001
        print(f"Not logged in to Hugging Face ({type(e).__name__}). Run `hf auth login` or export HF_TOKEN.",
              file=sys.stderr)
        return 2

    revision, files = remote_listing(api, args.revision)
    chosen = select_files(files, types, groups)
    total = sum(files[f]["size"] or 0 for f in chosen)
    by_type: dict[str, list[int]] = {}
    for f in chosen:
        by_type.setdefault(f.split("/")[0] if f not in METADATA_FILES else "metadata", []).append(files[f]["size"] or 0)
    print(f"{REPO_ID} @ {revision[:12]}: {len(chosen)} files, {fmt_bytes(total)}")
    for k, v in sorted(by_type.items()):
        print(f"  {k:12s} {len(v):4d} files  {fmt_bytes(sum(v))}")
    print(f"dest: {args.dest}")

    if args.dry_run:
        for f in chosen:
            print(f"  {fmt_bytes(files[f]['size'] or 0):>10s}  {f}")
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)

    if not args.verify_only:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # xet backend is the default in hub>=1.0
        t0 = time.time()
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            revision=revision,
            local_dir=str(args.dest),
            allow_patterns=chosen,
            max_workers=args.workers,
        )
        dt = time.time() - t0
        print(f"download finished in {dt / 3600:.2f} h ({fmt_bytes(total / max(dt, 1))}/s average incl. skipped files)")

    print("verifying sizes against remote listing…", flush=True)
    missing, bad = verify(args.dest, files, chosen, checksum=args.checksum)
    manifest = {
        "repo_id": REPO_ID,
        "revision": revision,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "types": sorted(types),
        "groups": sorted(groups) if groups else "all",
        "files": {f: files[f] for f in chosen},
        "missing": missing,
        "bad": bad,
    }
    (args.dest / "download_manifest.json").write_text(json.dumps(manifest, indent=1))
    if missing or bad:
        print(f"PROBLEMS: {len(missing)} missing, {len(bad)} bad — re-run to resume; details in download_manifest.json")
        for x in (missing + bad)[:20]:
            print("  ", x)
        return 1
    print(f"OK: {len(chosen)} files verified; manifest written to {args.dest / 'download_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
