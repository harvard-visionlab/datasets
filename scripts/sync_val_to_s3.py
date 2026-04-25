"""Upload ImageNet-100 / ImageNet-10 val caches to S3.

By default uploads every cache in :data:`CACHES` that exists locally.
Pass cache names as positional args, or ``--match <substr>`` to target a
subset.

Usage::

    # Sync all known val caches that exist locally (default)
    uv run python scripts/sync_val_to_s3.py

    # Sync only specific caches by exact name
    uv run python scripts/sync_val_to_s3.py \\
        imagenet100-s292_l584-jpeg-val imagenet100-s292_l584-yuv420-val

    # Sync any cache whose name contains a substring (e.g. all s292)
    uv run python scripts/sync_val_to_s3.py --match s292

    # List what would be synced without uploading
    uv run python scripts/sync_val_to_s3.py --match s292 --dry-run
"""
import argparse
from pathlib import Path

from visionlab.datasets.runtime_platform import get_platform_cache_dir
from slipstream.s3_sync import upload_s3_cache

S3_BASE = "s3://visionlab-datasets/slipstream-cache"

CACHES = [
    ("imagenet100-s256_l512-jpeg-val", "imagenet100"),
    ("imagenet100-s256_l512-yuv420-val", "imagenet100"),
    ("imagenet100-s292_l584-jpeg-val", "imagenet100"),
    ("imagenet100-s292_l584-yuv420-val", "imagenet100"),
    ("imagenet10-s256_l512-jpeg-val", "imagenet10"),
    ("imagenet10-s256_l512-yuv420-val", "imagenet10"),
]


def select_caches(names: list[str], match: str | None):
    """Filter CACHES by exact names and/or a substring."""
    selected = CACHES
    if names:
        known = dict(CACHES)
        unknown = [n for n in names if n not in known]
        if unknown:
            available = "\n  ".join(n for n, _ in CACHES)
            raise SystemExit(
                f"Unknown cache name(s): {unknown}\nAvailable:\n  {available}"
            )
        selected = [(n, known[n]) for n in names]
    if match:
        selected = [(n, s) for n, s in selected if match in n]
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "names", nargs="*",
        help="Specific cache names to sync (default: all in CACHES)",
    )
    parser.add_argument(
        "--match", default=None,
        help="Only sync caches whose name contains this substring",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be synced without uploading",
    )
    args = parser.parse_args()

    selected = select_caches(args.names, args.match)
    if not selected:
        raise SystemExit("No caches match your selection.")

    cache_base = Path(get_platform_cache_dir())
    print(f"Cache base: {cache_base}")
    print(f"Selected {len(selected)} cache(s):")
    for name, _ in selected:
        print(f"  - {name}")

    for cache_name, subset in selected:
        local = cache_base / cache_name
        remote = f"{S3_BASE}/{subset}/{cache_name}"
        if not local.exists():
            print(f"\nSkipping {cache_name} (not found at {local})")
            continue
        if args.dry_run:
            print(f"\n[dry-run] Would upload {cache_name} -> {remote}")
            continue
        print(f"\nUploading {cache_name} -> {remote}")
        upload_s3_cache(local, remote)
        print(f"Done: {cache_name}")


if __name__ == "__main__":
    main()
