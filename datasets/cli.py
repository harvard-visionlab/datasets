"""``visionlab-datasets`` command line: cache location, dataset availability, S3 access, sync.

::

    visionlab-datasets status                  # cache dir, permissions, S3 access, per-dataset table
    visionlab-datasets status --paths          # ... plus the full local path of every present cache
    visionlab-datasets list                    # registered datasets and their S3 caches
    visionlab-datasets path in100 val          # print local cache path(s) for a dataset
    visionlab-datasets sync in100 train,val    # download caches (default fmt: jpeg)
    visionlab-datasets sync in1k val --fmt all

Also runnable as ``python -m visionlab.datasets``.

Dataset names accept short aliases (``in10``, ``in100``, ``in1k``, ``in100_s292``)
as well as the full registry names. Splits are a comma-separated list
(``train,val``) or ``all``; ``--fmt`` likewise (``jpeg,yuv420`` or ``all``).

The heavy lifting (cache-dir resolution, integrity checks, S3 listing, s5cmd
download) lives in ``slipstream.cli`` (visionlab-slipstream >= 0.4.5); this
module is a thin, registry-aware front end so lab members have one command
scoped to *our* datasets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .registry import get_config, list_datasets
from .version import __version__

MIN_SLIPSTREAM = "0.4.5"

# Short names lab members actually type -> registry names.
ALIASES: dict[str, str] = {
    "in10": "imagenet10",
    "in100": "imagenet100",
    "in1k": "imagenet1k",
    "in1000": "imagenet1k",
    "in100_s292": "imagenet100_s292",
    "in100s292": "imagenet100_s292",
    "imagenet100s292": "imagenet100_s292",
}

# The ones we advertise in help/status (the rest are accepted spellings).
PRIMARY_ALIASES = ("in10", "in100", "in1k", "in100_s292")

SPLITS = ("train", "val")
FMTS = ("jpeg", "yuv420")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _slipstream_cli():
    """Import ``slipstream.cli`` or exit with an actionable upgrade message."""
    try:
        import slipstream.cli as scli  # type: ignore
    except ImportError:
        try:
            from slipstream import __version__ as sv  # type: ignore
        except Exception:  # pragma: no cover
            sv = "unknown"
        raise SystemExit(
            f"visionlab-datasets CLI needs visionlab-slipstream >= {MIN_SLIPSTREAM} "
            f"(installed: {sv}).\n"
            "  uv:  uv lock --upgrade-package visionlab-slipstream && uv sync\n"
            "  pip: pip install -U git+https://github.com/harvard-visionlab/slipstream.git"
        )
    return scli


def _pub(scli, name: str):
    """Prefer slipstream.cli's public name, fall back to the underscore one (< 0.4.6)."""
    fn = getattr(scli, name, None) or getattr(scli, "_" + name)
    return fn


def resolve_name(name: str) -> str:
    """Map an alias or registry name (case/dash-insensitive) to a registry name."""
    key = name.strip().lower().replace("-", "_")
    known = set(list_datasets())
    if key in known:
        return key
    if key in ALIASES and ALIASES[key] in known:
        return ALIASES[key]
    aliases = ", ".join(f"{a}={d}" for a, d in ALIASES.items() if d in known)
    raise SystemExit(
        f"Unknown dataset {name!r}.\n"
        f"  registered: {', '.join(sorted(known))}\n"
        f"  aliases:    {aliases}"
    )


def parse_list(value: str | None, choices: tuple[str, ...], default: list[str]) -> list[str]:
    """``'train,val'`` -> ``['train', 'val']``; ``'all'``/None -> default; validates."""
    if value is None:
        return list(default)
    v = value.strip().lower()
    if v in ("", "all"):
        return list(choices)
    out: list[str] = []
    for item in v.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in choices:
            raise SystemExit(f"Unknown value {item!r}; choose from {', '.join(choices)} or 'all'")
        if item not in out:
            out.append(item)
    return out


def _available(name: str) -> list[tuple[str, str]]:
    return list(get_config(name).remote_cache.keys())


def _local_path(scli, cache_base: Path, remote: str) -> Path:
    cache_name = remote.rstrip("/").rsplit("/", 1)[-1]
    return cache_base / cache_name


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_status(args: argparse.Namespace) -> int:
    scli = _slipstream_cli()
    status = scli.collect_status(
        check_remote_access=not args.no_remote, endpoint_url=args.endpoint_url
    )
    status["visionlab_datasets_version"] = __version__
    status["aliases"] = {a: d for a, d in ALIASES.items() if d in set(list_datasets())}
    if args.json:
        print(json.dumps(status, indent=2, default=str))
    else:
        scli.print_status(status)
        if args.paths:
            print()
            print("Local cache paths")
            present = [e for e in status["datasets"] if e["local_status"] == "ok"]
            if not present:
                print("  (none present)")
            for e in present:
                print(f"  {e['dataset']:<18}{e['split']:<7}{e['fmt']:<8}{e['local_path']}")
        print()
        print(
            "  aliases: "
            + ", ".join(f"{a}={ALIASES[a]}" for a in PRIMARY_ALIASES if a in status["aliases"])
        )
        print("  fetch:   visionlab-datasets sync in100 train,val [--fmt jpeg|yuv420|all]")
    probs = _pub(scli, "problems")(status)
    hard = [m for m in probs if "will be created" not in m and "not installed" not in m]
    return 1 if hard else 0


def cmd_list(args: argparse.Namespace) -> int:
    rows = []
    for name in list_datasets():
        cfg = get_config(name)
        aliases = sorted(a for a, d in ALIASES.items() if d == name)
        rows.append(
            {
                "name": name,
                "aliases": aliases,
                "num_classes": cfg.num_classes,
                "remote_cache": {f"{s}/{f}": r for (s, f), r in cfg.remote_cache.items()},
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    for r in rows:
        alias = f"  [{', '.join(r['aliases'])}]" if r["aliases"] else ""
        print(f"{r['name']}  ({r['num_classes']} classes){alias}")
        if not r["remote_cache"]:
            print("  (no remote caches registered)")
        for key, remote in r["remote_cache"].items():
            s, f = key.split("/")
            print(f"  {s:<6}{f:<8}{remote}")
    return 0


def cmd_path(args: argparse.Namespace) -> int:
    scli = _slipstream_cli()
    name = resolve_name(args.dataset)
    splits = parse_list(args.splits, SPLITS, list(SPLITS))
    fmts = parse_list(args.fmt, FMTS, ["jpeg"])
    cache = scli.resolve_cache_dir(_pub(scli, "import_registry")())
    cache_base = Path(args.dest) if args.dest else Path(cache.path)
    cfg = get_config(name)
    found = 0
    for split in splits:
        for fmt in fmts:
            remote = cfg.remote_cache.get((split, fmt))
            if remote is None:
                continue
            found += 1
            p = _local_path(scli, cache_base, remote)
            if args.quiet:
                print(p)
            else:
                mark = scli.OK if (p / scli.MANIFEST_FILE).exists() else scli.BAD
                print(f"{mark} {name} {split} {fmt}  {p}")
    if not found:
        avail = ", ".join(f"{s}/{f}" for s, f in _available(name)) or "none"
        raise SystemExit(f"{name!r} has no cache for splits={splits} fmts={fmts}. Available: {avail}")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    scli = _slipstream_cli()
    name = resolve_name(args.dataset)
    splits = parse_list(args.splits, SPLITS, ["val"])
    fmts = parse_list(args.fmt, FMTS, ["jpeg"])
    avail = set(_available(name))
    combos = [(s, f) for s in splits for f in fmts if (s, f) in avail]
    skipped = [(s, f) for s in splits for f in fmts if (s, f) not in avail]
    if not combos:
        raise SystemExit(
            f"{name!r} has no cache for splits={splits} fmts={fmts}. "
            f"Available: {', '.join(f'{s}/{f}' for s, f in sorted(avail)) or 'none'}"
        )
    for s, f in skipped:
        print(f"{scli.WARN} {name}: no {s}/{f} cache registered, skipping")

    rc = 0
    for split, fmt in combos:
        ns = argparse.Namespace(
            targets=[name],
            split=split,
            fmt=fmt,
            dest=args.dest,
            force=args.force,
            dry_run=args.dry_run,
            numworkers=args.numworkers,
            endpoint_url=args.endpoint_url,
            no_color=args.no_color,
        )
        print(f"== {name} {split} {fmt} ==")
        rc = max(rc, int(scli.cmd_sync(ns) or 0))
    return rc


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="visionlab-datasets",
        description="Lab dataset caches: where they live, what is present, S3 access, and sync.",
        epilog=(
            "aliases: " + ", ".join(f"{a}={ALIASES[a]}" for a in PRIMARY_ALIASES) + "\n"
            "examples:\n"
            "  visionlab-datasets status\n"
            "  visionlab-datasets status --paths --no-remote\n"
            "  visionlab-datasets path in100 val\n"
            "  visionlab-datasets sync in100 train,val\n"
            "  visionlab-datasets sync in1k val --fmt all --dry-run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"visionlab-datasets {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser(
        "status", aliases=["config"],
        help="Cache dir + permissions, S3 access, per-dataset local/remote availability",
    )
    sp.add_argument("--paths", action="store_true", help="Also list full local path of each present cache")
    sp.add_argument("--no-remote", action="store_true", help="Skip network checks (S3 identity/listing)")
    sp.add_argument("--endpoint-url", default=None, help="S3-compatible endpoint URL")
    sp.add_argument("--json", action="store_true", help="Machine-readable output")
    sp.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("list", aliases=["datasets"], help="List registered datasets and S3 caches")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("path", help="Print local cache path(s) for a dataset")
    sp.add_argument("dataset", help="Registry name or alias (in10, in100, in1k, in100_s292)")
    sp.add_argument("splits", nargs="?", default=None, help="train | val | train,val | all (default: all)")
    sp.add_argument("--fmt", default=None, help="jpeg | yuv420 | jpeg,yuv420 | all (default: jpeg)")
    sp.add_argument("--dest", default=None, help="Override cache dir")
    sp.add_argument("-q", "--quiet", action="store_true", help="Print bare paths only")
    sp.set_defaults(func=cmd_path)

    sp = sub.add_parser(
        "sync",
        help="Download dataset cache(s) from S3 into the cache dir",
        description=(
            "examples:\n"
            "  visionlab-datasets sync in100 train,val          # jpeg\n"
            "  visionlab-datasets sync in100 val --fmt yuv420\n"
            "  visionlab-datasets sync in1k val --fmt all --dry-run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp.add_argument("dataset", help="Registry name or alias (in10, in100, in1k, in100_s292)")
    sp.add_argument("splits", nargs="?", default=None, help="train | val | train,val | all (default: val)")
    sp.add_argument("--fmt", default=None, help="jpeg | yuv420 | jpeg,yuv420 | all (default: jpeg)")
    sp.add_argument("--dest", default=None, help="Override cache dir (default: resolved cache dir)")
    sp.add_argument("--force", action="store_true", help="Re-download even if present and intact")
    sp.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    sp.add_argument("--numworkers", type=int, default=32, help="s5cmd parallel workers (default: 32)")
    sp.add_argument("--endpoint-url", default=None, help="S3-compatible endpoint URL")
    sp.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    sp.set_defaults(func=cmd_sync)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "no_color", False) or getattr(args, "func", None) in (cmd_sync, cmd_status):
        try:
            _pub(_slipstream_cli(), "configure_color")(getattr(args, "no_color", False))
        except SystemExit:
            pass  # cmd_* will re-raise with the upgrade message
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
