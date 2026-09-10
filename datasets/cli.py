"""``visionlab-datasets`` command line: cache location, dataset availability, S3 access, sync.

::

    visionlab-datasets status                  # cache dir, permissions, S3 access, per-dataset table
    visionlab-datasets status --paths          # ... plus the full local path of every present cache
    visionlab-datasets status --no-remote      # offline: skip S3 checks
    visionlab-datasets list                    # registered datasets and their S3 caches
    visionlab-datasets path imagenet100 val    # print local cache path(s) for a dataset
    visionlab-datasets sync imagenet100 train,val jpeg    # download caches
    visionlab-datasets sync imagenet1k val all            # both formats

Also runnable as ``python -m visionlab.datasets``.

Dataset names are the registry names (``imagenet10``, ``imagenet100``, ...);
short aliases (``in10``, ``in100``, ``in1k``, ``in100_s292``) are accepted too.
Splits and formats are comma-separated lists (``train,val``, ``jpeg,yuv420``) or ``all``.

Division of labour: **visionlab-datasets** owns the lab dataset registry
(names, splits, formats, remote S3 caches, per-platform cache dir) and hence
this CLI. **slipstream** is the registry-agnostic plumbing; we only use its
generic building blocks (``slipstream.cli.inspect_dir/check_s3/remote_listing/
find_other_caches``, ``slipstream.cache.OptimizedCache.check_integrity``,
``slipstream.s3_sync.download_s3_cache``). ``slipstream status`` remains as a
bare plumbing check that knows nothing about lab datasets.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import platform as _platform
import re
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .registry import get_config, list_datasets
from .runtime_platform import CACHE_DIR_ENV_VAR, detect_platform, get_platform_cache_dir
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

OK, BAD, WARN, SKIP = "✓", "✗", "⚠", "-"

_GREEN, _RED, _YELLOW, _BOLD, _RESET = "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m"
_COLOR = False


def configure_color(mode: str = "auto") -> None:
    """mode: auto (TTY and not NO_COLOR) | always | never."""
    global _COLOR
    if mode == "always":
        _COLOR = True
    elif mode == "never":
        _COLOR = False
    else:
        _COLOR = (
            sys.stdout.isatty()
            and not os.environ.get("NO_COLOR")
            and os.environ.get("TERM", "") != "dumb"
        ) or bool(os.environ.get("FORCE_COLOR"))


def colorize(line: str) -> str:
    """Paint status glyphs. Applied after padding so column alignment is unaffected."""
    if not _COLOR:
        return line
    return (
        line.replace(OK, f"{_GREEN}{OK}{_RESET}")
        .replace(BAD, f"{_RED}{BAD}{_RESET}")
        .replace(WARN, f"{_YELLOW}{WARN}{_RESET}")
    )


def bold(text: str) -> str:
    return f"{_BOLD}{text}{_RESET}" if _COLOR else text


def _print(line: str = "") -> None:
    print(colorize(line))


# --------------------------------------------------------------------------- #
# slipstream plumbing (generic, registry-agnostic)
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


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"  # pragma: no cover


def _dir_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


# --------------------------------------------------------------------------- #
# name / list parsing
# --------------------------------------------------------------------------- #


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
    """``'train,val'`` -> ``['train', 'val']``; ``'all'`` -> all choices; None -> default."""
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




# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #


@dataclass
class CacheDirInfo:
    path: str
    source: str
    env_var: str | None
    platform: str | None
    access: dict[str, Any]  # asdict(slipstream DirAccess)


def resolve_cache_dir(dest: str | None = None) -> CacheDirInfo:
    """Cache dir the way ``visionlab.datasets.load`` resolves it (env var, else platform)."""
    scli = _slipstream_cli()
    env_val = os.environ.get(CACHE_DIR_ENV_VAR)
    plat = None
    try:
        plat = detect_platform()
        plat_name = getattr(plat, "value", str(plat))
    except Exception as exc:  # pragma: no cover - defensive
        plat_name = f"unknown ({exc})"
    if dest:
        path, source = Path(dest).expanduser(), "--dest"
    elif env_val:
        path, source = Path(env_val).expanduser(), f"{CACHE_DIR_ENV_VAR} environment variable"
    else:
        path = Path(get_platform_cache_dir(plat)).expanduser()
        source = f"visionlab-datasets platform default ({plat_name})"
    return CacheDirInfo(
        path=str(path),
        source=source,
        env_var=env_val,
        platform=plat_name,
        access=asdict(scli.inspect_dir(path)),
    )


@dataclass
class DatasetEntry:
    dataset: str
    split: str
    fmt: str
    cache_name: str
    remote: str
    local_path: str
    expected_samples: int | None = None  # registry metadata num_{split}
    local_status: str = "missing"  # ok | incomplete | downloading | empty | missing | unreadable
    local_problems: list[str] = field(default_factory=list)
    local_bytes: int | None = None
    num_samples: int | None = None
    remote_status: str = "unchecked"  # ok | missing | denied | error | unchecked
    remote_files: int | None = None
    remote_bytes: int | None = None
    remote_error: str | None = None


def cache_name_for(remote: str) -> str:
    """Local dir name for a remote cache (same rule as ``registry.load``)."""
    return remote.rstrip("/").rsplit("/", 1)[-1]


def registry_entries(cache_base: Path, names: list[str] | None = None) -> list[DatasetEntry]:
    out: list[DatasetEntry] = []
    for name in names or list_datasets():
        cfg = get_config(name)
        meta = cfg.metadata or {}
        for (split, fmt), remote in cfg.remote_cache.items():
            expected = meta.get(f"num_{split}")
            cache_name = cache_name_for(remote)
            out.append(
                DatasetEntry(
                    dataset=name,
                    split=split,
                    fmt=fmt,
                    cache_name=cache_name,
                    remote=remote.rstrip("/") + "/",
                    local_path=str(cache_base / cache_name),
                    expected_samples=int(expected) if isinstance(expected, int) else None,
                )
            )
    return out


def datasets_without_caches() -> list[str]:
    return [n for n in list_datasets() if not get_config(n).remote_cache]


def check_local(entry: DatasetEntry) -> None:
    from slipstream.cache import MANIFEST_FILE, OptimizedCache  # type: ignore

    path = Path(entry.local_path)
    manifest = path / MANIFEST_FILE
    entry.local_problems = []
    if not manifest.exists():
        # A directory with no manifest is what a scratch-filesystem cull leaves behind.
        entry.local_status = "empty" if path.is_dir() else "missing"
        return
    if not os.access(manifest, os.R_OK) or not os.access(path, os.R_OK | os.X_OK):
        entry.local_status = "unreadable"
        entry.local_problems = ["no read permission"]
        return
    ok, problems = OptimizedCache.check_integrity(path)
    entry.local_status = "ok" if ok else "incomplete"
    entry.local_problems = list(problems)
    if not ok:
        # s5cmd writes "<name><digits>" and renames on completion: a live download.
        inflight = [
            q for q in path.iterdir()
            if q.is_file() and re.fullmatch(r"(.+\.(?:bin|npy|json))\d+", q.name)
        ]
        if inflight:
            entry.local_status = "downloading"
            entry.local_problems = []
            for q in inflight:
                final_name = q.name.rstrip("0123456789")
                entry.local_problems.append(
                    f"download in progress: {final_name} ({fmt_bytes(q.stat().st_size)} so far)"
                )
    try:
        with open(manifest) as f:
            entry.num_samples = int(json.load(f).get("num_samples"))
    except Exception:
        entry.num_samples = None
    if (
        entry.expected_samples is not None
        and entry.num_samples is not None
        and entry.num_samples != entry.expected_samples
    ):
        entry.local_problems.append(
            f"sample count {entry.num_samples:,} != registry num_{entry.split} {entry.expected_samples:,}"
        )
    try:
        entry.local_bytes = _dir_bytes(path)
    except OSError:
        pass
    if ok:
        for p in path.iterdir():
            if p.is_file() and not os.access(p, os.R_OK):
                entry.local_status = "unreadable"
                entry.local_problems = [f"no read permission: {p.name}"]
                break


def check_remote(entry: DatasetEntry, *, endpoint_url: str | None, profile: str | None) -> None:
    scli = _slipstream_cli()
    try:
        n, total = scli.remote_listing(entry.remote, endpoint_url=endpoint_url, profile=profile)
    except Exception as exc:
        msg = str(exc)
        entry.remote_status = "denied" if "AccessDenied" in msg or "Forbidden" in msg else "error"
        entry.remote_error = f"{type(exc).__name__}: {msg}"
        return
    entry.remote_files, entry.remote_bytes = n, total
    entry.remote_status = "ok" if n > 0 else "missing"


def _remote_base(entries: list[DatasetEntry]) -> str:
    """Common S3 base (bucket + first key component) of the registry caches."""
    if not entries:
        return "s3://visionlab-datasets/slipstream-cache/"
    rest = entries[0].remote[len("s3://") :]
    bucket, _, key = rest.partition("/")
    return f"s3://{bucket}/{key.split('/', 1)[0]}/"


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #


def collect_status(*, check_remote_access: bool = True, endpoint_url: str | None = None) -> dict:
    scli = _slipstream_cli()
    import slipstream  # type: ignore

    cache = resolve_cache_dir()
    cache_base = Path(cache.path)
    entries = registry_entries(cache_base)

    s3 = scli.check_s3(_remote_base(entries), check_remote=check_remote_access, endpoint_url=endpoint_url)

    for e in entries:
        check_local(e)
    if check_remote_access and s3.credentials_found and entries:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(
                    lambda e: check_remote(e, endpoint_url=endpoint_url, profile=s3.profile),
                    entries,
                )
            )

    others = scli.find_other_caches(cache_base, {e.cache_name for e in entries})
    return {
        "visionlab_datasets_version": __version__,
        "slipstream_version": getattr(slipstream, "__version__", "?"),
        "python": _platform.python_version(),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "cache": asdict(cache),
        "s3": asdict(s3),
        "datasets": [asdict(e) for e in entries],
        "datasets_without_caches": datasets_without_caches(),
        "other_caches": [{"name": n, "bytes": b} for n, b in others],
        "aliases": {a: ALIASES[a] for a in PRIMARY_ALIASES if ALIASES[a] in set(list_datasets())},
    }


def problems(status: dict) -> list[str]:
    """Human-readable list of things that will block loading/training."""
    out: list[str] = []
    acc = status["cache"]["access"]
    cache_path = status["cache"]["path"]
    if acc.get("error"):
        out.append(f"Cannot inspect cache dir {cache_path}: {acc['error']}")
    elif not acc["exists"]:
        if acc["can_create"]:
            out.append(f"Cache dir {cache_path} does not exist yet (will be created on first sync).")
        else:
            out.append(
                f"Cache dir {cache_path} does not exist and cannot be created (parent not writable)."
            )
    else:
        if not acc["readable"] or not acc["traversable"]:
            out.append(
                f"No read access to cache dir {cache_path} (owner {acc['owner']}, mode {acc['mode']})."
            )
        if not acc["writable"]:
            out.append(
                f"No write access to cache dir {cache_path}; existing caches usable, cannot sync new ones."
            )
    s3 = status["s3"]
    if s3["s5cmd_error"]:
        out.append(f"s5cmd not usable: {s3['s5cmd_error']}  (fix: uv tool install s5cmd)")
    if not s3["credentials_found"]:
        out.append(
            "No AWS credentials found (set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or configure ~/.aws/credentials)."
        )
    elif s3["checked"] and s3["bucket_readable"] is False:
        out.append(f"Cannot list {s3['bucket_url']}: {s3['bucket_error']}")
    unreadable = [e for e in status["datasets"] if e["local_status"] == "unreadable"]
    for e in unreadable[:3]:
        out.append(f"Cache {e['cache_name']} present but not readable: {'; '.join(e['local_problems'])}")
    return out


def hard_problems(status: dict) -> list[str]:
    return [m for m in problems(status) if "will be created" not in m]


def _mark(ok: bool | None) -> str:
    return SKIP if ok is None else (OK if ok else BAD)


def print_status(status: dict, *, paths: bool = False) -> None:
    from slipstream.cache import MANIFEST_FILE  # type: ignore

    p = _print
    c = status["cache"]
    a = c["access"]
    s3 = status["s3"]

    p(
        f"visionlab-datasets {status['visionlab_datasets_version']}"
        f"  ·  slipstream {status['slipstream_version']}"
        f"  ·  python {status['python']}"
    )
    p(f"{status['user']}@{status['host']}  ·  platform {c['platform']}")
    p()

    p("Cache directory  (slipstream caches on this machine live here)")
    p(f"  path        {bold(c['path'])}" + (f"  -> {a['resolved']}" if a.get("is_symlink") else ""))
    p(f"  source      {c['source']}")
    p(f"  {CACHE_DIR_ENV_VAR}  {c['env_var'] or '(not set)'}")
    if a.get("error"):
        p(f"  exists      {BAD}  {a['error']}")
    elif a["exists"]:
        p(f"  exists      {OK}  owner {a['owner']}:{a['group']}  mode {a['mode']}")
        p(f"  read        {_mark(a['readable'] and a['traversable'])}")
        p(f"  write       {_mark(a['writable'])}")
    else:
        p(f"  exists      {BAD}  (not created yet; can create: {_mark(a['can_create'])})")
    if a.get("free_bytes") is not None:
        p(f"  disk free   {fmt_bytes(a['free_bytes'])} of {fmt_bytes(a['total_bytes'])}")
    p()

    p("S3 access")
    if s3["s5cmd_path"]:
        p(f"  s5cmd       {OK}  {s3['s5cmd_path']} ({s3['s5cmd_version']})")
    else:
        p(f"  s5cmd       {BAD}  {s3['s5cmd_error']}")
    if s3["credentials_found"]:
        extra = f" via {s3['credentials_method']}" if s3["credentials_method"] else ""
        extra += f", profile {s3['profile']}" if s3["profile"] else ""
        extra += f", region {s3['region']}" if s3["region"] else ""
        p(f"  credentials {OK} {extra.strip()}")
    else:
        p(f"  credentials {BAD}  none found")
    if not s3["checked"]:
        p(f"  identity    {SKIP}  (remote checks skipped)")
    elif s3["identity_arn"]:
        p(f"  identity    {OK}  {s3['identity_arn']}")
    elif s3["identity_error"]:
        p(f"  identity    {BAD}  {s3['identity_error']}")
    if s3["checked"]:
        if s3["bucket_readable"]:
            p(f"  read        {OK}  {s3['bucket_url']}")
        elif s3["bucket_readable"] is False:
            p(f"  read        {BAD}  {s3['bucket_url']}: {s3['bucket_error']}")
    p()

    entries = status["datasets"]
    if entries:
        p("Lab datasets  (local = in cache dir, remote = readable on S3)")
        w_ds = max(len("dataset"), *(len(e["dataset"]) for e in entries)) + 2
        w_split = max(len("split"), *(len(e["split"]) for e in entries)) + 2
        w_fmt = max(len("fmt"), *(len(e["fmt"]) for e in entries)) + 2
        p(f"  {'dataset':<{w_ds}}{'split':<{w_split}}{'fmt':<{w_fmt}}{'local':<22}{'remote':<14}cache name")
        for e in entries:
            ls = e["local_status"]
            if ls == "ok":
                local = f"{OK} {fmt_bytes(e['local_bytes']):>9}"
            elif ls == "incomplete":
                local = f"{WARN} incomplete"
            elif ls == "downloading":
                local = f"{WARN} downloading"
            elif ls == "unreadable":
                local = f"{BAD} unreadable"
            elif ls == "empty":
                local = f"{BAD} empty dir"
            else:
                local = f"{BAD} missing"
            rs = e["remote_status"]
            if rs == "ok":
                remote = f"{OK} {fmt_bytes(e['remote_bytes']):>9}"
            elif rs == "unchecked":
                remote = SKIP
            elif rs == "missing":
                remote = f"{BAD} not found"
            elif rs == "denied":
                remote = f"{BAD} denied"
            else:
                remote = f"{BAD} error"
            p(
                f"  {e['dataset']:<{w_ds}}{e['split']:<{w_split}}{e['fmt']:<{w_fmt}}"
                f"{local:<22}{remote:<14}{e['cache_name']}"
            )
        p(f"  local root: {c['path']}")
        for e in [e for e in entries if e["local_problems"]]:
            p(f"  {WARN} {e['cache_name']}: {'; '.join(e['local_problems'][:3])}")
        errs = [e for e in entries if e["remote_status"] in ("denied", "error")]
        if errs:
            p(f"  {WARN} remote error example ({errs[0]['cache_name']}): {errs[0]['remote_error']}")
        empty = [e for e in entries if e["local_status"] == "empty"]
        if empty:
            p(
                f"  {WARN} {len(empty)} cache dir(s) exist without {MANIFEST_FILE} "
                "(files culled by the filesystem, or a download that never finished); "
                "sync re-downloads them"
            )
        missing = [e for e in entries if e["local_status"] != "ok"]
        if missing:
            ex = missing[0]
            p(f"  to fetch:   visionlab-datasets sync {ex['dataset']} {ex['split']} {ex['fmt']}")
        if status.get("datasets_without_caches"):
            p(f"  registered but no remote caches yet: {', '.join(status['datasets_without_caches'])}")
        p()

        if paths:
            p("Local cache paths")
            present = [e for e in entries if e["local_status"] == "ok"]
            if not present:
                p("  (none present)")
            for e in present:
                p(f"  {e['dataset']:<{w_ds}}{e['split']:<{w_split}}{e['fmt']:<{w_fmt}}{e['local_path']}")
            p()

    if status["other_caches"]:
        p("Other slipstream caches in cache dir")
        for o in status["other_caches"]:
            p(f"  {o['name']:<45}{fmt_bytes(o['bytes']):>10}")
        p()

    if status.get("aliases"):
        p("  aliases: " + ", ".join(f"{a}={d}" for a, d in status["aliases"].items()))
        p()

    probs = problems(status)
    if probs:
        p("Problems")
        for msg in probs:
            p(f"  {BAD} {msg}")
    else:
        p(f"{OK} Everything looks good.")


def cmd_status(args: argparse.Namespace) -> int:
    status = collect_status(check_remote_access=not args.no_remote, endpoint_url=args.endpoint_url)
    if args.json:
        print(json.dumps(status, indent=2, default=str))
    else:
        print_status(status, paths=args.paths)
    return 1 if hard_problems(status) else 0


# --------------------------------------------------------------------------- #
# list / path
# --------------------------------------------------------------------------- #


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


def _select(name: str, splits: list[str], fmts: list[str], cache_base: Path, *, default_note: str):
    entries = registry_entries(cache_base, [name])
    wanted = [(s, f) for s in splits for f in fmts]
    avail = {(e.split, e.fmt): e for e in entries}
    selected = [avail[k] for k in wanted if k in avail]
    skipped = [k for k in wanted if k not in avail]
    if not selected:
        have = ", ".join(f"{s}/{f}" for s, f in avail) or "none"
        raise SystemExit(
            f"{name!r} has no cache for splits={splits} fmts={fmts} ({default_note}). Available: {have}"
        )
    return selected, skipped


def cmd_path(args: argparse.Namespace) -> int:
    from slipstream.cache import MANIFEST_FILE  # type: ignore

    name = resolve_name(args.dataset)
    splits = parse_list(args.splits, SPLITS, list(SPLITS))
    fmts = parse_list(args.fmt, FMTS, list(FMTS))
    cache_base = Path(resolve_cache_dir(args.dest).path)
    selected, _ = _select(name, splits, fmts, cache_base, default_note="defaults: all splits, all formats")
    for e in selected:
        p = Path(e.local_path)
        if args.quiet:
            print(p)
        else:
            mark = OK if (p / MANIFEST_FILE).exists() else BAD
            _print(f"{mark} {e.dataset} {e.split} {e.fmt}  {p}")
    return 0


# --------------------------------------------------------------------------- #
# sync
# --------------------------------------------------------------------------- #


def _download_kwargs(download_fn, concurrency: int | None, part_size: int | None) -> dict:
    """s5cmd per-file tuning, passed only if the installed slipstream accepts it (>= 0.5.0).

    Default concurrency=1 (sequential writes per file): NFS volumes such as
    /n/lab_storage collapse to a few MB/s under s5cmd's default 5 concurrent
    50 MB part writes, while sequential writes run near line rate. Files are
    still downloaded in parallel across ``--numworkers``.
    """
    import inspect

    try:
        params = inspect.signature(download_fn).parameters
    except (TypeError, ValueError):  # pragma: no cover
        params = {}
    wanted = {"concurrency": concurrency, "part_size_mb": part_size}
    out = {k: v for k, v in wanted.items() if v is not None and k in params}
    dropped = [k for k, v in wanted.items() if v is not None and k not in params]
    if dropped:
        _print(
            f"  {WARN} installed slipstream ignores {', '.join(dropped)} "
            "(upgrade visionlab-slipstream >= 0.5.0 for per-file s5cmd tuning)"
        )
    return out


def cmd_sync(args: argparse.Namespace) -> int:
    _slipstream_cli()  # fail early with the upgrade hint if slipstream is too old
    from slipstream.s3_sync import download_s3_cache  # type: ignore

    name = resolve_name(args.dataset)
    splits = parse_list(args.splits, SPLITS, list(SPLITS))
    fmts = parse_list(args.fmt, FMTS, list(FMTS))
    cache = resolve_cache_dir(args.dest)
    cache_base = Path(cache.path)
    entries, skipped = _select(name, splits, fmts, cache_base, default_note="as requested")
    for s, f in skipped:
        _print(f"{WARN} {name}: no {s}/{f} cache registered, skipping")

    _print(f"Cache dir: {cache_base}  ({cache.source})")
    todo: list[DatasetEntry] = []
    for e in entries:
        check_local(e)
        if e.local_status == "ok" and not args.force:
            _print(
                f"  {OK} {e.cache_name}: already present ({fmt_bytes(e.local_bytes)}), "
                "skipping (use --force to re-download)"
            )
        else:
            todo.append(e)
    if not todo:
        return 0

    total_needed = 0
    for e in todo:
        check_remote(e, endpoint_url=args.endpoint_url, profile=os.environ.get("AWS_PROFILE"))
        state = {
            "missing": "not present locally",
            "empty": "dir exists but no manifest (culled?)",
            "downloading": "download already in progress elsewhere",
            "incomplete": "incomplete locally",
            "unreadable": "unreadable locally",
        }.get(e.local_status, "re-download")
        if e.remote_status == "ok":
            _print(
                f"  {e.cache_name}: {state}; remote {e.remote_files} files, "
                f"{fmt_bytes(e.remote_bytes)}  <- {e.remote}"
            )
            total_needed += e.remote_bytes or 0
        else:
            _print(
                f"  {BAD} {e.cache_name}: remote {e.remote_status} "
                f"({e.remote_error or 'no files at ' + e.remote})"
            )
    todo = [e for e in todo if e.remote_status == "ok"]
    if not todo:
        return 1

    acc = cache.access
    if acc.get("free_bytes") is not None:
        _print(f"  need ~{fmt_bytes(total_needed)}, free {fmt_bytes(acc['free_bytes'])} at {cache_base}")
        if acc["free_bytes"] < total_needed:
            _print(f"  {BAD} not enough free disk space")
            if not args.force:
                return 1
    if acc["exists"] and not acc["writable"]:
        _print(f"  {BAD} cache dir {cache_base} is not writable")
        return 1
    if not acc["exists"] and not acc["can_create"]:
        _print(f"  {BAD} cache dir {cache_base} cannot be created")
        return 1

    if args.dry_run:
        _print("[dry-run] nothing downloaded")
        return 0

    dl_kwargs = _download_kwargs(download_s3_cache, args.concurrency, args.part_size)
    failed = 0
    for e in todo:
        _print()
        ok = download_s3_cache(
            e.remote,
            Path(e.local_path),
            endpoint_url=args.endpoint_url,
            numworkers=args.numworkers,
            verbose=True,
            **dl_kwargs,
        )
        if ok:
            check_local(e)
            ok = e.local_status == "ok"
            if not ok:
                _print(
                    f"  {BAD} {e.cache_name}: downloaded but integrity check failed: "
                    f"{'; '.join(e.local_problems[:3])}"
                )
        if ok:
            _print(f"  {OK} {e.cache_name} -> {e.local_path}")
        else:
            failed += 1
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="visionlab-datasets",
        description="Lab dataset caches: where they live, what is present, S3 access, and sync.",
        epilog=(
            "examples:\n"
            "  visionlab-datasets status\n"
            "  visionlab-datasets status --paths --no-remote\n"
            "  visionlab-datasets list\n"
            "  visionlab-datasets path imagenet100 val\n"
            "  visionlab-datasets sync imagenet100 train,val jpeg\n"
            "  visionlab-datasets sync imagenet1k val all --dry-run\n"
            "dataset aliases also accepted: "
            + ", ".join(f"{a}={ALIASES[a]}" for a in PRIMARY_ALIASES) + "\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"visionlab-datasets {__version__}")
    p.add_argument(
        "--color", choices=("auto", "always", "never"), default="auto",
        help="Colour ✓/✗/⚠ (default: auto = only on a TTY; NO_COLOR/FORCE_COLOR honoured)",
    )
    p.add_argument("--no-color", dest="color", action="store_const", const="never", help="Same as --color never")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser(
        "status", aliases=["config"],
        help="Cache dir + permissions, S3 access, per-dataset local/remote availability",
    )
    sp.add_argument("--paths", action="store_true", help="Also list full local path of each present cache")
    sp.add_argument("--no-remote", action="store_true", help="Skip network checks (S3 identity/listing)")
    sp.add_argument("--endpoint-url", default=None, help="S3-compatible endpoint URL")
    sp.add_argument("--json", action="store_true", help="Machine-readable output")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("list", aliases=["datasets"], help="List registered datasets and S3 caches")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("path", help="Print local cache path(s) for a dataset")
    sp.add_argument("dataset", help="Registry name (see `list`) or alias")
    sp.add_argument("splits", nargs="?", default=None, help="train | val | train,val | all (default: all)")
    sp.add_argument("--fmt", default=None, help="jpeg | yuv420 | jpeg,yuv420 | all (default: all)")
    sp.add_argument("--dest", default=None, help="Override cache dir")
    sp.add_argument("-q", "--quiet", action="store_true", help="Print bare paths only")
    sp.set_defaults(func=cmd_path)

    sp = sub.add_parser(
        "sync",
        help="Download dataset cache(s) from S3 into the cache dir",
        description=(
            "usage: visionlab-datasets sync <dataset> <splits> <fmt>\n"
            "examples:\n"
            "  visionlab-datasets sync imagenet100 train,val jpeg\n"
            "  visionlab-datasets sync imagenet100 val yuv420\n"
            "  visionlab-datasets sync imagenet1k val all --dry-run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp.add_argument("dataset", help="Registry name (see `list`) or alias")
    sp.add_argument("splits", help="train | val | train,val | all")
    sp.add_argument("fmt", help="jpeg | yuv420 | jpeg,yuv420 | all")
    sp.add_argument("--dest", default=None, help="Override cache dir (default: resolved cache dir)")
    sp.add_argument("--force", action="store_true", help="Re-download even if present and intact")
    sp.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    sp.add_argument("--numworkers", type=int, default=32, help="s5cmd parallel workers (default: 32)")
    sp.add_argument(
        "--concurrency", type=int, default=1,
        help="s5cmd concurrent parts per file (default: 1 = sequential writes; "
        "s5cmd's default 5 is very slow on some NFS volumes, e.g. /n/lab_storage)",
    )
    sp.add_argument("--part-size", type=int, default=None, help="s5cmd multipart size in MiB (default: s5cmd's 50)")
    sp.add_argument("--endpoint-url", default=None, help="S3-compatible endpoint URL")
    sp.set_defaults(func=cmd_sync)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_color(args.color)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
