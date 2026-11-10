"""Pre-flight check of local caches: complete? how old? how long until a scratch purge?

Meant for launch scripts (``visionlab-datasets check ...`` on a login node; milliseconds, no network): run it before
submitting GPU jobs, and if anything is missing, refill in a CPU-only job first (``visionlab-datasets sync ...
--source <master>``, or ``load(...)``) instead of letting the GPU job do it.

A cache is checked the way ``load()`` checks it (every manifest file present at its size, plus the master copy's
file list when there is one). Its age is that of its oldest file: purges go by last write (mtime), so a cache that is
read every day still expires ``purge_days`` after it was synced.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .sync import MANIFEST_FILE, _master_listing, _size, cache_file_names


@dataclass
class CacheCheck:
    dataset: str
    split: str
    fmt: str
    cache_name: str
    path: str
    status: str                       # "ok" | "incomplete" | "empty" (dir, no manifest) | "missing"
    missing: list[str] = field(default_factory=list)
    age_days: float | None = None     # age of the oldest file (days since it was written)
    purge_days: int | None = None     # the filesystem's purge age, None = never purged
    days_left: float | None = None    # purge_days - age_days
    master: str | None = None         # intact master copy to refill from (free), if any
    remote: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def check_cache(path: Path | str, *, master: Path | str | None = None, purge_days: int | None = None,
                now: float | None = None) -> dict:
    """status / missing / age / days_left of one cache dir (see :class:`CacheCheck`)."""
    from slipstream.cache import OptimizedCache  # type: ignore

    path = Path(path)
    out: dict = {"status": "ok", "missing": [], "age_days": None, "purge_days": purge_days, "days_left": None,
                 "master": None}
    src = Path(master) if master else None
    master_files = _master_listing(src, path) if src is not None else None
    if master_files is not None:
        out["master"] = str(src)
    if not (path / MANIFEST_FILE).exists():
        out["status"] = "empty" if path.is_dir() else "missing"
        return out
    ok, probs = OptimizedCache.check_integrity(path)
    missing = [str(p) for p in probs] if not ok else []
    if master_files is not None:
        missing += [f"{n} (in the master, not here)" for n, sz in master_files.items()
                    if n != MANIFEST_FILE and _size(path / n) != sz and not any(n in m for m in missing)]
    if missing:
        out["status"], out["missing"] = "incomplete", missing
    try:
        names = cache_file_names(json.loads((path / MANIFEST_FILE).read_text())) or {MANIFEST_FILE}
    except (OSError, ValueError):
        names = {MANIFEST_FILE}
    mtimes = [q.stat().st_mtime for q in (path / n for n in names) if q.is_file()]
    if mtimes:
        out["age_days"] = ((now or time.time()) - min(mtimes)) / 86400
        if purge_days is not None:
            out["days_left"] = purge_days - out["age_days"]
    return out


def check(name: str, splits: list[str] | None = None, fmts: list[str] | None = None, *,
          cache_dir: Path | str | None = None, master_dir: Path | str | None = None,
          purge_days: int | None = None, **video_kwargs) -> list[CacheCheck]:
    """Check the caches ``load(name, split, fmt, ...)`` would use, without loading or downloading them.

    Image datasets: every (split, fmt) pair (default: every registered split, fmt jpeg). Video datasets: per split,
    the store load() picks (``video_kwargs``: res, fps, rate_hz, split_version; default split: the config's).
    ``cache_dir`` defaults to the resolved cache dir (``$SLIPSTREAM_CACHE_DIR`` or the platform's), ``master_dir`` to
    the machine's master, ``purge_days`` to the cache filesystem's (90 on netscratch, else none).
    """
    from .registry import get_config
    from .runtime_platform import get_master_dir, get_platform_cache_dir, purge_days_for

    cfg = get_config(name)
    cache_dir = Path(cache_dir or get_platform_cache_dir()).expanduser()
    master_dir = master_dir if master_dir is not None else get_master_dir()
    purge = purge_days if purge_days is not None else purge_days_for(cache_dir)
    targets: list[tuple[str, str, str]] = []           # (split, fmt, remote)
    if cfg.is_video:
        from .video_dataset import store_remotes
        for s in splits or [None]:
            meta = cfg.metadata or {}
            remote = store_remotes(cfg, split=s, fmt=(fmts or [None])[0], cache_base=cache_dir, **video_kwargs)[0]
            targets.append((s or meta.get("default_split", "train"), (fmts or [None])[0] or meta.get("default_fmt", "h265"),
                            remote))
    else:
        splits = splits or sorted({s for s, _ in cfg.remote_cache})
        fmts = fmts or ["jpeg"]
        targets = [(s, f, cfg.remote_cache[(s, f)]) for s in splits for f in fmts if (s, f) in cfg.remote_cache]
        if not targets:
            raise KeyError(f"{name!r} has no cache for splits={splits} fmts={fmts}; "
                           f"registered: {sorted(cfg.remote_cache)}")
    out, seen = [], set()
    for s, f, remote in targets:
        cache_name = remote.rstrip("/").rsplit("/", 1)[-1]
        if cache_name in seen:
            continue
        seen.add(cache_name)
        path = cache_dir / cache_name
        if cfg.is_video and not path.exists():
            # like load(): with no copy in the cache dir, a store in a mounted shared tree (lab QNAP) is used
            from .video_dataset import local_roots
            meta = cfg.metadata or {}
            shared = [r / "stores" / cache_name for r in local_roots(meta.get("tree", name))]
            path = next((q for q in shared if (q / MANIFEST_FILE).exists()), path)
        r = check_cache(path, master=Path(master_dir) / cache_name if master_dir else None, purge_days=purge)
        out.append(CacheCheck(dataset=name, split=s, fmt=f, cache_name=cache_name, path=str(path), remote=remote, **r))
    return out


def as_dicts(checks: list[CacheCheck]) -> list[dict]:
    return [asdict(c) | {"ok": c.ok} for c in checks]
