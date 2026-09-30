"""Safe sync of slipstream caches from S3, including onto shared (multi-user) directories.

One sync of ``<base>/<cache>`` works like this:

1. **Lock** ``<base>/.<cache>.sync.lock`` (O_EXCL create; owner/host/pid/start in the file,
   mtime refreshed every ``HEARTBEAT_S``). A lock whose process is gone (same host) or whose
   heartbeat is older than ``STALE_S`` is stale and broken automatically; a live one makes the
   sync exit non-zero without touching anything.
2. **Plan** against the S3 listing: fetch only files missing locally or with the wrong size
   (a partially purged cache re-downloads just what is gone). The remote manifest is always
   fetched and compared; a different remote manifest means the cache was rebuilt, and mixing
   versions is refused without ``--force``.
3. **Stage** into ``<base>/.<cache>.sync.partial/`` (same filesystem, so renames are atomic).
   Staged files that already match the listing are reused, so an interrupted sync resumes.
4. **Verify** staged files: size against the listing, and sha256 when the manifest carries
   ``file_sha256``.
5. **Commit**: rename staged files into the cache, ``manifest.json`` last (a fresh cache is one
   directory rename). Nothing already in the cache is deleted; only files that were missing or
   wrong are replaced. Then the cache's own integrity check runs.

On any failure the staging dir stays, with ``SYNC_INCOMPLETE.json`` saying why, and the cache
is left as it was.

Shared base dirs: when ``<base>`` is group-writable (e.g. a setgid 2775 lab dir), everything sync
creates (lock, staging, files, dirs) is made group-writable regardless of the caller's umask, so
any group member can later repair or re-sync the cache. Personal (non-group-writable) bases are
left to the umask.

Sources: S3 via ``s5cmd run`` (per-file ``--concurrency``: ``auto`` = 32 parts for files
>= ``AUTO_CONCURRENCY_MIN_MB``, else 1), or a local/NFS copy of the same cache (``source_dir``,
e.g. the lab_storage master) read with parallel ranged ``pread``/``pwrite`` into the staging dir.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

MANIFEST_FILE = "manifest.json"
HASHES_KEY = "file_sha256"  # manifest: {"<file name>": "<sha256 hex>"}
LOCK_SUFFIX = ".sync.lock"
PARTIAL_SUFFIX = ".sync.partial"
INCOMPLETE_MARKER = "SYNC_INCOMPLETE.json"
COMMANDS_FILE = ".s5cmd-commands.txt"
AUTO_CONCURRENCY = 32  # FASRC -> S3, one 3.8 GB file: 16 parts 31.9 MB/s, 32 parts 46.0 MB/s
AUTO_CONCURRENCY_MIN_MB = 256
HEARTBEAT_S = 60
STALE_S = 15 * 60
_STAGING_JUNK = {INCOMPLETE_MARKER, COMMANDS_FILE}


def lock_path(target: Path) -> Path:
    return target.parent / f".{target.name}{LOCK_SUFFIX}"


def partial_path(target: Path) -> Path:
    return target.parent / f".{target.name}{PARTIAL_SUFFIX}"


def shared_base(base: Path) -> bool:
    """Group-writable base dir: sync makes what it creates group-writable too."""
    try:
        return bool(Path(base).stat().st_mode & stat.S_IWGRP)
    except OSError:
        return False


def _group_write(p: Path) -> None:
    """chmod g+w (dirs: g+rwxs) on something we own; silently skip what we can't change."""
    try:
        st = p.stat()
        if st.st_uid != os.getuid():
            return
        extra = (stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP | stat.S_ISGID) if stat.S_ISDIR(st.st_mode) \
            else (stat.S_IRGRP | stat.S_IWGRP)
        if st.st_mode & extra != extra:
            os.chmod(p, stat.S_IMODE(st.st_mode) | extra)
    except OSError:
        pass


def not_group_writable(path: Path, limit: int = 1000) -> list[Path]:
    """Items under ``path`` (itself included) lacking group write, up to ``limit``."""
    out: list[Path] = []
    for p in [Path(path), *Path(path).rglob("*")]:
        try:
            if not p.lstat().st_mode & stat.S_IWGRP:
                out.append(p)
        except OSError:
            continue
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
# lock
# --------------------------------------------------------------------------- #


@dataclass
class LockInfo:
    user: str
    host: str
    pid: int
    started: float
    token: str
    command: str = ""
    heartbeat: float | None = None  # lock file mtime

    def describe(self) -> str:
        since = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.started))
        age = f", heartbeat {int(time.time() - self.heartbeat)} s ago" if self.heartbeat else ""
        return f"{self.user}@{self.host} pid {self.pid} since {since}{age}"


class LockHeld(RuntimeError):
    def __init__(self, path: Path, info: LockInfo | None):
        self.path, self.info = path, info
        who = info.describe() if info else "unknown owner (unreadable lock file)"
        super().__init__(f"another sync holds {path}: {who}")


def read_lock(target: Path) -> LockInfo | None:
    p = lock_path(target)
    try:
        raw = json.loads(p.read_text())
        mtime = p.stat().st_mtime
    except (OSError, ValueError):
        return None
    try:
        return LockInfo(**{k: raw[k] for k in ("user", "host", "pid", "started", "token")},
                        command=raw.get("command", ""), heartbeat=mtime)
    except (KeyError, TypeError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


def lock_stale_reason(info: LockInfo | None, lock_file: Path, now: float | None = None) -> str | None:
    """Why a lock is stale, or None if it looks live."""
    now = time.time() if now is None else now
    if info is None:
        try:
            age = now - lock_file.stat().st_mtime
        except OSError:
            return "lock vanished"
        # Unreadable content: a writer mid-create, or garbage. Only stale once old.
        return f"unreadable lock file {int(age)} s old" if age > STALE_S else None
    if info.host == socket.gethostname() and not _pid_alive(info.pid):
        return f"process {info.pid} on {info.host} is gone"
    if info.heartbeat is not None and now - info.heartbeat > STALE_S:
        return f"no heartbeat for {int(now - info.heartbeat)} s (> {STALE_S} s)"
    return None


class CacheLock:
    """Exclusive per-cache sync lock (works across hosts on a shared filesystem)."""

    def __init__(self, target: Path, *, break_lock: bool = False, log: Callable[[str], None] = print):
        self.target = Path(target)
        self.path = lock_path(self.target)
        self.break_lock = break_lock
        self.log = log
        self.info: LockInfo | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _try_create(self) -> bool:
        info = LockInfo(
            user=getpass.getuser(), host=socket.gethostname(), pid=os.getpid(),
            started=time.time(), token=uuid.uuid4().hex, command=" ".join(sys.argv),
        )
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
        except FileExistsError:
            return False
        if shared_base(self.path.parent):
            os.fchmod(fd, 0o664)  # umask 022 would leave it unbreakable by other group members
        with os.fdopen(fd, "w") as f:
            d = asdict(info)
            d.pop("heartbeat")
            json.dump(d, f)
        self.info = info
        return True

    def acquire(self) -> "CacheLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._try_create():
            held = read_lock(self.target)
            reason = "--break-lock" if self.break_lock else lock_stale_reason(held, self.path)
            if reason is None:
                raise LockHeld(self.path, held)
            # Rename first: only one of several breakers wins the rename, the rest retry O_EXCL.
            aside = self.path.with_name(f"{self.path.name}.broken-{uuid.uuid4().hex[:8]}")
            try:
                os.rename(self.path, aside)
                os.unlink(aside)
                who = held.describe() if held else "unknown owner"
                self.log(f"  ⚠ broke stale sync lock ({reason}): {who}")
            except FileNotFoundError:
                pass
            if not self._try_create():
                raise LockHeld(self.path, read_lock(self.target))
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def _heartbeat(self) -> None:
        while not self._stop.wait(HEARTBEAT_S):
            try:
                os.utime(self.path)
            except OSError:
                return

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        held = read_lock(self.target)
        if held is not None and self.info is not None and held.token == self.info.token:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "CacheLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


# --------------------------------------------------------------------------- #
# which files belong to a cache
# --------------------------------------------------------------------------- #


# visionlab-datasets' own per-store files next to slipstream's (video stores, prep/spatialvid_hq/merge.py):
# records.parquet (record_idx <-> clip_id; read by VideoDataset) and store_manifest.json (encode settings).
DATASETS_SIDECARS = frozenset({"records.parquet", "store_manifest.json"})


def cache_file_names(manifest: dict) -> set[str] | None:
    """Files a cache consists of, per its manifest; None if the manifest doesn't say (very old).

    = manifest.json + ``file_sizes`` keys + each field's storage files + ``<field>_index.npy``
    (slipstream's ``write_index`` output: not in the manifest, but auto-discovered by the loader)
    + ``DATASETS_SIDECARS``.
    """
    fields = manifest.get("fields") or {}
    sizes = manifest.get("file_sizes") or {}
    if not fields and not sizes:
        return None
    try:
        from slipstream.cache import _get_expected_files  # type: ignore
    except ImportError:  # pragma: no cover
        _get_expected_files = None
    names = {MANIFEST_FILE, *sizes, *DATASETS_SIDECARS}
    for f, meta in fields.items():
        if _get_expected_files is not None:
            names.update(_get_expected_files(f, (meta or {}).get("type", "")))
        names.add(f"{f}_index.npy")
    return names


def same_cache_version(a: dict, b: dict) -> bool:
    """Two manifests describe the same cache build: equal apart from ``file_sha256``, which a copy
    gains later (``slipstream hash``). If both carry hashes, those must agree too."""
    ha, hb = a.get(HASHES_KEY), b.get(HASHES_KEY)
    if ha and hb and ha != hb:
        return False
    strip = lambda m: {k: v for k, v in m.items() if k != HASHES_KEY}  # noqa: E731
    return strip(a) == strip(b)


def split_listing(files: dict[str, int], manifest: dict) -> tuple[dict[str, int], dict[str, int]]:
    """``(cache files, unlisted extras)`` of a ``{name: size}`` listing. Keeps everything if the
    manifest can't tell (so an old cache is never truncated)."""
    names = cache_file_names(manifest)
    if names is None:
        return dict(files), {}
    kept = {n: s for n, s in files.items() if n in names}
    return kept, {n: s for n, s in files.items() if n not in names}


def read_remote_manifest(remote: str, *, endpoint_url: str | None = None, profile: str | None = None) -> dict:
    import boto3  # type: ignore

    rest = remote[len("s3://"):] if remote.startswith("s3://") else remote
    bucket, _, prefix = rest.partition("/")
    s3 = boto3.Session(profile_name=profile).client("s3", endpoint_url=endpoint_url)
    body = s3.get_object(Bucket=bucket, Key=prefix.rstrip("/") + "/" + MANIFEST_FILE)["Body"].read()
    return json.loads(body)


def describe_extras(extras: dict[str, int], limit: int = 3) -> str:
    tops = sorted({n.split("/", 1)[0] + ("/" if "/" in n else "") for n in extras})
    shown = ", ".join(tops[:limit]) + (", ..." if len(tops) > limit else "")
    return f"{len(extras)} file(s), {sum(extras.values()) / 1e9:.2f} GB: {shown}"


# --------------------------------------------------------------------------- #
# remote listing + fetch (patched in tests)
# --------------------------------------------------------------------------- #


def list_remote_files(remote: str, *, endpoint_url: str | None = None, profile: str | None = None) -> dict[str, int]:
    """``{relative name: size}`` for every object under an S3 prefix. Raises on error."""
    import boto3  # type: ignore
    from botocore.config import Config  # type: ignore

    rest = remote[len("s3://"):] if remote.startswith("s3://") else remote
    bucket, _, prefix = rest.partition("/")
    prefix = prefix.rstrip("/") + "/"
    s3 = boto3.Session(profile_name=profile).client(
        "s3", endpoint_url=endpoint_url, config=Config(retries={"max_attempts": 5, "mode": "adaptive"})
    )
    out: dict[str, int] = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"][len(prefix):]
            if name and not name.endswith("/"):
                out[name] = int(obj["Size"])
    return out


def _tree_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


class _Progress:
    """Background progress line: bytes from ``done()``, rate, elapsed (every 2 s on a TTY, else 30 s)."""

    def __init__(self, total_bytes: int, done: Callable[[], int]):
        self.total, self.done, self.t0 = total_bytes, done, time.time()
        self.stop = threading.Event()
        self.tty = sys.stdout.isatty()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        last = 0.0
        while not self.stop.wait(2.0):
            got, dt = self.done(), time.time() - self.t0
            if self.tty or dt - last >= 30:
                last = dt
                msg = (f"    {got / 1e9:7.2f} / {self.total / 1e9:.2f} GB  "
                       f"{got / 1e6 / max(dt, 1e-9):7.1f} MB/s  {int(dt)} s")
                print(("\r" + msg) if self.tty else msg, end="" if self.tty else "\n", flush=True)

    def __enter__(self) -> "_Progress":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.thread.join(timeout=5)
        if self.tty and time.time() - self.t0 > 2:
            print(flush=True)


def _concurrency_for(size: int | None, concurrency: int | str | None) -> int | None:
    if concurrency == "auto":
        return AUTO_CONCURRENCY if (size or 0) >= AUTO_CONCURRENCY_MIN_MB * 1_000_000 else 1
    return None if concurrency is None else int(concurrency)


def fetch_files(
    remote: str,
    names: list[str],
    dest: Path,
    *,
    total_bytes: int,
    sizes: dict[str, int] | None = None,
    endpoint_url: str | None = None,
    numworkers: int = 32,
    concurrency: int | str | None = "auto",
    part_size_mb: int | None = None,
    umask: int | None = None,
    log: Callable[[str], None] = print,
) -> bool:
    """Download ``remote/<name>`` -> ``dest/<name>`` for each name with one ``s5cmd run``.

    ``concurrency``: parts per file (``s5cmd cp --concurrency``); ``"auto"`` picks per file from ``sizes``.
    ``umask``: for the s5cmd process (0o002 in shared bases, so its files come out group-writable).
    """
    if not names:
        return True
    base = remote.rstrip("/") + "/"
    lines = []
    for n in names:
        src, dst = base + n, str(dest / n)
        if any(c.isspace() for c in src + dst):
            raise ValueError(f"whitespace in path not supported by s5cmd run: {src!r} -> {dst!r}")
        flags: list[str] = []
        c = _concurrency_for((sizes or {}).get(n), concurrency)
        if c is not None:
            flags += ["--concurrency", str(c)]
        if part_size_mb is not None:
            flags += ["--part-size", str(int(part_size_mb))]
        lines.append(" ".join(["cp", *flags, src, dst]))
    cmd_file = dest / COMMANDS_FILE
    cmd_file.write_text("\n".join(lines) + "\n")
    cmd = ["s5cmd"]
    if endpoint_url:
        cmd += ["--endpoint-url", endpoint_url]
    cmd += ["--numworkers", str(numworkers), "run", str(cmd_file)]

    start_bytes = _tree_bytes(dest)
    errors: list[str] = []
    with _Progress(total_bytes, lambda: max(0, _tree_bytes(dest) - start_bytes)):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                umask=-1 if umask is None else umask)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if line.startswith("ERROR"):
                    errors.append(line.rstrip())
            rc = proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
    for e in errors[:5]:
        log(f"    {e}")
    if len(errors) > 5:
        log(f"    ... {len(errors) - 5} more s5cmd errors")
    cmd_file.unlink(missing_ok=True)
    return rc == 0 and not errors


def list_local_files(src: Path) -> dict[str, int]:
    """``{relative name: size}`` of a local cache copy (sync's own hidden files excluded)."""
    src = Path(src)
    out: dict[str, int] = {}
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if any(part.startswith(".") for part in rel.parts) or not p.is_file():
            continue
        out[rel.as_posix()] = p.stat().st_size
    return out


COPY_SUFFIX = ".copying"


def copy_local_files(
    src: Path,
    names: list[str],
    dest: Path,
    *,
    total_bytes: int,
    sizes: dict[str, int],
    readers: int = 16,
    chunk_mb: int = 64,
    log: Callable[[str], None] = print,
) -> bool:
    """Copy ``src/<name>`` -> ``dest/<name>`` with ``readers`` parallel ranged pread/pwrite.

    Each file is preallocated as ``<name>.copying`` and renamed when all its chunks are written,
    so a half-copied file never has its final name (``plan`` would otherwise reuse it by size).
    """
    if not names:
        return True
    src, dest = Path(src), Path(dest)
    chunk = chunk_mb << 20
    jobs: list[tuple[str, int, int]] = []
    left: dict[str, int] = {}
    for n in names:
        size = sizes[n]
        tmp = dest / (n + COPY_SUFFIX)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            f.truncate(size)
        offs = list(range(0, size, chunk)) or [0]
        left[n] = len(offs)
        jobs += [(n, o, min(chunk, size - o)) for o in offs]
    # Interleave files so large files spread across readers from the start.
    jobs.sort(key=lambda j: (j[1], j[0]))
    copied = [0]
    mu = threading.Lock()
    errors: list[str] = []

    def one(job: tuple[str, int, int]) -> None:
        n, off, length = job
        if errors:
            return
        try:
            fin = os.open(src / n, os.O_RDONLY)
            fout = os.open(dest / (n + COPY_SUFFIX), os.O_WRONLY)
            try:
                pos, end = off, off + length
                while pos < end:
                    buf = os.pread(fin, min(8 << 20, end - pos), pos)
                    if not buf:
                        raise OSError(f"short read at {pos} (source shrank?)")
                    os.pwrite(fout, buf, pos)
                    pos += len(buf)
                    with mu:
                        copied[0] += len(buf)
            finally:
                os.close(fin)
                os.close(fout)
            with mu:
                left[n] -= 1
                finished = left[n] == 0
            if finished:
                os.replace(dest / (n + COPY_SUFFIX), dest / n)
        except OSError as exc:
            with mu:
                errors.append(f"{n} @ {off}: {exc}")

    with _Progress(total_bytes, lambda: copied[0]):
        with ThreadPoolExecutor(max_workers=max(1, readers)) as pool:
            list(pool.map(one, jobs))
    for e in errors[:5]:
        log(f"    copy error: {e}")
    return not errors


# --------------------------------------------------------------------------- #
# hashing / deep check
# --------------------------------------------------------------------------- #


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _hash_mismatches(root: Path, hashes: dict[str, str], names: list[str] | None = None,
                     workers: int = 8) -> dict[str, str]:
    """``{name: problem}`` for files under ``root`` that are missing or whose sha256 differs."""
    names = sorted(hashes) if names is None else [n for n in names if n in hashes]

    def one(n: str) -> str | None:
        p = root / n
        if not p.exists():
            return f"missing: {n}"
        return None if sha256_file(p) == hashes[n] else f"sha256 mismatch: {n}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {n: m for n, m in zip(names, pool.map(one, names)) if m}


def manifest_hashes(manifest_path: Path) -> dict[str, str]:
    try:
        m = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return {}
    h = m.get(HASHES_KEY)
    return h if isinstance(h, dict) else {}


def deep_check(cache_dir: Path) -> tuple[str, list[str]]:
    """``("ok" | "mismatch" | "unavailable", problems)``: sha256 of every file the manifest hashes.

    Uses slipstream's own deep check when it has one (``check_integrity(path, deep=True)``).
    """
    import inspect

    from slipstream.cache import OptimizedCache  # type: ignore

    cache_dir = Path(cache_dir)
    hashes = manifest_hashes(cache_dir / MANIFEST_FILE)
    if not hashes:  # checked first: a size-only pass must never report "ok" for a sha256 check
        return "unavailable", [f"manifest has no {HASHES_KEY} (built by a slipstream without per-file hashes)"]
    try:
        has_deep = "deep" in inspect.signature(OptimizedCache.check_integrity).parameters
    except (TypeError, ValueError):
        has_deep = False
    if has_deep:
        ok, probs = OptimizedCache.check_integrity(cache_dir, deep=True)
        return ("ok" if ok else "mismatch"), list(probs)
    probs = list(_hash_mismatches(cache_dir, hashes).values())
    return ("mismatch" if probs else "ok"), probs


# --------------------------------------------------------------------------- #
# sync
# --------------------------------------------------------------------------- #


@dataclass
class SyncResult:
    ok: bool
    target: str
    fetched_files: int = 0
    fetched_bytes: int = 0
    reused_files: int = 0  # already correct in the cache or in the staging dir
    seconds: float = 0.0
    problems: list[str] = field(default_factory=list)
    lock_held: bool = False  # another live sync holds the lock; nothing was done

    @property
    def mb_per_s(self) -> float:
        return self.fetched_bytes / 1e6 / self.seconds if self.seconds > 0 else 0.0


def _size(p: Path) -> int | None:
    try:
        return p.stat().st_size
    except OSError:
        return None


def _shared_perms(base: Path, paths: list[Path]) -> None:
    """In a group-writable base dir, make what sync wrote group-writable so any member can repair it."""
    if shared_base(base):
        for p in paths:
            _group_write(p)


def _mark_incomplete(staging: Path, reason: str, **extra: Any) -> None:
    marker = staging / INCOMPLETE_MARKER
    try:
        tmp = marker.with_name(f".{marker.name}.{os.getpid()}")
        tmp.write_text(json.dumps(
            {"reason": reason, "time": time.time(), "user": getpass.getuser(), "host": socket.gethostname(), **extra},
            indent=2,
        ))
        _shared_perms(staging.parent, [tmp])
        os.replace(tmp, marker)  # works over another member's marker (dir write is enough)
    except OSError:
        pass


def plan(target: Path, staging: Path, remote_files: dict[str, int], *, force: bool,
         bad_in_cache: set[str] = frozenset()) -> tuple[list[str], int]:
    """Names to fetch into staging, and how many are already right (in the cache or staging)."""
    todo, reused = [], 0
    for name, size in sorted(remote_files.items()):
        if name == MANIFEST_FILE:
            continue  # always fetched fresh
        in_cache = _size(target / name) == size and name not in bad_in_cache
        in_staging = _size(staging / name) == size
        if in_staging or (in_cache and not force):
            reused += 1
        else:
            todo.append(name)
    return todo, reused


def sync_cache(
    remote: str,
    target: Path,
    *,
    remote_files: dict[str, int],
    force: bool = False,
    deep: bool = False,
    break_lock: bool = False,
    endpoint_url: str | None = None,
    numworkers: int = 32,
    concurrency: int | str | None = "auto",
    part_size_mb: int | None = None,
    source_dir: Path | None = None,
    readers: int = 16,
    chunk_mb: int = 64,
    log: Callable[[str], None] = print,
) -> SyncResult:
    """Sync one cache into ``target`` from S3 (``remote``) or, with ``source_dir``, a local copy.

    ``remote_files`` is the source's ``{name: size}`` listing (S3 or ``list_local_files``).
    """
    from slipstream.cache import OptimizedCache  # type: ignore

    target = Path(target)
    res = SyncResult(ok=False, target=str(target))
    origin = str(source_dir) if source_dir is not None else remote
    if MANIFEST_FILE not in remote_files:
        res.problems = [f"source has no {MANIFEST_FILE}: {origin}"]
        return res
    if source_dir is not None and Path(source_dir).resolve() == target.resolve():
        res.problems = [f"source and target are the same directory: {target}"]
        return res
    umask = 0o002 if shared_base(target.parent) else None

    def fetch(names: list[str], total: int, workers: int) -> bool:
        if source_dir is not None:
            return copy_local_files(Path(source_dir), names, staging, total_bytes=total, sizes=remote_files,
                                    readers=readers, chunk_mb=chunk_mb, log=log)
        return fetch_files(remote, names, staging, total_bytes=total, sizes=remote_files,
                           endpoint_url=endpoint_url, numworkers=workers, concurrency=concurrency,
                           part_size_mb=part_size_mb, umask=umask, log=log)

    try:
        lock = CacheLock(target, break_lock=break_lock, log=log).acquire()
    except LockHeld as exc:
        res.problems, res.lock_held = [str(exc)], True
        return res
    staging = partial_path(target)
    t0 = time.time()
    try:
        if target.exists() and not os.access(target, os.W_OK | os.X_OK):
            try:
                import pwd
                owner = pwd.getpwuid(target.stat().st_uid).pw_name
            except (KeyError, OSError):
                owner = "its owner"
            raise _SyncFailed(
                f"no write permission on {target} (owner {owner}); "
                f"{owner} must run: chmod -R g+w {target}"
            )
        staging.mkdir(parents=True, exist_ok=True)
        _shared_perms(target.parent, [staging])
        _mark_incomplete(staging, "sync in progress", remote=origin)
        # Staged files of the wrong size are partial downloads (never a good copy): drop them.
        for p in staging.rglob("*"):
            if p.is_file() and p.name not in _STAGING_JUNK:
                rel = p.relative_to(staging).as_posix()
                if remote_files.get(rel) != p.stat().st_size or rel == MANIFEST_FILE:
                    p.unlink()

        # 1. remote manifest first: decides whether the local copy is the same cache version.
        if not fetch([MANIFEST_FILE], remote_files[MANIFEST_FILE], 1):
            raise _SyncFailed(f"could not fetch {MANIFEST_FILE}")
        try:
            staged_manifest = json.loads((staging / MANIFEST_FILE).read_text())
        except ValueError as exc:
            raise _SyncFailed(f"source {MANIFEST_FILE} is not valid JSON: {exc}") from None
        remote_files, extras = split_listing(remote_files, staged_manifest)
        if extras:
            log(f"  ignoring files not in the manifest ({describe_extras(extras)})")
        local_m: dict | None = None
        if (target / MANIFEST_FILE).exists() and not force:
            try:
                local_m = json.loads((target / MANIFEST_FILE).read_text())
            except (OSError, ValueError):
                local_m = None  # unreadable local manifest: the source's replaces it
            if local_m is not None and not same_cache_version(local_m, staged_manifest):
                raise _SyncFailed(
                    f"source {MANIFEST_FILE} differs from the local one (cache rebuilt upstream?); "
                    "refusing to mix versions. Re-run with --force to replace the local copy."
                )
        # A local copy hashed after sync (`slipstream hash`) keeps its hashes when the source has none:
        # they verify what we fetch, and its manifest stays instead of the hashless one.
        hashes = staged_manifest.get(HASHES_KEY) or {}
        keep_local_manifest = bool(local_m and local_m.get(HASHES_KEY) and not hashes)
        if keep_local_manifest:
            hashes = local_m[HASHES_KEY]

        # 2. fetch what's missing or wrong (--deep: also what's the right size but wrong content).
        bad_in_cache: set[str] = set()
        if deep and target.exists() and not force:
            if hashes:
                log("  hashing files already in the cache (--deep) ...")
                present = [n for n, s in remote_files.items() if _size(target / n) == s]
                bad_in_cache = set(_hash_mismatches(target, hashes, present))
                for n in sorted(bad_in_cache):
                    log(f"    ✗ sha256 mismatch in cache: {n} (will re-fetch)")
            else:
                log(f"  ⚠ --deep: no {HASHES_KEY} in the source or local manifest; checking sizes only")
        todo, res.reused_files = plan(target, staging, remote_files, force=force, bad_in_cache=bad_in_cache)
        need = sum(remote_files[n] for n in todo)
        log(f"  fetching {len(todo)} of {len(remote_files) - 1} data files ({need / 1e9:.2f} GB); "
            f"{res.reused_files} already correct")
        t_fetch = time.time()
        if todo and not fetch(todo, need, numworkers):
            what = "copy" if source_dir is not None else "s5cmd"
            raise _SyncFailed(f"{what} reported errors (see above); staged files kept for the next run")
        res.seconds = time.time() - t_fetch
        res.fetched_files, res.fetched_bytes = len(todo), need

        # 3. verify staged files: size always, sha256 when the manifest has hashes.
        bad = {n: f"size mismatch: {n} (expected {remote_files[n]}, got {_size(staging / n)})"
               for n in todo if _size(staging / n) != remote_files[n]}
        if hashes and not bad:
            bad = _hash_mismatches(staging, hashes, todo)
        if bad:
            for n in bad:  # drop the bad ones so the next run re-fetches them
                (staging / n).unlink(missing_ok=True)
            raise _SyncFailed("staged files failed verification: " + "; ".join(list(bad.values())[:5]))

        # 4. commit: data files, then manifest.json last.
        if keep_local_manifest:
            (staging / MANIFEST_FILE).unlink()
        staged = [p for p in staging.rglob("*") if p.is_file() and p.name not in _STAGING_JUNK]
        _shared_perms(target.parent, staged + [d for d in staging.rglob("*") if d.is_dir()])
        for j in _STAGING_JUNK:
            (staging / j).unlink(missing_ok=True)
        if not target.exists():
            os.rename(staging, target)  # fresh cache: one atomic rename
        else:
            _shared_perms(target.parent, [target])
            for p in sorted(staged, key=lambda p: p.name == MANIFEST_FILE):
                dst = target / p.relative_to(staging)
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(p, dst)
            for d in sorted((d for d in staging.rglob("*") if d.is_dir()), reverse=True):
                d.rmdir()
            staging.rmdir()

        # 5. the cache's own check (sizes from its manifest) on the result.
        ok, probs = OptimizedCache.check_integrity(target)
        missing = [f"missing or wrong size after sync: {n}" for n, s in remote_files.items()
                   if _size(target / n) != s and not (keep_local_manifest and n == MANIFEST_FILE)]
        if not ok or missing:
            res.problems = list(probs) + missing
            return res
        res.ok = True
        return res
    except _SyncFailed as exc:
        res.problems = [str(exc)]
        _mark_incomplete(staging, str(exc), remote=origin)
        return res
    except BaseException as exc:
        if staging.exists():
            _mark_incomplete(staging, f"{type(exc).__name__}: {exc}", remote=origin)
        raise
    finally:
        if not res.seconds:
            res.seconds = time.time() - t0
        lock.release()


class _SyncFailed(Exception):
    pass


def ensure_cache(
    remote: str,
    target: Path,
    *,
    endpoint_url: str | None = None,
    poll_s: float = 10.0,
    log: Callable[[str], None] = print,
) -> SyncResult:
    """Sync ``remote`` -> ``target`` for ``load()``: same lock/stage/verify path as the CLI.

    If another process is syncing the same cache (e.g. a teammate on the shared dir), wait for it
    and use its result instead of downloading a second copy. Raises RuntimeError on failure.
    """
    from slipstream.cache import OptimizedCache  # type: ignore

    target = Path(target)
    waited = 0.0
    while True:
        files = list_remote_files(remote, endpoint_url=endpoint_url, profile=os.environ.get("AWS_PROFILE"))
        if not files:
            raise RuntimeError(f"No files at {remote}")
        res = sync_cache(remote, target, remote_files=files, endpoint_url=endpoint_url, log=log)
        if res.ok:
            return res
        if not res.lock_held:
            raise RuntimeError(f"Sync of {remote} -> {target} failed: " + "; ".join(res.problems[:5]))
        if waited == 0.0 or waited % 60 < poll_s:
            log(f"  waiting for another sync of {target.name}: {res.problems[0]}")
        while lock_path(target).exists() and lock_stale_reason(read_lock(target), lock_path(target)) is None:
            time.sleep(poll_s)
            waited += poll_s
        if (target / MANIFEST_FILE).exists() and OptimizedCache.check_integrity(target)[0]:
            return SyncResult(ok=True, target=str(target))
