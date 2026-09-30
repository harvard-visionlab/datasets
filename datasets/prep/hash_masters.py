"""Add per-file sha256 (slipstream >= 0.10 ``file_sha256``) to the master copies of the lab caches.

One coordinated pass, so every master ends up with byte-identical manifests:

1. ``hash-s3 OUT``: stream-hash every data file of each registry image cache from S3 (no local copy).
2. ``hash-local BASE OUT``: the same for a local master (e.g. lab_storage); read-only.
3. ``compare A B``: per cache, both sides saw the same manifest and got the same hashes.
4. ``publish-s3 HASHES``: back up each S3 manifest, upload the hashed one. Only if the S3 manifest is
   still the one that was hashed and the hashes were confirmed by ``compare``.
5. ``apply-local BASE``: write the published (S3) manifest into a local master, atomically. Only if
   the local manifest is the same build and the local hashes match the published ones.

Hash files (``OUT/<cache>.json``): remote/local, manifest + manifest_sha256 (the manifest hashed
against), file_sizes, file_sha256. The hashed manifest is the old one with ``file_sha256`` added,
serialised like ``slipstream.cache.write_manifest_atomic`` (``json.dump(indent=2)``), which is
what ``slipstream hash`` would write.

    python -m visionlab.datasets.prep.hash_masters hash-s3 hashes/s3
    python -m visionlab.datasets.prep.hash_masters hash-local /n/lab_storage/.../slipstream hashes/lab
    python -m visionlab.datasets.prep.hash_masters compare hashes/s3 hashes/lab
    python -m visionlab.datasets.prep.hash_masters publish-s3 hashes/s3 --confirmed-by hashes/lab
    python -m visionlab.datasets.prep.hash_masters apply-local /n/lab_storage/.../slipstream hashes/lab
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from visionlab.datasets import get_config, list_datasets

MANIFEST = "manifest.json"
BACKUP_PREFIX = "s3://visionlab-datasets/slipstream-cache-manifest-backups"


def image_caches(only: list[str] | None = None) -> list[tuple[str, str]]:
    """``(cache_name, remote)`` for every registry image cache (video stores are separate)."""
    out = []
    for n in list_datasets():
        for (_split, _fmt), remote in get_config(n).remote_cache.items():
            name = remote.rstrip("/").rsplit("/", 1)[-1]
            if not only or name in only:
                out.append((name, remote.rstrip("/") + "/"))
    return sorted(out)


def data_files(manifest: dict) -> list[str]:
    """Files ``file_sha256`` covers: every field's storage files (= ``file_sizes`` keys)."""
    from slipstream.cache import _get_expected_files  # type: ignore

    return [f for name, meta in manifest["fields"].items() for f in _get_expected_files(name, meta.get("type", ""))]


def serialise(manifest: dict) -> bytes:
    return json.dumps(manifest, indent=2).encode()


def hashed_manifest(manifest: dict, hashes: dict[str, str]) -> dict:
    m = json.loads(json.dumps(manifest))
    m["file_sha256"] = {f: hashes[f] for f in data_files(m)}
    return m


# --------------------------------------------------------------------------- #
# S3
# --------------------------------------------------------------------------- #


def _s3():
    import boto3  # type: ignore

    return boto3.client("s3")


def _split(url: str) -> tuple[str, str]:
    rest = url[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def s3_get(url: str) -> bytes:
    b, k = _split(url)
    return _s3().get_object(Bucket=b, Key=k)["Body"].read()


def s3_sha256(url: str, size: int, *, workers: int = 16, part_mb: int = 32, progress=None) -> str:
    """sha256 of an S3 object, streamed: ranged GETs in parallel, fed to one hasher in order."""
    s3, (b, k) = _s3(), _split(url)
    part = part_mb << 20
    offs = list(range(0, size, part))
    h = hashlib.sha256()

    def get(off: int) -> bytes:
        for attempt in range(5):
            try:
                r = s3.get_object(Bucket=b, Key=k, Range=f"bytes={off}-{min(off + part, size) - 1}")
                data = r["Body"].read()
                if len(data) == min(part, size - off):
                    return data
            except Exception:
                if attempt == 4:
                    raise
            time.sleep(2 ** attempt)
        raise RuntimeError(f"short read {url} @ {off}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        window: deque = deque()
        it = iter(offs)
        for off in it:
            window.append(pool.submit(get, off))
            if len(window) >= 2 * workers:
                break
        while window:
            data = window.popleft().result()
            h.update(data)
            if progress:
                progress(len(data))
            nxt = next(it, None)
            if nxt is not None:
                window.append(pool.submit(get, nxt))
    return h.hexdigest()


def cmd_hash_s3(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for name, remote in image_caches(args.only):
        dest = out / f"{name}.json"
        raw = s3_get(remote + MANIFEST)
        man = json.loads(raw)
        msha = hashlib.sha256(raw).hexdigest()
        if dest.exists() and json.loads(dest.read_text()).get("manifest_sha256") == msha:
            print(f"= {name}: already hashed")
            continue
        if man.get("file_sha256"):
            print(f"= {name}: S3 manifest already has file_sha256, skipping")
            continue
        files = data_files(man)
        sizes = man.get("file_sizes") or {}
        if sorted(files) != sorted(sizes):
            print(f"✗ {name}: data files {sorted(files)} != file_sizes keys {sorted(sizes)}; skipping")
            rc = 1
            continue
        total, done, t0 = sum(sizes.values()), [0], time.time()

        def progress(n: int) -> None:
            done[0] += n
            if sys.stdout.isatty():
                dt = time.time() - t0
                print(f"\r  {name}: {done[0] / 1e9:7.2f} / {total / 1e9:.2f} GB  {done[0] / 1e6 / max(dt, 1e-9):6.1f} MB/s",
                      end="", flush=True)

        hashes = {f: s3_sha256(remote + f, sizes[f], workers=args.workers, progress=progress) for f in files}
        dt = time.time() - t0
        if sys.stdout.isatty():
            print()
        dest.write_text(json.dumps({"cache": name, "remote": remote, "manifest_sha256": msha, "manifest": man,
                                    "file_sizes": sizes, "file_sha256": hashes, "seconds": round(dt, 1)}, indent=1))
        print(f"✓ {name}: {len(hashes)} files, {total / 1e9:.2f} GB in {dt:.0f} s ({total / 1e6 / max(dt, 1e-9):.1f} MB/s)")
    return rc


# --------------------------------------------------------------------------- #
# local master
# --------------------------------------------------------------------------- #


def cmd_hash_local(args) -> int:
    from slipstream.cache import OptimizedCache, compute_file_hashes  # type: ignore

    base, out = Path(args.base), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for name, remote in image_caches(args.only):
        d = base / name
        if not (d / MANIFEST).exists():
            print(f"- {name}: not in {base}")
            continue
        raw = (d / MANIFEST).read_bytes()
        man, msha = json.loads(raw), hashlib.sha256(raw).hexdigest()
        dest = out / f"{name}.json"
        if dest.exists() and json.loads(dest.read_text()).get("manifest_sha256") == msha:
            print(f"= {name}: already hashed")
            continue
        ok, probs = OptimizedCache.check_integrity(d)
        if not ok:
            print(f"✗ {name}: not intact, not hashing: {probs[:3]}")
            rc = 1
            continue
        files, t0 = data_files(man), time.time()
        hashes = compute_file_hashes(d, files, workers=args.workers)
        dt = time.time() - t0
        total = sum((d / f).stat().st_size for f in files)
        dest.write_text(json.dumps({"cache": name, "local": str(d), "manifest_sha256": msha, "manifest": man,
                                    "file_sizes": {f: (d / f).stat().st_size for f in files},
                                    "file_sha256": hashes, "seconds": round(dt, 1)}, indent=1))
        print(f"✓ {name}: {len(hashes)} files, {total / 1e9:.2f} GB in {dt:.0f} s ({total / 1e6 / max(dt, 1e-9):.1f} MB/s)")
    return rc


# --------------------------------------------------------------------------- #
# compare / publish / apply
# --------------------------------------------------------------------------- #


def _load(d: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(d).glob("*.json"))}


def compare(a: dict[str, dict], b: dict[str, dict]) -> dict[str, list[str]]:
    """Per cache in both: list of disagreements (empty = confirmed)."""
    out = {}
    for name in sorted(set(a) & set(b)):
        from visionlab.datasets.sync import same_cache_version

        x, y = a[name], b[name]
        probs = []
        if not same_cache_version(x["manifest"], y["manifest"]):
            probs.append("manifests describe different builds")
        if x["file_sizes"] != y["file_sizes"]:
            probs.append("file sizes differ")
        probs += [f"sha256 differs: {f}" for f in sorted(set(x["file_sha256"]) | set(y["file_sha256"]))
                  if x["file_sha256"].get(f) != y["file_sha256"].get(f)]
        out[name] = probs
    return out


def cmd_compare(args) -> int:
    a, b = _load(args.a), _load(args.b)
    res = compare(a, b)
    for name, probs in res.items():
        print(f"{'✓' if not probs else '✗'} {name}" + (": " + "; ".join(probs[:5]) if probs else ""))
    for name in sorted(set(a) ^ set(b)):
        print(f"- {name}: only in {args.a if name in a else args.b}")
    return 1 if any(res.values()) else 0


def cmd_publish_s3(args) -> int:
    hashes = _load(args.hashes)
    confirmed = compare(hashes, _load(args.confirmed_by)) if args.confirmed_by else {}
    stamp = time.strftime("%Y-%m-%d")
    s3, rc = _s3(), 0
    for name, h in hashes.items():
        if args.only and name not in args.only:
            continue
        if args.confirmed_by and confirmed.get(name) != []:
            print(f"- {name}: not confirmed by {args.confirmed_by} ({confirmed.get(name, 'missing there')}); skipping")
            continue
        remote = h["remote"]
        raw = s3_get(remote + MANIFEST)
        if hashlib.sha256(raw).hexdigest() != h["manifest_sha256"]:
            print(f"✗ {name}: S3 manifest changed since it was hashed; skipping")
            rc = 1
            continue
        new = serialise(hashed_manifest(json.loads(raw), h["file_sha256"]))
        backup = f"{BACKUP_PREFIX}/{stamp}/{name}/{MANIFEST}"
        if args.dry_run:
            print(f"[dry-run] {name}: back up to {backup}, upload {len(new)} B manifest with {len(h['file_sha256'])} hashes")
            continue
        bb, bk = _split(backup)
        s3.put_object(Bucket=bb, Key=bk, Body=raw)
        b, k = _split(remote + MANIFEST)
        s3.put_object(Bucket=b, Key=k, Body=new)
        if s3_get(remote + MANIFEST) != new:
            print(f"✗ {name}: read-back differs after upload")
            rc = 1
            continue
        print(f"✓ {name}: hashed manifest published (backup {backup})")
    return rc


def cmd_apply_local(args) -> int:
    from slipstream.cache import write_manifest_atomic  # type: ignore

    from visionlab.datasets.sync import same_cache_version

    base, local_hashes, rc = Path(args.base), _load(args.hashes), 0
    for name, remote in image_caches(args.only):
        d = base / name
        if not (d / MANIFEST).exists():
            continue
        published = json.loads(s3_get(remote + MANIFEST))
        current = json.loads((d / MANIFEST).read_bytes())
        if not published.get("file_sha256"):
            print(f"- {name}: S3 manifest not hashed yet")
            continue
        if current.get("file_sha256") == published["file_sha256"] and current == published:
            print(f"= {name}: already applied")
            continue
        if not same_cache_version(current, published):
            print(f"✗ {name}: local manifest is a different build from S3's; not applying")
            rc = 1
            continue
        mine = local_hashes.get(name, {}).get("file_sha256")
        if mine != published["file_sha256"]:
            print(f"✗ {name}: local hashes {'missing' if mine is None else 'differ from S3'}; run hash-local first")
            rc = 1
            continue
        if args.dry_run:
            print(f"[dry-run] {name}: would write the published manifest")
            continue
        write_manifest_atomic(d, published)
        if (d / MANIFEST).read_bytes() != serialise(published):
            print(f"⚠ {name}: written, but bytes differ from S3's (json formatting?)")
        print(f"✓ {name}: hashed manifest applied")
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hash_masters", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("hash-s3")
    s.add_argument("out")
    s.add_argument("--workers", type=int, default=16, help="parallel ranged GETs per object")
    s = sub.add_parser("hash-local")
    s.add_argument("base")
    s.add_argument("out")
    s.add_argument("--workers", type=int, default=None, help="parallel file hashers")
    s = sub.add_parser("compare")
    s.add_argument("a")
    s.add_argument("b")
    s = sub.add_parser("publish-s3")
    s.add_argument("hashes")
    s.add_argument("--confirmed-by", default=None, help="second hash dir that must agree (e.g. lab_storage's)")
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("apply-local")
    s.add_argument("base")
    s.add_argument("hashes", help="this master's hash-local output (must match the published hashes)")
    s.add_argument("--dry-run", action="store_true")
    for s in sub.choices.values():
        s.add_argument("--only", nargs="*", default=None, help="cache names to limit to")
    args = p.parse_args(argv)
    return {"hash-s3": cmd_hash_s3, "hash-local": cmd_hash_local, "compare": cmd_compare,
            "publish-s3": cmd_publish_s3, "apply-local": cmd_apply_local}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
