"""Helpers for the per-dataset demo notebooks (`notebooks/datasets/<name>.py`): what a `load()` would download and
where it would go, before anything is downloaded; and removing that local copy afterwards.

    from visionlab.datasets.demo import download_plan, remove_local_copy
    plan = download_plan("procthor-walks-objects", split="val")   # prints store, S3 size, local path, already cached?
    ds = load("procthor-walks-objects", split="val")
    remove_local_copy(ds)                                          # dry run: says what it would delete
    remove_local_copy(ds, confirm=True)

Video datasets only for now (image datasets: `visionlab-datasets status` shows the same information).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

# Shared lab copies that a demo must never delete (masters and shared training caches).
PROTECTED_PREFIXES = ("/n/lab_storage/", "/n/netscratch/", "/n/holylabs/", "/mnt/QNAP", "/mnt/qnap")


@dataclass
class DownloadPlan:
    dataset: str
    split: str
    store: str
    remote: str
    size_bytes: int | None
    local_path: Path | None       # existing local copy (no download needed), else None
    download_to: Path              # where load() would put it

    @property
    def size_gb(self) -> float | None:
        return None if self.size_bytes is None else self.size_bytes / 1e9

    def __str__(self) -> str:
        size = "unknown (not on S3, or no S3 access)" if self.size_bytes is None else f"{self.size_gb:,.2f} GB"
        where = (f"already available locally at {self.local_path}" if self.local_path
                 else f"NOT cached: load() will download {size} to {self.download_to}")
        return (f"{self.dataset} split={self.split!r}\n  store    {self.store}\n  S3       {self.remote}  ({size})\n"
                f"  local    {where}")


def download_plan(name: str, split: str | None = None, split_version: str | None = None, fmt: str | None = None,
                  res: str | None = None, rate_hz: float | None = None, fps=None, verbose: bool = True) -> DownloadPlan:
    """Which store `load(name, split=..., ...)` would use, its size on S3, and whether it is already local."""
    import pandas as pd

    from .registry import get_config
    from .runtime_platform import configure_slipstream_cache
    from .video_dataset import rank_stores, resolve_file, resolve_store, store_name, store_part

    cfg = get_config(name)
    if not cfg.is_video:
        raise NotImplementedError("download_plan covers video datasets; use `visionlab-datasets status` for image datasets")
    meta = cfg.metadata or {}
    tree = meta.get("tree", name)
    split = split or meta.get("default_split", "train")
    split_version = split_version or meta.get("default_split_version")
    fmt = fmt or meta.get("default_fmt")
    res = meta.get("res_aliases", {}).get(str(res), res) if res else meta.get("default_res")
    if rate_hz is None and fps is None:
        rate_hz = meta.get("default_rate_hz")
    cache_base = Path(configure_slipstream_cache())
    split_df = pd.read_parquet(resolve_file(f"splits/{split_version}.parquet", cfg.splits[split_version], tree, cache_base))
    key = rank_stores(cfg.stores, fmt, res, rate_hz, None if fps == "native" else fps)[0]
    remote = store_part(cfg.stores[key], split_df, split)
    local = resolve_store(remote, tree, cache_base, download=False)
    size = None
    try:
        from .sync import list_remote_files
        size = sum(list_remote_files(remote).values()) or None      # 0 = nothing listed (not uploaded / no access)
    except Exception:
        pass
    plan = DownloadPlan(name, split, store_name(remote), remote, size, local, cache_base / store_name(remote))
    if verbose:
        print(plan)
    return plan


def remove_local_copy(ds, confirm: bool = False) -> bool:
    """Delete the local store directory of a loaded dataset (`ds.store_dir`). Refuses shared lab copies (lab_storage,
    netscratch, holylabs, QNAP): those are masters / shared caches, not your download. Dry run unless confirm=True."""
    p = Path(ds.store_dir).resolve()
    if any(str(p).startswith(x) for x in PROTECTED_PREFIXES):
        print(f"Not deleting {p}: it is a shared lab copy, not a personal download.")
        return False
    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
    if not confirm:
        print(f"Would delete {p} ({size:.2f} GB). Re-run with confirm=True to delete it.")
        return False
    shutil.rmtree(p)
    print(f"Deleted {p} ({size:.2f} GB).")
    return True
