"""Multi-epoch DecodeVideoWindow throughput on a SpatialVID-HQ store: cold (disk) vs warm (OS page cache) epochs.

Reads the same fixed set of windows every epoch (seeded anchors from ``VideoDataset.window_sampler``), so epoch 1
measures decode + disk and epochs 2+ decode + page cache. ``--drop-cache`` evicts the store's pages first
(``posix_fadvise(DONTNEED)``, no root needed) and the residency of the selected records is printed before each epoch.

    OMP_NUM_THREADS=1 uv run --no-sync --group video python benchmarks/bench_spatialvid_window.py \
        --fps 15 --n-clips 8000 --epochs 3 --workers 64 --resize 224 --drop-cache
    ... --fps 30 / --fps native            # other stores of the same resolution
    ... --no-local-cache                   # ignore $SLIPSTREAM_CACHE_DIR: read the store off the QNAP mount (CIFS)

One line per epoch plus a summary line; no progress bars (runs under docker exec).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


def drop_page_cache(store_dir: Path) -> None:
    """Evict every file of the store from the page cache (clean pages only; works for any user)."""
    for p in store_dir.iterdir():
        if p.is_file():
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="spatialvid-hq")
    ap.add_argument("--split", default="val")
    ap.add_argument("--res", default="456x256")
    ap.add_argument("--fps", default="15", help="store: 15, 30 or native")
    ap.add_argument("--rate", type=float, default=15.0)
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--n-clips", type=int, default=8000, help="windows per epoch (one per clip, seeded pick)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-cache", action="store_true", help="posix_fadvise(DONTNEED) the store before epoch 1")
    ap.add_argument("--no-local-cache", action="store_true", help="point SLIPSTREAM_CACHE_DIR at an empty dir so the store resolves from the shared mount")
    ap.add_argument("--tag", default="", help="free text echoed in the summary line")
    a = ap.parse_args()

    if a.no_local_cache:
        os.environ["SLIPSTREAM_CACHE_DIR"] = tempfile.mkdtemp(prefix="empty-slipstream-cache-")

    import torch
    torch.set_num_threads(1)
    from slipstream.decoders import DecodeVideoWindow
    from slipstream.loader import SlipstreamLoader
    from visionlab.datasets import load
    from visionlab.datasets.video_dataset import VideoDataset

    fps = None if a.fps == "native" else int(a.fps)
    t = time.perf_counter()
    ds: VideoDataset = load(a.dataset, split=a.split, res=a.res, fps=fps, download=False)
    print(f"{ds}  store_dir={ds.store_dir}  load {time.perf_counter() - t:.1f} s", flush=True)

    window_s = a.T / a.rate
    recs, t0 = ds.window_sampler(window_s, seed=a.seed).sample(0)
    n_elig = len(recs)
    rng = np.random.default_rng(a.seed)
    pick = np.sort(rng.choice(n_elig, min(a.n_clips, n_elig), replace=False))
    recs, t0 = recs[pick], t0[pick]
    print(f"{len(recs):,} windows of {window_s:g} s picked from {n_elig:,} eligible of {len(ds):,} clips; T={a.T} rate={a.rate:g}", flush=True)

    stage = DecodeVideoWindow(T=a.T, rate_hz=a.rate, seed=a.seed, t0_key="t0", device=a.device, num_workers=a.workers,
                              num_ffmpeg_threads=1, resize=a.resize)
    ahead = max(3, -(-stage.num_workers // a.batch_size))
    from slipstream.dataset import SlipstreamDataset
    sds = SlipstreamDataset(local_dir=str(ds.store_dir))          # the loader wants a dataset wrapper, not the OptimizedCache
    loader = SlipstreamLoader(sds, batch_size=a.batch_size, shuffle=True, seed=a.seed, drop_last=True, indices=recs,
                              sample_data={"t0": t0}, batches_ahead=ahead, image_field="video", pipelines={"video": [stage]},
                              verbose=False)
    gb = None
    try:
        meta = loader._image_storage._metadata
        gb = float(meta["data_size"][recs].astype(np.int64).sum()) / 1e9
    except Exception:
        pass

    if a.drop_cache:
        drop_page_cache(ds.store_dir)
    rows = []
    for ep in range(a.epochs):
        res = loader.page_cache_residency()
        n_win = 0; shape = None
        t = time.perf_counter()
        for batch in loader:
            v = batch["video"]
            if v.device.type == "cuda":
                torch.cuda.synchronize(v.device)
            n_win += v.shape[0]; shape = tuple(v.shape)
        dt = time.perf_counter() - t
        rows.append((ep, res, n_win / dt, dt))
        print(f"epoch {ep + 1}: residency_before={res if res is None else f'{res:.2f}'}  {n_win / dt:,.1f} windows/s  "
              f"{n_win * a.T / dt:,.0f} frames/s  {dt:.1f} s  ({n_win} windows, shape {shape})", flush=True)
    loader.shutdown()

    cold, warm = rows[0][2], (np.mean([r[2] for r in rows[1:]]) if len(rows) > 1 else float("nan"))
    print(f"SUMMARY store={ds.store_dir.name} src={'mount' if a.no_local_cache else 'local'} {a.tag} T={a.T} rate={a.rate:g} "
          f"B={a.batch_size} workers={stage.num_workers} resize={a.resize} windows={len(recs)} "
          f"bytes={'?' if gb is None else f'{gb:.2f} GB'}: cold {cold:,.1f} windows/s, warm {warm:,.1f} windows/s "
          f"(x{warm / cold:.2f})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
