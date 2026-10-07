"""Multi-epoch throughput of the ProcTHOR r160 H5 loader (``procthor_r160_loader.py``, the training loader).

Runs the loader file unchanged (path passed via ``--loader``; it is not vendored here) over a run's recorded window
list, moves every batch to the GPU and touches it (``float().add_(0.01)``) so the samples materialise, and reports
windows/s and frames/s per epoch. ``--drop-cache`` evicts the selected H5 files from the page cache before epoch 1
(``posix_fadvise(DONTNEED)``), so epoch 1 is cold and epochs 2+ are warm when the files fit in RAM (the val split,
~58 GB, does; the train split, ~520 GB, does not, so cold is the training-realistic number).

    uv run --no-sync --group procthor python benchmarks/bench_procthor_h5.py --loader $SANDBOX_DIR/procthor/procthor_r160_loader.py \
        --split val --epochs 3 --workers 15 --batch-size 256 --drop-cache
    ... --profile 200          # single process, per-window breakdown: open / raw chunk read (I/O) / read+decompress

One line per epoch plus a summary line; no progress bars.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import numpy as np


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("procthor_r160_loader", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                           # DataLoader workers pickle the dataset by module name
    spec.loader.exec_module(mod)
    return mod


def drop_page_cache(paths) -> None:
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def ms(x) -> float:
    return 1e3 * float(np.median(x))


def profile(m, windows, n: int, seed: int) -> None:
    """Per-window cost in one process: h5 open, raw (compressed) chunk reads, and the loader's sliced read."""
    import h5py
    rng = np.random.default_rng(seed)
    pick = [windows[i] for i in rng.choice(len(windows), n, replace=False)]
    t_open, t_raw, t_full, raw_bytes, n_chunks = [], [], [], 0, 0
    for path, start in pick:
        end = start + m.LEN_WINDOW
        t = time.perf_counter()
        f = h5py.File(path, "r", libver="latest", rdcc_nbytes=64 * 1024 ** 2)
        d = f[m.GROUP]["visual_scenes"]
        t_open.append(time.perf_counter() - t)
        c = d.chunks[0]
        t = time.perf_counter()
        for c0 in range(start // c * c, end, c):           # I/O only: the compressed chunks the window touches
            _, buf = d.id.read_direct_chunk((c0, 0, 0, 0))
            raw_bytes += len(buf)
            n_chunks += 1
        t_raw.append(time.perf_counter() - t)
        f.close()
    drop_page_cache(sorted({p for p, _ in pick}))          # the full read below should not hit the raw read's cache
    ds = m.WindowDataset(pick)
    for i in range(len(pick)):
        t = time.perf_counter()
        ds[i]
        t_full.append(time.perf_counter() - t)
    print(f"PROFILE n={n}: median per window open {ms(t_open):.1f} ms | raw chunk read {ms(t_raw):.1f} ms "
          f"({n_chunks / n:.2f} chunks, {raw_bytes / n / 1e6:.2f} MB) | loader __getitem__ (cold) {ms(t_full):.1f} ms "
          f"=> decompress+copy ~{ms(t_full) - ms(t_raw) - ms(t_open):.1f} ms", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loader", required=True, help="path to procthor_r160_loader.py")
    ap.add_argument("--run", default="pt_objects_brownian__bridge_r160")
    ap.add_argument("--split", default="val", help="window list of the run: train or val")
    ap.add_argument("--max-windows", type=int, default=0, help="0 = all windows of the split")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--drop-cache", action="store_true")
    ap.add_argument("--profile", type=int, default=0, help="only run the single-process per-window profile on N windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import torch
    m = load_module(a.loader)
    windows = m.training_windows(a.run, a.split)
    if a.max_windows:
        rng = np.random.default_rng(a.seed)
        windows = [windows[i] for i in np.sort(rng.choice(len(windows), a.max_windows, replace=False))]
    files = sorted({p for p, _ in windows})
    gb = sum(os.path.getsize(p) for p in files) / 1e9
    print(f"{a.run} {a.split}: {len(windows):,} windows from {len(files):,} files ({gb:.1f} GB on disk)", flush=True)

    if a.profile:
        profile(m, windows, a.profile, a.seed)
        return

    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    loader = m.make_loader(windows, batch_size=a.batch_size, shuffle=True, num_workers=a.workers)
    if a.drop_cache:
        t = time.perf_counter()
        drop_page_cache(files)
        print(f"dropped page cache for {len(files):,} files in {time.perf_counter() - t:.1f} s", flush=True)
    rows = []
    for ep in range(a.epochs):
        n_win, first, acc = 0, None, torch.zeros((), device=dev)
        t = time.perf_counter()
        for frames, positions, headings in loader:
            x = frames.to(dev).float().add_(0.01)
            acc += x[:, :, 0, 0, 0].sum() + positions.to(dev).sum() + headings.to(dev).sum()
            n_win += frames.shape[0]
            if first is None:
                torch.cuda.synchronize()
                first = time.perf_counter() - t
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        rows.append(n_win / dt)
        print(f"epoch {ep + 1}: {n_win / dt:,.1f} windows/s  {n_win * m.LEN_WINDOW / dt:,.0f} frames/s  "
              f"{n_win * m.LEN_WINDOW * 120 * 160 * 3 / dt / 1e9:.2f} GB/s decoded  {dt:.1f} s  first batch {first:.1f} s  "
              f"({n_win} windows, frames {tuple(frames.shape)}, check {acc.item():.3g})", flush=True)

    cold, warm = rows[0], (float(np.mean(rows[1:])) if len(rows) > 1 else float("nan"))
    print(f"SUMMARY h5 {a.tag} run={a.run} split={a.split} B={a.batch_size} workers={a.workers} windows={len(windows)} "
          f"files={len(files)} {gb:.1f} GB: epoch1 {cold:,.1f} windows/s ({cold * m.LEN_WINDOW:,.0f} frames/s), "
          f"epochs2+ {warm:,.1f} windows/s ({warm * m.LEN_WINDOW:,.0f} frames/s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
