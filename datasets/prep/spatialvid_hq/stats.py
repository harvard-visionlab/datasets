"""Per-store RGB normalisation stats for the SpatialVID-HQ video stores (→ `metadata["stats"][<store>]` in the config).

Decodes seeded 8 s windows at 15 Hz from the *train* population with the same `DecodeVideoWindow` stage training uses
(decoder-side resize to the model input), accumulates an exact per-channel uint8 histogram over every frame (4x4 spatial
stride), and prints a JSON block to paste into `_configs/spatialvid_hq.py`. Frame-level stats are what `normalize(mean, std)` expects
on [0, 1] floats. The three fps variants of one resolution differ only by decimation, so their stats agree to ~1e-3;
each is measured anyway so the config never has to guess.

    OMP_NUM_THREADS=1 uv run --no-sync --group video python -m datasets.prep.spatialvid_hq.stats \
        --res 456x256 --fps 15 30 native --n-clips 4000 --workers 64 --out <tree>/index/stats_456x256.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


def store_stats(res: str, fps, n_clips: int, T: int, rate: float, resize: int, workers: int, seed: int, split: str, stride: int = 4) -> dict:
    import torch
    torch.set_num_threads(1)
    from slipstream.dataset import SlipstreamDataset
    from slipstream.decoders import DecodeVideoWindow
    from slipstream.loader import SlipstreamLoader
    from visionlab.datasets import load

    ds = load("spatialvid-hq", split=split, res=res, fps=fps, download=False)
    recs, t0 = ds.window_sampler(T / rate, seed=seed).sample(0)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(recs), min(n_clips, len(recs)), replace=False))
    recs, t0 = recs[pick], t0[pick]
    stage = DecodeVideoWindow(T=T, rate_hz=rate, seed=seed, t0_key="t0", device="cpu", num_workers=workers, resize=resize)
    loader = SlipstreamLoader(SlipstreamDataset(local_dir=str(ds.store_dir)), batch_size=8, shuffle=True, seed=seed, drop_last=False,
                              indices=recs, sample_data={"t0": t0}, batches_ahead=8, image_field="video", pipelines={"video": [stage]},
                              verbose=False)
    # Per-channel histogram of uint8 values (exact, no float pass over 256 M elements per batch) on a 4x4 spatial stride:
    # ~1/16 of the pixels, still ~10^8 samples per store, and the reduction stays far cheaper than the decode.
    hist = torch.zeros(3, 256, dtype=torch.int64); t = time.perf_counter(); n_batches = 0
    for batch in loader:
        v = batch["video"][:, :, :, ::stride, ::stride]                  # [B, T, 3, H/4, W/4] uint8
        for c in range(3):
            hist[c] += torch.bincount(v[:, :, c].reshape(-1), minlength=256)
        n_batches += 1
        if n_batches % 100 == 0:
            print(f"  {ds.store_dir.name}: {n_batches} batches, {n_batches * 8 / (time.perf_counter() - t):.0f} windows/s", flush=True)
    loader.shutdown()
    levels = torch.arange(256, dtype=torch.float64) / 255
    h = hist.to(torch.float64); n = int(h[0].sum())
    mean = (h * levels).sum(1) / n; std = torch.sqrt((h * levels ** 2).sum(1) / n - mean ** 2)
    mean, std = mean.numpy(), std.numpy()
    out = dict(store=ds.store_dir.name, split=split, subset=ds.subset, windows=int(len(recs)), frames=int(len(recs) * T),
               pixel_samples=n, spatial_stride=stride, resize=resize, rate_hz=rate, T=T, seed=seed,
               mean=[round(float(x), 6) for x in mean], std=[round(float(x), 6) for x in std], seconds=round(time.perf_counter() - t, 1))
    print(json.dumps(out), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="456x256")
    ap.add_argument("--fps", nargs="+", default=["15", "30", "native"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-clips", type=int, default=4000)
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--rate", type=float, default=15.0)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="JSON file (list of per-store results)")
    a = ap.parse_args()
    results = [store_stats(a.res, "native" if f == "native" else int(f), a.n_clips, a.T, a.rate, a.resize, a.workers, a.seed, a.split)
               for f in a.fps]
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(results, indent=1))
    print("\n# paste into _configs/spatialvid_hq.py metadata['stats']:")
    print(json.dumps({r["store"]: {"mean": r["mean"], "std": r["std"]} for r in results}, indent=4))


if __name__ == "__main__":
    main()
