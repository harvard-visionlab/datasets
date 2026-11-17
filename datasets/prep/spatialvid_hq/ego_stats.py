"""Ego-motion normalisation stats for SpatialVID-HQ (→ `metadata["stats"]["egomotion"]` in the config).

Same convention as `VideoDataset.ego_motion_at` / `video.ego_motion`: per frame step, `[dx dy dz rx ry rz]` in the
previous frame's camera axes (OpenCV: x right, y down, z forward), translation in raw MegaSaM pose units (no per-clip
rescale: their scale is consistent enough that clip speeds order correctly, checked by eye 2026-11-16), rotation in
radians. Sampled like training: seeded random windows of T frames at `rate_hz` from the train split (one per clip per
epoch, `ds.window_sampler(T=T)`), poses interpolated to the frame times (`t0 + k / rate_hz`), deltas between consecutive
frames. Mean/std per component (z-scoring = one global rescale, so clip ordering is untouched), plus quantiles for
the tails. No video is decoded.

    uv run --no-sync --group video python -m datasets.prep.spatialvid_hq.ego_stats --subset person_carried_v0 \
        --T 61 --rate-hz 15 --epochs 4 --out <tree>/index/ego_stats_person_carried_v0.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

COMPONENTS = ("dx", "dy", "dz", "rx", "ry", "rz")


def ego_stats(subset: str, T: int, rate_hz: float, epochs: int, seed: int, split: str, where: str | None,
              batch: int = 4096) -> dict:
    from visionlab.datasets import load

    ds = load("spatialvid-hq", split=split, subset=subset, rate_hz=rate_hz, where=where, download=False)
    sampler = ds.window_sampler(T=T, rate_hz=rate_hz, seed=seed)    # T / rate_hz, as the decoder (v0.23.0 used (T-1)/rate)
    k = np.arange(T) / rate_hz
    n = 0; s1 = np.zeros(6); s2 = np.zeros(6); chunks = []
    rng = np.random.default_rng(seed); t0_ = time.time()
    for ep in range(epochs):
        recs, t0 = sampler.sample(ep)
        for i in range(0, len(recs), batch):
            _, ego = ds.ego_motion_at(recs[i:i + batch], t0[i:i + batch, None] + k[None])
            e = ego.reshape(-1, 6).astype(np.float64)
            n += len(e); s1 += e.sum(0); s2 += (e ** 2).sum(0)
            chunks.append(e[rng.random(len(e)) < 0.02])            # 2 % reservoir for quantiles
        print(f"  epoch {ep}: {len(recs):,} windows, {n:,} steps, {time.time() - t0_:.0f} s", flush=True)
    mean = s1 / n; std = np.sqrt(s2 / n - mean ** 2)
    sample = np.concatenate(chunks)
    qs = [0.001, 0.01, 0.5, 0.99, 0.999]
    return dict(subset=subset, split=split, where=where, T=T, rate_hz=rate_hz, epochs=epochs, seed=seed,
                windows_per_epoch=int(len(sampler)), steps=int(n), components=list(COMPONENTS),
                translation_units="MegaSaM pose units (~8 m per unit at walking speed; not metric)", rotation_units="rad",
                mean=[float(f"{x:.6g}") for x in mean], std=[float(f"{x:.6g}") for x in std],
                quantiles={str(q): [float(f"{x:.4g}") for x in np.quantile(sample, q, axis=0)] for q in qs},
                seconds=round(time.time() - t0_, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="person_carried_v0")
    ap.add_argument("--split", default="train")
    ap.add_argument("--where", default=None, help="extra clip filter (pandas query on subset columns)")
    ap.add_argument("--T", type=int, default=61)
    ap.add_argument("--rate-hz", type=float, default=15.0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the JSON here too")
    a = ap.parse_args()
    out = ego_stats(a.subset, a.T, a.rate_hz, a.epochs, a.seed, a.split, a.where)
    print(json.dumps(out, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
