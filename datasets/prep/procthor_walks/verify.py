"""Stage 3: verify a built store against its H5 sources.

    python -m datasets.prep.procthor_walks.verify --dataset procthor-walks-objects --store <store dir> [--windows 256] [--threads 16]

1. every record: the mp4 parses, has 1,000 frames at 30 fps, 160x120 (torchcodec metadata; no full decode);
2. records.parquet matches the stage-1 index order (clip_id per record_idx) when --out is given;
3. `--windows` random training windows (start in 0, 60, ..., 900; spread over all folders), decoded the way training
   decodes them (torchcodec get_frames_in_range on the record's bytes), compared with the H5 frames:
   PSNR mean >= 40 dB and min >= 33 dB, positions and headings bit-identical.
Prints one VERIFY line; exit code 1 on any failure.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DATASETS, FPS, H, N_STEPS, W, read_walk

LEN_WINDOW = 61


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="stage-1 work dir (checks records.parquet against index/walks.parquet)")
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    from slipstream.cache import OptimizedCache
    from torchcodec.decoders import VideoDecoder

    cache = OptimizedCache.load(a.store, verbose=False)
    rec = pd.read_parquet(a.store / "records.parquet")
    fails = []
    if len(cache) != len(rec):
        fails.append(f"store has {len(cache)} records, records.parquet {len(rec)}")
    if a.out:
        idx = pd.read_parquet(a.out / a.dataset / "index" / "walks.parquet")
        if not (len(idx) == len(rec) and (idx["clip_id"].to_numpy() == rec["clip_id"].to_numpy()).all()):
            fails.append("records.parquet clip_id order differs from index/walks.parquet")
    vf = cache.fields["video"]

    def blob(i: int) -> bytes:
        out = vf.load_batch(np.array([i], dtype=np.int64), parallel=False)
        return bytes(out["data"][0][: int(out["sizes"][0])])

    def meta_ok(i: int) -> str | None:
        m = VideoDecoder(blob(i), seek_mode="exact").metadata
        if (m.num_frames, m.width, m.height) != (N_STEPS, W, H) or abs((m.average_fps or 0) - FPS) > 1e-3:
            return f"record {i}: frames {m.num_frames} {m.width}x{m.height} fps {m.average_fps}"
        return None

    with ThreadPoolExecutor(a.threads) as ex:
        bad = [r for r in ex.map(meta_ok, range(len(rec))) if r]
    fails += bad[:10]

    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(rec), min(a.windows, len(rec)), replace=False)
    starts = rng.integers(0, 16, len(pick)) * 60
    pos_f = cache.fields["positions"]
    head_f = cache.fields["headings"]

    def check(k: int) -> tuple[float, bool]:
        i, s = int(pick[k]), int(starts[k])
        frames, pos, head, _ = read_walk(rec["src_path"].iloc[i])
        d = VideoDecoder(blob(i), seek_mode="exact", num_ffmpeg_threads=1, dimension_order="NHWC")
        win = d.get_frames_in_range(s, s + LEN_WINDOW).data.numpy()
        p = np.asarray(pos_f.load_batch(np.array([i], dtype=np.int64))["data"][0])
        h = np.asarray(head_f.load_batch(np.array([i], dtype=np.int64))["data"][0])
        same = np.array_equal(p.reshape(N_STEPS, 2), pos) and np.array_equal(h.reshape(N_STEPS), head)
        return psnr(win, frames[s:s + LEN_WINDOW]), same

    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(check, range(len(pick))))
    ps = np.array([r[0] for r in res])
    if not all(r[1] for r in res):
        fails.append(f"{sum(not r[1] for r in res)} windows with positions/headings != H5")
    if ps.mean() < 40 or ps.min() < 33:
        fails.append(f"PSNR mean {ps.mean():.2f} / min {ps.min():.2f} below 40 / 33 dB")
    folders = rec["folder"].iloc[pick].value_counts().to_dict()
    print(f"VERIFY {a.dataset} {a.store.name}: {len(rec):,} records (mp4 metadata ok: {len(rec) - len(bad):,}); "
          f"{len(pick)} windows {folders}: PSNR mean {ps.mean():.2f} dB, min {ps.min():.2f} dB, poses identical: "
          f"{all(r[1] for r in res)} -> {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
