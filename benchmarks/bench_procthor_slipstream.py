"""Multi-epoch slipstream throughput on a trial ProcTHOR r160 store (``procthor_build_stores.py``), same windows and
same GPU touch as ``bench_procthor_h5.py``, so the numbers compare one to one.

    jpeg store: per-frame records, ``window=(61, 1)``, anchors walk*1000 + start, ``DecodeOnly`` (turbojpeg) + stack
    mp4 store : per-walk records, ``DecodeVideoWindow(T=61, rate_hz=30)`` with t0 = (start + 0.5) / 30 (mid-frame,
                so float rounding cannot pick the neighbour), positions/headings sliced from the per-walk arrays

``--verify N`` first compares N windows against the H5 frames (PSNR, max abs diff; positions must match exactly).

    OMP_NUM_THREADS=1 uv run --no-sync --group video python benchmarks/bench_procthor_slipstream.py \
        --loader $SANDBOX_DIR/procthor/procthor_r160_loader.py --store $LAB_SCRATCH/procthor-stores/jpeg_q100_420/val \
        --format jpeg --threads 16 --epochs 3 --drop-cache --verify 64
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_procthor_h5 import drop_page_cache, load_module  # noqa: E402

FPS, LEN_WINDOW = 30, 61


def unpack(batch, fmt, B):
    """-> frames uint8 (B, 61, 120, 160, 3) CPU or as delivered, positions (B, 61, 2), headings (B, 61), (walk, start)."""
    import torch
    if fmt == "jpeg":
        imgs = batch["image"]
        frames = torch.from_numpy(np.stack(imgs).reshape(B, LEN_WINDOW, 120, 160, 3)) if isinstance(imgs, list) else imgs
        anchors = batch["_anchors"].cpu().numpy() if hasattr(batch["_anchors"], "cpu") else np.asarray(batch["_anchors"])
        return frames, batch["positions"], batch["headings"], (anchors // 1000, anchors % 1000)
    v = batch["video"]                                          # [B, T, 3, H, W] CHW view over HWC memory
    frames = v.permute(0, 1, 3, 4, 2)
    start = batch["start"].long()
    idx = start[:, None] + torch.arange(LEN_WINDOW)[None]
    pos = torch.gather(batch["positions"], 1, idx[..., None].expand(-1, -1, 2))
    head = torch.gather(batch["headings"], 1, idx)
    return frames, pos, head, (batch["walk"].cpu().numpy(), start.numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loader", required=True, help="procthor_r160_loader.py (window list + H5 reference for --verify)")
    ap.add_argument("--run", default="pt_objects_brownian__bridge_r160")
    ap.add_argument("--split", default="val")
    ap.add_argument("--store", required=True)
    ap.add_argument("--format", required=True, choices=["jpeg", "mp4"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=16, help="decode threads (DecodeOnly) / decoder workers (video)")
    ap.add_argument("--batches-ahead", type=int, default=3)
    ap.add_argument("--drop-cache", action="store_true")
    ap.add_argument("--verify", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    from slipstream.dataset import SlipstreamDataset
    from slipstream.loader import SlipstreamLoader

    m = load_module(a.loader)
    store = Path(a.store)
    walk_of = {p: i for i, p in enumerate(store.joinpath("walks.txt").read_text().split())}
    windows = m.training_windows(a.run, a.split)
    w_idx = np.array([walk_of[p] for p, _ in windows], np.int64)
    starts = np.array([s for _, s in windows], np.int64)
    files = [p for p in store.iterdir() if p.is_file()]
    gb = sum(p.stat().st_size for p in files) / 1e9
    print(f"{a.format} store {store} ({gb:.1f} GB): {len(windows):,} windows from {len(walk_of):,} walks", flush=True)

    sds = SlipstreamDataset(local_dir=str(store))
    if a.format == "jpeg":
        from slipstream.decoders import DecodeOnly
        loader = SlipstreamLoader(sds, batch_size=a.batch_size, shuffle=True, seed=a.seed, drop_last=False,
                                  indices=w_idx * 1000 + starts, window=(LEN_WINDOW, 1), image_field="image",
                                  pipelines={"image": [DecodeOnly(num_threads=a.threads)]},
                                  batches_ahead=a.batches_ahead, exclude_fields=["walk", "step"], verbose=False)
    else:
        from slipstream.decoders import DecodeVideoWindow
        stage = DecodeVideoWindow(T=LEN_WINDOW, rate_hz=FPS, t0_key="t0", num_workers=a.threads, num_ffmpeg_threads=1)
        loader = SlipstreamLoader(sds, batch_size=a.batch_size, shuffle=True, seed=a.seed, drop_last=False, indices=w_idx,
                                  sample_data={"t0": (starts + 0.5) / FPS, "start": starts}, image_field="video",
                                  pipelines={"video": [stage]}, batches_ahead=max(a.batches_ahead, -(-a.threads // a.batch_size) + 2),
                                  verbose=False)

    if a.verify:
        walks = store.joinpath("walks.txt").read_text().split()
        ref = m.WindowDataset([])
        n, psnrs, maxd, pos_err = 0, [], 0, 0.0
        for batch in loader:
            B = len(batch["_anchors"]) if a.format == "jpeg" else batch["video"].shape[0]
            frames, pos, head, (wk, st) = unpack(batch, a.format, B)
            frames = frames.cpu().numpy()
            for i in range(B):
                ref.windows = [(walks[int(wk[i])], int(st[i]))]
                rf, rp, rh = ref[0]
                d = frames[i].astype(np.float32) - rf.numpy()
                psnrs.append(10 * np.log10(255 ** 2 / max(np.mean(d ** 2), 1e-12)))
                maxd = max(maxd, int(np.abs(d).max()))
                pos_err = max(pos_err, float((pos[i].cpu() - rp).abs().max()), float((head[i].cpu() - rh).abs().max()))
                n += 1
                if n >= a.verify:
                    break
            if n >= a.verify:
                break
        print(f"VERIFY {n} windows vs H5: PSNR mean {np.mean(psnrs):.2f} dB min {np.min(psnrs):.2f} dB, max abs diff {maxd}, "
              f"max pose diff {pos_err:.3g}", flush=True)
        loader.set_epoch(0) if hasattr(loader, "set_epoch") else None

    dev = torch.device(a.device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    if a.drop_cache:
        t = time.perf_counter()
        drop_page_cache(files)
        print(f"dropped page cache for {len(files)} store files in {time.perf_counter() - t:.1f} s", flush=True)
    rows = []
    for ep in range(a.epochs):
        n_win, first, acc = 0, None, torch.zeros((), device=dev)
        t = time.perf_counter()
        for batch in loader:
            B = len(batch["_anchors"]) if a.format == "jpeg" else batch["video"].shape[0]
            frames, pos, head, _ = unpack(batch, a.format, B)
            x = frames.to(dev).float().add_(0.01)
            acc += x[:, :, 0, 0, 0].sum() + pos.to(dev).sum() + head.to(dev).sum()
            n_win += B
            if first is None:
                sync()
                first = time.perf_counter() - t
        sync()
        dt = time.perf_counter() - t
        rows.append(n_win / dt)
        print(f"epoch {ep + 1}: {n_win / dt:,.1f} windows/s  {n_win * LEN_WINDOW / dt:,.0f} frames/s  {dt:.1f} s  "
              f"first batch {first:.1f} s  ({n_win} windows, check {acc.item():.3g})", flush=True)
    loader.shutdown()

    cold, warm = rows[0], (float(np.mean(rows[1:])) if len(rows) > 1 else float("nan"))
    print(f"SUMMARY slipstream-{a.format} {a.tag} B={a.batch_size} threads={a.threads} windows={len(windows)} {gb:.1f} GB: "
          f"epoch1 {cold:,.1f} windows/s ({cold * LEN_WINDOW:,.0f} frames/s), epochs2+ {warm:,.1f} windows/s "
          f"({warm * LEN_WINDOW:,.0f} frames/s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
