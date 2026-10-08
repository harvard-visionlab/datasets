"""Side-by-side quality check: H5 original vs h264 yuv444p crf10 (the proposed store) vs JPEG q100 4:2:0, plus |diff|.

Encodes each picked walk with the store's settings (``procthor_encode_trial.encode_mp4``), decodes 61-frame windows the
way training does (torchcodec ``get_frames_in_range``), and writes per-frame PNG rows
``original | mp4 | jpeg | 4x|orig - mp4|`` at 3x nearest upscale, plus ``index.tsv`` (walk, frame, PSNR mp4/jpeg).

    uv run --no-sync python benchmarks/procthor_quality_sbs.py --out $LAB_SCRATCH/procthor-quality --n-walks 4
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from procthor_encode_trial import DATA, decode_mp4, encode_mp4, psnr, read_walk  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="pt_objects_brownian,pt_empty_brownian")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-walks", type=int, default=4)
    ap.add_argument("--frames", default="0,200,480,730,960")
    ap.add_argument("--crf", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    from PIL import Image
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = ["cell\twalk\tframe\tpsnr_mp4\tpsnr_jpeg\tpng"]
    rng = np.random.default_rng(a.seed)
    houses = sorted(os.listdir(DATA / a.cells.split(",")[0] / a.split))
    picks = [houses[i] for i in rng.choice(len(houses), a.n_walks, replace=False)]
    for cell in a.cells.split(","):
        for house in picks:                                       # same houses (and seeds) in both cells: paired walks
            path = DATA / cell / a.split / house / "seed_1000.h5"
            frames = read_walk(path)
            blob = encode_mp4(frames, "libx264", a.crf, pix_fmt="yuv444p")
            for fi in map(int, a.frames.split(",")):
                start = min(fi // 60 * 60, 900)
                mp4 = decode_mp4(blob, start)[fi - start]
                b = io.BytesIO()
                Image.fromarray(frames[fi]).save(b, format="JPEG", quality=100, subsampling=2)
                jpg = np.asarray(Image.open(b).convert("RGB"))
                diff = np.clip(np.abs(frames[fi].astype(int) - mp4.astype(int)) * 4, 0, 255).astype(np.uint8)
                row = np.concatenate([frames[fi], mp4, jpg, diff], axis=1)
                name = f"{cell}_{house}_f{fi:03d}.png"
                Image.fromarray(row).resize((row.shape[1] * 3, row.shape[0] * 3), Image.NEAREST).save(out / name)
                rows.append(f"{cell}\t{house}/seed_1000\t{fi}\t{psnr(frames[fi], mp4):.2f}\t{psnr(frames[fi], jpg):.2f}\t{name}")
            print(f"{cell} {house}: mp4 {len(blob) / 1e6:.2f} MB/walk", flush=True)
    (out / "index.tsv").write_text("\n".join(rows) + "\n")
    print(f"wrote {len(rows) - 1} rows to {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
