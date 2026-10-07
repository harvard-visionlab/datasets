"""Re-encode a sample of ProcTHOR r160 H5 walks as per-frame JPEG and as mp4; report size, PSNR and decode cost.

For each picked walk (1000 frames 120x160 RGB) and each codec: encoded bytes per walk, PSNR vs the H5 frames (all
961 frames the 16 windows cover), and single-thread decode time per 61-frame window (windows start every 60 frames, as in training).
JPEG decode uses torchvision's libjpeg-turbo ``decode_jpeg`` (a proxy for slipstream's turbojpeg path); mp4 decode
uses torchcodec ``get_frames_in_range`` (what slipstream's ``DecodeVideoWindow`` wraps), keyframe every 60 frames.
Saves side-by-side PNGs (original | each codec, 3x nearest upscale) for a few frames to ``--out``.

    uv run --no-sync --group procthor --group video python benchmarks/procthor_encode_trial.py \
        --cell pt_objects_brownian --n-walks 40 --out $LAB_SCRATCH/procthor-trial
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np

DATA = Path("/n/netscratch/kempner_konkle_lab/Everyone/rtawiahquashie/habitat_data/datasets_r160_zstd")
LEN_WINDOW, STRIDE = 61, 60

# name -> (kind, params). JPEG q=100 (not tuned for size); mp4 GOP = window stride, so every window starts on a keyframe.
CODECS = {
    "jpeg_q100_420": ("jpeg", dict(quality=100, subsampling=2)),
    "jpeg_q100_444": ("jpeg", dict(quality=100, subsampling=0)),
    "h264_crf18": ("mp4", dict(codec="libx264", crf=18)),
    "h264_crf23": ("mp4", dict(codec="libx264", crf=23)),
    "h265_crf23": ("mp4", dict(codec="libx265", crf=23)),
    "h265_crf29": ("mp4", dict(codec="libx265", crf=29)),
}


def read_walk(path: Path) -> np.ndarray:
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(path, "r") as f:
        return f["trajectory"]["visual_scenes"][:]


def encode_jpeg(frames: np.ndarray, quality: int, subsampling: int) -> list[bytes]:
    from PIL import Image
    out = []
    for fr in frames:
        b = io.BytesIO()
        Image.fromarray(fr).save(b, format="JPEG", quality=quality, subsampling=subsampling)
        out.append(b.getvalue())
    return out


def encode_mp4(frames: np.ndarray, codec: str, crf: int, fps: int = 30) -> bytes:
    import av
    b = io.BytesIO()
    with av.open(b, "w", format="mp4") as c:
        s = c.add_stream(codec, rate=fps)
        s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        s.time_base = Fraction(1, fps)
        if codec == "libx265":
            s.options = {"crf": str(crf), "preset": "medium", "x265-params": f"keyint={STRIDE}:min-keyint={STRIDE}:scenecut=0:log-level=error"}
        else:
            s.options = {"crf": str(crf), "preset": "medium", "g": str(STRIDE), "keyint_min": str(STRIDE), "sc_threshold": "0"}
        for i, fr in enumerate(frames):
            vf = av.VideoFrame.from_ndarray(fr, format="rgb24")
            vf.pts = i
            for p in s.encode(vf):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return b.getvalue()


def decode_jpeg_window(blobs: list[bytes], start: int) -> np.ndarray:
    import torch
    from torchvision.io import decode_jpeg
    ts = [torch.frombuffer(bytearray(x), dtype=torch.uint8) for x in blobs[start:start + LEN_WINDOW]]
    return torch.stack(decode_jpeg(ts)).permute(0, 2, 3, 1).numpy()


def decode_mp4_window(blob: bytes, start: int) -> np.ndarray:
    from torchcodec.decoders import VideoDecoder
    d = VideoDecoder(blob, seek_mode="exact", num_ffmpeg_threads=1, dimension_order="NHWC")
    return d.get_frames_in_range(start, start + LEN_WINDOW).data.numpy()


def decode_mp4_window_av(blob: bytes, start: int, fps: int = 30) -> np.ndarray:
    """PyAV fallback (no torchcodec FFmpeg libs): seek to the keyframe at ``start`` and decode 61 frames."""
    import av
    with av.open(io.BytesIO(blob)) as c:
        s = c.streams.video[0]
        s.thread_count = 1
        c.seek(int(start / fps / s.time_base), stream=s, backward=True, any_frame=False)
        out = []
        for f in c.decode(s):
            i = round(float(f.pts * s.time_base) * fps)
            if i >= start:
                out.append(f.to_ndarray(format="rgb24"))
                if len(out) == LEN_WINDOW:
                    break
    return np.stack(out)


MP4_DECODER = "torchcodec"


def decode_mp4(blob: bytes, start: int) -> np.ndarray:
    """torchcodec, or PyAV from the first failure on (torchcodec loads its FFmpeg libs lazily, at first decode)."""
    global MP4_DECODER
    if MP4_DECODER == "torchcodec":
        try:
            return decode_mp4_window(blob, start)
        except Exception as e:
            print(f"torchcodec unavailable ({type(e).__name__}); decoding mp4 with PyAV", flush=True)
            MP4_DECODER = "pyav"
    return decode_mp4_window_av(blob, start)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="pt_objects_brownian")
    ap.add_argument("--split", default="train", help="folder of the cell to sample walks from")
    ap.add_argument("--n-walks", type=int, default=40)
    ap.add_argument("--codecs", default=",".join(CODECS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sbs", type=int, default=4, help="walks to save side-by-side PNGs for")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    from PIL import Image

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    houses = sorted(os.listdir(DATA / a.cell / a.split))
    rng = np.random.default_rng(a.seed)
    walks = [DATA / a.cell / a.split / houses[i] / f"seed_{1000 + int(rng.integers(2))}.h5"
             for i in rng.choice(len(houses), a.n_walks, replace=False)]
    names = a.codecs.split(",")
    stats = {n: dict(bytes=[], psnr=[], dec_ms=[], enc_s=[]) for n in ["h5"] + names}

    for wi, path in enumerate(walks):
        t = time.perf_counter()
        frames = read_walk(path)
        stats["h5"]["dec_ms"].append(1e3 * (time.perf_counter() - t) / 16)   # whole walk / 16 windows: lower bound per window
        stats["h5"]["bytes"].append(path.stat().st_size)
        recon = {}
        for n in names:
            kind, kw = CODECS[n]
            t = time.perf_counter()
            enc = encode_jpeg(frames, **kw) if kind == "jpeg" else encode_mp4(frames, **kw)
            stats[n]["enc_s"].append(time.perf_counter() - t)
            stats[n]["bytes"].append(sum(map(len, enc)) if kind == "jpeg" else len(enc))
            dec = decode_jpeg_window if kind == "jpeg" else decode_mp4
            parts, times = [], []
            for s in range(0, 1000 - LEN_WINDOW + 1, STRIDE):
                t = time.perf_counter()
                w = dec(enc, s)
                times.append(time.perf_counter() - t)
                parts.append(w[:STRIDE])
            parts.append(w[STRIDE:])                                          # frame 960: last window ends there
            r = np.concatenate(parts)
            assert r.shape == (961,) + frames.shape[1:], r.shape
            stats[n]["dec_ms"].append(1e3 * float(np.median(times)))
            stats[n]["psnr"].append(psnr(frames[:961], r))
            recon[n] = r
        if wi < a.n_sbs:
            for fi in (0, 330, 660, 960):
                row = np.concatenate([frames[fi]] + [recon[n][fi] for n in names], axis=1)
                Image.fromarray(row).resize((row.shape[1] * 3, row.shape[0] * 3), Image.NEAREST).save(
                    out / f"sbs_w{wi}_f{fi:03d}.png")
        print(f"[{wi + 1}/{len(walks)}] {path.parent.name}/{path.name} " + " ".join(
            f"{n}={stats[n]['bytes'][-1] / 1e6:.1f}MB/{stats[n]['psnr'][-1]:.1f}dB" for n in names), flush=True)

    h5_mb = np.mean(stats["h5"]["bytes"]) / 1e6
    raw_mb = 1000 * 120 * 160 * 3 / 1e6
    print(f"\nSUMMARY {a.cell} n_walks={len(walks)} mp4 decoder={MP4_DECODER} (order: " + ", ".join(["h5"] + names) + ")\n"
          f"{'codec':<16}{'MB/walk':>9}{'vs h5':>8}{'vs raw':>8}{'PSNR dB':>9}{'ms/window':>11}{'enc s/walk':>11}")
    for n in ["h5"] + names:
        s = stats[n]
        mb = np.mean(s["bytes"]) / 1e6
        ps = f"{np.mean(s['psnr']):.2f}" if s["psnr"] else "lossless"
        enc = f"{np.mean(s['enc_s']):.1f}" if s["enc_s"] else "-"
        print(f"{n:<16}{mb:>9.2f}{mb / h5_mb:>8.3f}{mb / raw_mb:>8.3f}{ps:>9}{np.median(s['dec_ms']):>11.1f}{enc:>11}")
    print("(h5 ms/window = whole-walk read+decompress / 16; full dataset = 24,000 walks per cell)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
