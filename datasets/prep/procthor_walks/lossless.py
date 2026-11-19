"""Lossless store of one split's walks: frames byte-identical to the H5, positions/headings exactly as stored (float64).

    python -m datasets.prep.procthor_walks.lossless --dataset procthor-walks-objects --store-root <dir> \
        [--split-version rtq-r160 --split val_unseen] [--workers 16] [--verify-only]

Why: the h264 stores are near-lossless (crf 10, ~42.6 dB); benchmarks scored on the original H5 frames (tjepa-dev V3:
rtq-r160 val_unseen, 16 windows of 61 frames per walk) need the exact pixels to reproduce their numbers.

Store `<dataset>-lossless-160x120-<split>` (one record per walk, in split order = sorted clip_id; clip_id as in the h264
stores, `val/house_XXXXX/seed_XXXX`):
    video        bytes           mp4: H.264 RGB lossless (libx264rgb, qp 0, rgb24), GOP 60, 30 fps, frame i at pts i
    positions    float64[1000,2] source agent_positions, unchanged
    headings     float64[1000]   source agent_headings (1000, 1) flattened, unchanged
    fps / num_frames / duration_s  30 / 1000 / 33.33 (container convention, as the h264 stores)
plus records.parquet (the h264 store's walk metadata; `record_idx` renumbered 0..n-1, `h264_record_idx` = the record in
the h264 store of the same folder) and store_manifest.json, whose `verify` block records the checks below.

Verification (every walk, nothing sampled; exit 1 on any failure):
  1. all 1,000 frames decoded with torchcodec (`get_frames_in_range`) == H5 `visual_scenes`, byte for byte;
  2. the 16 benchmark windows (61 frames from 0, 60, ..., 900) decoded through slipstream's `DecodeVideoWindow`
     (t0 = (start + 0.5) / 30, rate 30 Hz) == the H5 slices;
  3. stored positions / headings == H5 arrays (dtype float64, identical bytes);
  4. source sha256 per file, and a tree hash sha256 over sorted "relpath sha256" lines (relpath under the cell dir).
Trial 2026-11-19: libx264 yuv444p qp 0 is *not* byte-identical (RGB->YUV rounding); libx264rgb qp 0 is, at ~17 MB/walk
(H5 ~27 MB/walk).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DATASETS, FPS, N_STEPS, RES, SRC_DATA, read_meta, software_versions

FMT = "lossless"
ENCODE = dict(codec="libx264rgb", qp=0, preset="medium", pix_fmt="rgb24", gop=60, fps=30, threads=1)
SHARED = Path("/n/netscratch/alvarez_lab/Lab/datasets/slipstream")
STARTS = list(range(0, 901, 60))
T = 61


def store_name(dataset: str, split: str) -> str:
    return f"{dataset}-{FMT}-{RES}-{split}"


def read_walk64(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """frames uint8 (1000,120,160,3), positions float64 (1000,2), headings float64 (1000,): no casts."""
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(path, "r") as f:
        g = f["trajectory"]
        frames = g["visual_scenes"][:]
        pos = np.asarray(g["agent_positions"][:])
        head = np.asarray(g["agent_headings"][:]).reshape(-1)
        read_meta(f)
    if pos.dtype != np.float64 or head.dtype != np.float64:
        raise ValueError(f"{path}: positions {pos.dtype}, headings {head.dtype}; expected float64")
    return frames, pos, head


def encode_lossless(frames: np.ndarray) -> bytes:
    import av
    e = ENCODE
    b = io.BytesIO()
    with av.open(b, "w", format="mp4") as c:
        s = c.add_stream(e["codec"], rate=e["fps"])
        s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], e["pix_fmt"]
        s.time_base = Fraction(1, e["fps"])
        s.thread_count = e["threads"]
        s.options = {"qp": str(e["qp"]), "preset": e["preset"], "g": str(e["gop"]), "keyint_min": str(e["gop"]),
                     "sc_threshold": "0"}
        for i, fr in enumerate(frames):
            vf = av.VideoFrame.from_ndarray(fr, format="rgb24")
            vf.pts = i
            for p in s.encode(vf):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return b.getvalue()


class WalkSource:
    def __init__(self, paths: list[str], out: Path):
        self.paths, self.out = paths, out

    @property
    def cache_path(self) -> Path:
        return self.out

    @property
    def field_types(self) -> dict[str, str]:
        return {"video": "bytes", "positions": f"float64[{N_STEPS},2]", "headings": f"float64[{N_STEPS}]",
                "fps": "float", "num_frames": "int", "duration_s": "float"}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        frames, pos, head = read_walk64(self.paths[i])
        if frames.shape[0] != N_STEPS:
            raise ValueError(f"{self.paths[i]}: {frames.shape[0]} frames, expected {N_STEPS}")
        return {"video": encode_lossless(frames), "positions": pos, "headings": head,
                "fps": float(FPS), "num_frames": N_STEPS, "duration_s": N_STEPS / FPS}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(store: Path, rec: pd.DataFrame, threads: int) -> dict:
    import torch
    torch.set_num_threads(1)
    from slipstream.cache import OptimizedCache
    from slipstream.decoders import DecodeVideoWindow
    from torchcodec.decoders import VideoDecoder

    cache = OptimizedCache.load(store, verbose=False)
    fv, fp, fh = cache.fields["video"], cache.fields["positions"], cache.fields["headings"]
    stage = DecodeVideoWindow(T=T, rate_hz=FPS, t0_key="t0", device="cpu", num_workers=4)

    def read(field, i):
        out = field.load_batch(np.array([i], dtype=np.int64), parallel=False)
        return {k: np.array(v, copy=True) for k, v in out.items()}

    def one(i: int) -> dict:
        path = rec["src_path"].iloc[i]
        frames, pos, head = read_walk64(path)
        v = read(fv, i); blob = bytes(v["data"][0][: int(v["sizes"][0])])
        allf = VideoDecoder(blob, dimension_order="NHWC").get_frames_in_range(0, N_STEPS).data.numpy()
        frames_ok = allf.shape == frames.shape and np.array_equal(allf, frames)
        sp = np.asarray(read(fp, i)["data"][0]); sh = np.asarray(read(fh, i)["data"][0])
        pos_ok = sp.dtype == np.float64 and sp.reshape(pos.shape).tobytes() == pos.tobytes()
        head_ok = sh.dtype == np.float64 and sh.reshape(head.shape).tobytes() == head.tobytes()
        return dict(i=i, frames=frames_ok, pos=pos_ok, head=head_ok, sha=sha256_file(path), mb=len(blob) / 1e6)

    t = time.perf_counter()
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(one, range(len(rec))))
    # benchmark windows through the training/eval decode stage, every walk x 16 starts
    win_bad = []
    for i in range(len(rec)):
        frames, _, _ = read_walk64(rec["src_path"].iloc[i])
        v = read(fv, i); n = len(STARTS)
        t0 = np.array([(s + 0.5) / FPS for s in STARTS], np.float32)        # as the loader passes sample_data
        out = stage({"data": np.repeat(v["data"][:1], n, axis=0), "sizes": np.repeat(v["sizes"][:1], n),
                     "field": "video", "sample_data": {"t0": t0}})
        got = out["video"].permute(0, 1, 3, 4, 2).cpu().numpy()            # [16, T, H, W, 3]
        for k, s in enumerate(STARTS):
            if not np.array_equal(got[k], frames[s:s + T]):
                win_bad.append(f"{rec['clip_id'].iloc[i]}@{s}")
    cell = Path(rec["src_path"].iloc[0]).parents[2]
    lines = sorted(f"{Path(rec['src_path'].iloc[r['i']]).relative_to(cell)} {r['sha']}" for r in res)
    out = dict(
        walks=len(rec), windows=len(rec) * len(STARTS),
        frames_byte_identical=sum(r["frames"] for r in res), positions_identical=sum(r["pos"] for r in res),
        headings_identical=sum(r["head"] for r in res), windows_byte_identical=len(rec) * len(STARTS) - len(win_bad),
        window_failures=win_bad[:20], decode="torchcodec get_frames_in_range (all frames) + slipstream DecodeVideoWindow "
        f"(T={T}, rate {FPS} Hz, t0=(start+0.5)/{FPS}, starts {STARTS[0]}..{STARTS[-1]} step 60)",
        source_tree_sha256=hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(),
        source_tree_hash_rule="sha256 of sorted lines '<relpath under cell dir> <file sha256>' joined by newline",
        source_files_sha256=dict(l.split(" ") for l in lines),
        video_mb_per_walk=round(float(np.mean([r["mb"] for r in res])), 2),
        seconds=round(time.perf_counter() - t), verified=time.strftime("%Y-%m-%dT%H:%M:%S"))
    out["ok"] = (out["frames_byte_identical"] == out["positions_identical"] == out["headings_identical"] == len(rec)
                 and not win_bad)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--store-root", required=True, type=Path)
    ap.add_argument("--split-version", default="rtq-r160")
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args(argv)
    from slipstream.cache import OptimizedCache

    split = pd.read_parquet(SHARED / a.dataset / "splits" / f"{a.split_version}.parquet")
    ids = sorted(split.loc[split["split"] == a.split, "clip_id"])
    folders = {c.split("/", 1)[0] for c in ids}
    if len(folders) != 1:
        sys.exit(f"split {a.split} spans folders {sorted(folders)}")
    folder = folders.pop()
    h264 = pd.read_parquet(SHARED / f"{a.dataset}-h264-{RES}-{folder}" / "records.parquet")
    rec = h264.set_index("clip_id").loc[ids].reset_index().rename(columns={"record_idx": "h264_record_idx"})
    rec.insert(0, "record_idx", range(len(rec)))
    rec = rec[["record_idx", "clip_id"] + [c for c in rec.columns if c not in ("record_idx", "clip_id")]]
    name = store_name(a.dataset, a.split)
    store = a.store_root / name
    if not a.verify_only:
        if store.exists() and any(store.iterdir()):
            sys.exit(f"{store} exists and is not empty; remove it to rebuild")
        print(f"{a.dataset}: {len(rec)} walks of {a.split_version}/{a.split} -> {store}, {ENCODE}", flush=True)
        t = time.perf_counter()
        cache = OptimizedCache.build(WalkSource(rec["src_path"].tolist(), store), output_dir=store, verbose=True,
                                     num_workers=a.workers)
        if len(cache) != len(rec):
            sys.exit(f"store has {len(cache)} records, expected {len(rec)}")
        rec.to_parquet(store / "records.parquet", index=False)
        vbytes = sum(p.stat().st_size for p in store.glob("video*") if p.is_file())
        sm = dict(dataset=a.dataset, store=name, folder=folder, split_version=a.split_version, split=a.split,
                  num_records=len(rec), cell=DATASETS[a.dataset]["cell"], source=str(SRC_DATA), encode=ENCODE,
                  fields=WalkSource([], store).field_types, software=software_versions(), video_bytes=vbytes,
                  encode_seconds=round(time.perf_counter() - t), built=time.strftime("%Y-%m-%dT%H:%M:%S"))
        (store / "store_manifest.json").write_text(json.dumps(sm, indent=1))
        print(f"BUILT {name}: {len(rec)} records, video {vbytes / 1e9:.2f} GB", flush=True)
    v = verify(store, pd.read_parquet(store / "records.parquet"), a.workers)
    sm = json.loads((store / "store_manifest.json").read_text())
    sm["verify"] = v
    (store / "store_manifest.json").write_text(json.dumps(sm, indent=1))
    short = {k: v[k] for k in v if k != "source_files_sha256"}
    print("VERIFY " + ("OK " if v["ok"] else "FAILED ") + json.dumps(short), flush=True)
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
