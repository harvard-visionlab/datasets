"""Shared constants and helpers for the ProcTHOR walks prep (see README.md in this folder).

Source: Rupert Tawiah-Quashie's (Harvard Vision Lab) AI2-THOR renders of ProcTHOR-10k houses, one H5 file per
1,000-step Brownian walk. Two cells with identical houses, seeds and camera paths: furnished ("objects") and the same
houses with no furniture ("empty").
"""
from __future__ import annotations

import io
import json
import os
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")      # shared scratch: H5 locking can hang

import numpy as np

SRC_ROOT = Path("/n/netscratch/kempner_konkle_lab/Everyone/rtawiahquashie/habitat_data")
SRC_DATA = SRC_ROOT / "datasets_r160_zstd"                   # <cell>/{train,val,test}/house_XXXXX/seed_XXXX.h5
SRC_RUNS = SRC_ROOT / "runs"                                 # <run>/active_reproducibility/{train,val}_windows.tsv

# dataset name -> source cell, and the student's runs that trained on that cell (all share one window list; checked
# by build_index)
DATASETS = {
    "procthor-walks-objects": dict(cell="pt_objects_brownian"),
    "procthor-walks-empty": dict(cell="pt_empty_brownian"),
}
RUN_SETTINGS = ("bridge", "lr6e4", "lrsplit", "cov01")
FOLDERS = ("train", "val", "test")                           # record order: folder, then house, then seed

N_STEPS = 1000
H, W = 120, 160
RES = f"{W}x{H}"
FMT = "h264"
# Encode settings, chosen from benchmarks/procthor_encode_trial.py (2026-10-07): yuv444p avoids the ~38 dB chroma ceiling
# of 4:2:0 on 160x120 renders; crf 10 = 42.6 dB mean PSNR at 2.4 MB/walk (h5 source: 28 MB/walk). Keyframe every 60
# frames = the training window stride, so every window [60k, 60k+61) starts on a keyframe. One x264 thread for a
# deterministic bitstream (parallelism comes from encoding many walks at once).
ENCODE = dict(codec="libx264", crf=10, preset="medium", pix_fmt="yuv444p", gop=60, fps=30, threads=1)
# The walks have no time axis (step-indexed random walk; the H5 "fps" field is render throughput). 30 fps is a container
# convention: frame i is at t = i / 30 s.
FPS = ENCODE["fps"]


def store_name(dataset: str, folder: str) -> str:
    """One store per source folder: e.g. procthor-walks-objects-h264-160x120-val."""
    return f"{dataset}-{FMT}-{RES}-{folder}"


def clip_id(folder: str, house: str, seed: str) -> str:
    """Same id in both datasets for paired walks, e.g. 'val/house_00000/seed_1000'."""
    return f"{folder}/{house}/{seed}"


def read_walk(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """frames (1000,120,160,3) uint8, positions (1000,2) float32, headings (1000,) float32 radians, metadata dict."""
    import h5py
    import hdf5plugin  # noqa: F401  -- blosc/zstd filter of the frames
    with h5py.File(path, "r") as f:
        g = f["trajectory"]
        frames = g["visual_scenes"][:]
        pos = np.asarray(g["agent_positions"][:], np.float32)
        head = np.asarray(g["agent_headings"][:], np.float32).reshape(-1)
        meta = read_meta(f)
    return frames, pos, head, meta


def read_meta(f) -> dict:
    raw = f["meta"]["metadata"][()]
    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)


def encode_walk(frames: np.ndarray) -> bytes:
    """One walk -> mp4 bytes (h264 yuv444p crf 10, GOP 60, 30 fps, pts = frame index)."""
    import av
    e = ENCODE
    b = io.BytesIO()
    with av.open(b, "w", format="mp4") as c:
        s = c.add_stream(e["codec"], rate=e["fps"])
        s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], e["pix_fmt"]
        s.time_base = Fraction(1, e["fps"])
        s.thread_count = e["threads"]
        s.options = {"crf": str(e["crf"]), "preset": e["preset"], "g": str(e["gop"]), "keyint_min": str(e["gop"]),
                     "sc_threshold": "0"}
        for i, fr in enumerate(frames):
            vf = av.VideoFrame.from_ndarray(fr, format="rgb24")
            vf.pts = i
            for p in s.encode(vf):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return b.getvalue()


def software_versions() -> dict:
    import av
    out = {"pyav": av.__version__, "libav": {k: ".".join(map(str, v)) for k, v in av.library_versions.items()}}
    try:
        import slipstream
        out["slipstream"] = getattr(slipstream, "__version__", "?")
    except ImportError:
        pass
    return out
