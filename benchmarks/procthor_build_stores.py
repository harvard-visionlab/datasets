"""Build trial slipstream stores of ProcTHOR r160 walks (one window list's walks, e.g. a run's val split).

    --format jpeg : one record per frame: image (JPEG q100 4:2:0, ImageBytes), positions float32[2], headings float32,
                    walk int, step int. Windows = loader ``window=(61, 1)`` with anchors walk*1000 + start.
    --format mp4  : one record per walk: video (h264 yuv444p crf10, keyframe every 60 frames, 30 fps; bytes),
                    positions float32[1000,2], headings float32[1000]. Windows = ``DecodeVideoWindow`` with t0.

Writes ``walks.txt`` (H5 path per walk, in store order) next to the store; the benchmark maps window lists through it.

    uv run --no-sync python benchmarks/procthor_build_stores.py --loader $SANDBOX_DIR/procthor/procthor_r160_loader.py \
        --format jpeg --out $LAB_SCRATCH/procthor-stores/jpeg_q100_420/val --workers 8
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np

N_STEPS = 1000


def read_walk(path: str):
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(path, "r") as f:
        g = f["trajectory"]
        return (g["visual_scenes"][:], np.asarray(g["agent_positions"][:], np.float32),
                np.asarray(g["agent_headings"][:], np.float32).reshape(-1))


class FrameSource:
    """Per-frame records; caches the current walk (workers get contiguous ranges, so each walk is read once)."""

    def __init__(self, walks: list[str], out: Path, quality: int = 100, subsampling: int = 2):
        self.walks, self.out, self.quality, self.subsampling = walks, out, quality, subsampling
        self._cur, self._data = None, None

    cache_path = property(lambda self: self.out)
    field_types = property(lambda self: {"image": "ImageBytes", "positions": "float32[2]", "headings": "float32",
                                         "walk": "int", "step": "int"})

    def __len__(self):
        return len(self.walks) * N_STEPS

    def __getitem__(self, i):
        from PIL import Image
        w, s = divmod(int(i), N_STEPS)
        if self._cur != w:
            self._cur, self._data = w, read_walk(self.walks[w])
        frames, pos, head = self._data
        b = io.BytesIO()
        Image.fromarray(frames[s]).save(b, format="JPEG", quality=self.quality, subsampling=self.subsampling)
        return {"image": b.getvalue(), "positions": pos[s], "headings": head[s], "walk": w, "step": s}


class WalkSource:
    """One record per walk: the whole walk as one mp4."""

    def __init__(self, walks: list[str], out: Path, crf: int = 10, pix_fmt: str = "yuv444p"):
        self.walks, self.out, self.crf, self.pix_fmt = walks, out, crf, pix_fmt

    cache_path = property(lambda self: self.out)
    field_types = property(lambda self: {"video": "bytes", "positions": f"float32[{N_STEPS},2]",
                                         "headings": f"float32[{N_STEPS}]", "walk": "int"})

    def __len__(self):
        return len(self.walks)

    def __getitem__(self, i):
        sys.path.insert(0, str(Path(__file__).parent))
        from procthor_encode_trial import encode_mp4
        frames, pos, head = read_walk(self.walks[i])
        return {"video": encode_mp4(frames, "libx264", self.crf, pix_fmt=self.pix_fmt), "positions": pos,
                "headings": head, "walk": int(i)}


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("procthor_r160_loader", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loader", required=True)
    ap.add_argument("--run", default="pt_objects_brownian__bridge_r160")
    ap.add_argument("--split", default="val")
    ap.add_argument("--format", required=True, choices=["jpeg", "mp4"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-walks", type=int, default=0)
    a = ap.parse_args()

    from slipstream.cache import OptimizedCache
    m = load_module(a.loader)
    walks = sorted({p for p, _ in m.training_windows(a.run, a.split)})
    if a.max_walks:
        walks = walks[:a.max_walks]
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty")
    src = FrameSource(walks, out) if a.format == "jpeg" else WalkSource(walks, out)
    print(f"{a.format}: {len(walks):,} walks -> {len(src):,} records -> {out}", flush=True)
    t = time.perf_counter()
    cache = OptimizedCache.build(src, output_dir=out, verbose=True, num_workers=a.workers)
    (out / "walks.txt").write_text("\n".join(walks) + "\n")
    gb = sum(p.stat().st_size for p in out.iterdir() if p.is_file()) / 1e9
    print(f"BUILT {a.format} {len(cache):,} records, {gb:.2f} GB, {time.perf_counter() - t:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
