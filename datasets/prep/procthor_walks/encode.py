"""Stage 2: encode every walk of one dataset into its slipstream video store.

    python -m datasets.prep.procthor_walks.encode --dataset procthor-walks-objects --folder train --out <work> --store-root <dir> --workers 32

Reads <work>/<dataset>/index/walks.parquet (stage 1) and writes one store per source folder,
<store-root>/<dataset>-h264-160x120-<folder>/:
slipstream fields (one record per walk, record_idx = walks.parquet order)
    video        bytes           mp4: h264 yuv444p crf 10, GOP 60, 30 fps (common.ENCODE), frame i at pts i
    positions    float32[1000,2] agent (x, y) per step, metres (source agent_positions)
    headings     float32[1000]   agent heading per step, radians (source agent_headings)
    fps          float           30 (container convention; the walks have no time axis)
    num_frames   int             1000
    duration_s   float           num_frames / fps
plus records.parquet (record_idx -> clip_id, walk_idx (the stage-1 index across folders) and the walk metadata) and store_manifest.json
(encode settings, software versions, source, counts). `--limit N` builds a test store of the first N walks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from .common import DATASETS, ENCODE, FPS, N_STEPS, SRC_DATA, encode_walk, read_walk, software_versions, store_name


class WalkSource:
    """slipstream build source: record i = walk i of walks.parquet."""

    def __init__(self, paths: list[str], out: Path):
        self.paths, self.out = paths, out

    @property
    def cache_path(self) -> Path:
        return self.out

    @property
    def field_types(self) -> dict[str, str]:
        return {"video": "bytes", "positions": f"float32[{N_STEPS},2]", "headings": f"float32[{N_STEPS}]",
                "fps": "float", "num_frames": "int", "duration_s": "float"}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        frames, pos, head, _ = read_walk(self.paths[i])
        if frames.shape[0] != N_STEPS:
            raise ValueError(f"{self.paths[i]}: {frames.shape[0]} frames, expected {N_STEPS}")
        return {"video": encode_walk(frames), "positions": pos, "headings": head,
                "fps": float(FPS), "num_frames": N_STEPS, "duration_s": N_STEPS / FPS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--out", required=True, type=Path, help="work dir of stage 1")
    ap.add_argument("--store-root", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--folder", required=True, choices=["train", "val", "test"], help="one store per source folder")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    from slipstream.cache import OptimizedCache

    walks = pd.read_parquet(a.out / a.dataset / "index" / "walks.parquet")
    walks = walks[walks["folder"] == a.folder].rename(columns={"record_idx": "walk_idx"}).reset_index(drop=True)
    walks.insert(0, "record_idx", range(len(walks)))          # record_idx is per store; walk_idx = index across folders
    if a.limit:
        walks = walks.iloc[:a.limit]
    name = store_name(a.dataset, a.folder) + (f"-limit{a.limit}" if a.limit else "")
    store = a.store_root / name
    if store.exists() and any(store.iterdir()):
        sys.exit(f"{store} exists and is not empty; remove it to rebuild")
    print(f"{a.dataset}: encoding {len(walks):,} walks -> {store} with {a.workers} workers, {ENCODE}", flush=True)
    t = time.perf_counter()
    cache = OptimizedCache.build(WalkSource(walks["src_path"].tolist(), store), output_dir=store, verbose=True,
                                 num_workers=a.workers)
    dt = time.perf_counter() - t
    if len(cache) != len(walks):
        sys.exit(f"store has {len(cache)} records, expected {len(walks)}")
    walks.to_parquet(store / "records.parquet", index=False)
    vbytes = sum(p.stat().st_size for p in store.glob("video*") if p.is_file())
    sm = dict(dataset=a.dataset, store=name, folder=a.folder, num_records=len(walks), cell=DATASETS[a.dataset]["cell"], source=str(SRC_DATA),
              encode=ENCODE, fields=WalkSource([], store).field_types, software=software_versions(),
              video_bytes=vbytes, encode_seconds=round(dt), built=time.strftime("%Y-%m-%dT%H:%M:%S"),
              counts=walks["folder"].value_counts().to_dict())
    (store / "store_manifest.json").write_text(json.dumps(sm, indent=1))
    print(f"BUILT {name}: {len(walks):,} records, video {vbytes / 1e9:.2f} GB, {dt / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
