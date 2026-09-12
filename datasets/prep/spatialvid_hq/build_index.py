"""Stage 1: build index/clips.parquet (one row per HQ clip) and index/annotations/group_XXXX.parquet.

Joins the HQ metadata CSV with the SpatialVID-RAW source table (YouTube id + timestamps) and streams
every annotation tar (skipping dyn_masks) to record per-clip pose/intrinsics/frame-index arrays,
instructions and captions. Nothing is decoded here.

    uv run python -m datasets.prep.spatialvid_hq.build_index --raw <hf mirror> --out <work dir> \
        [--source-csv <metadata_short_duration.csv>] [--groups 1-4] [--workers 8]

`--source-csv` defaults to downloading SpatialVID/SpatialVID-RAW/metadata_short_duration.csv (needs HF login).
Re-running skips groups whose annotation parquet already exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .common import GROUP_FMT, RAW_METADATA_CSV, RAW_SOURCE_CSV, CSV_TO_FIELD, Layout, hms_to_us, parse_groups, read_group_annotations


def load_metadata(lay: Layout) -> pd.DataFrame:
    df = pd.read_csv(lay.raw / RAW_METADATA_CSV)
    df = df.rename(columns={"group id": "group_id", "num frames": "num_frames", **CSV_TO_FIELD})
    df["clip_id"] = df["id"]
    df["duration_s"] = df["num_frames"] / df["fps"]
    for c in ("scene_type", "motion_tags", "brightness", "time_of_day", "weather", "crowd_density"):
        df[c] = df[c].fillna("").astype(str)
    df["scene_l1"] = df["scene_type"].str.split(";").str[0].str.strip()
    df["scene_l2"] = df["scene_type"].str.split(";").str[1].str.strip().fillna("")
    return df.drop(columns=["id", "video path", "annotation path", "resolution"])


def load_sources(path: Path) -> pd.DataFrame:
    src = pd.read_csv(path, usecols=["id", "YouTube id", "timestamp_start", "timestamp_end"])
    src = src.rename(columns={"id": "clip_id", "YouTube id": "source_id"})
    src["src_start_us"] = src.pop("timestamp_start").map(hms_to_us)
    src["src_end_us"] = src.pop("timestamp_end").map(hms_to_us)
    return src


def _group_annotations(args: tuple[str, str, int]) -> tuple[int, int]:
    tar_path, out_path, gid = args
    clips = read_group_annotations(Path(tar_path))
    rows = []
    for cid, d in clips.items():
        poses = d.get("poses"); intr = d.get("intrinsics"); fidx = d.get("annot_frame_idx")
        ok = poses is not None and intr is not None and fidx is not None and len(poses) == len(intr) == len(fidx)
        rows.append(dict(
            clip_id=cid, group_id=gid, n_annot=int(len(poses)) if ok else 0, annot_ok=bool(ok), notes=";".join(d.get("notes", [])),
            annot_frame_idx=fidx.astype(np.int32).tolist() if ok else [],
            poses=poses.astype(np.float32).reshape(-1).tolist() if ok else [],          # flat, reshape (-1, 7)
            intrinsics=intr.astype(np.float32).reshape(-1).tolist() if ok else [],      # flat, reshape (-1, 4)
            instructions=d.get("instructions", ""), caption=d.get("caption", ""),
        ))
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return gid, len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source-csv", type=Path, default=None)
    ap.add_argument("--groups", default="all"); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    lay = Layout(a.raw, a.out); lay.index_dir.mkdir(parents=True, exist_ok=True); (lay.index_dir / "annotations").mkdir(exist_ok=True)

    t0 = time.time(); meta = load_metadata(lay); print(f"metadata: {len(meta):,} clips, {meta.group_id.nunique()} groups")
    if a.source_csv is None:
        from huggingface_hub import hf_hub_download
        a.source_csv = Path(hf_hub_download("SpatialVID/SpatialVID-RAW", RAW_SOURCE_CSV, repo_type="dataset", local_dir=str(lay.index_dir / "raw_source")))
    src = load_sources(a.source_csv)
    meta = meta.merge(src, on="clip_id", how="left")
    miss = meta["source_id"].isna().sum(); print(f"source join: {len(meta) - miss:,} matched, {miss} missing, {meta.source_id.nunique():,} sources")

    groups = parse_groups(a.groups, sorted(meta.group_id.unique().tolist()))
    jobs = [(str(lay.annotation_tar(g)), str(lay.index_dir / "annotations" / f"{GROUP_FMT.format(gid=g)}.parquet"), g)
            for g in groups if not (lay.index_dir / "annotations" / f"{GROUP_FMT.format(gid=g)}.parquet").exists()]
    print(f"annotation tars to read: {len(jobs)} of {len(groups)} groups")
    with ProcessPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(_group_annotations, j) for j in jobs]):
            gid, n = fut.result(); print(f"  group {gid:04d}: {n} clips  [{time.time()-t0:.0f}s]", flush=True)

    ann = pd.concat([pd.read_parquet(lay.index_dir / "annotations" / f"{GROUP_FMT.format(gid=g)}.parquet",
                                     columns=["clip_id", "n_annot", "annot_ok", "notes"]) for g in groups], ignore_index=True)
    clips = meta[meta.group_id.isin(groups)].merge(ann, on="clip_id", how="left")
    clips["annot_ok"] = clips["annot_ok"].fillna(False).astype(bool); clips["n_annot"] = clips["n_annot"].fillna(0).astype(int)
    clips = clips.sort_values(["group_id", "clip_id"]).reset_index(drop=True)
    clips.to_parquet(lay.index_dir / "clips.parquet", index=False)
    summary = dict(clips=len(clips), groups=len(groups), sources=int(clips.source_id.nunique()), annot_ok=int(clips.annot_ok.sum()), annot_notes=int((clips.notes.fillna("") != "").sum()),
                   hours=float(clips.duration_s.sum() / 3600), built=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (lay.index_dir / "summary.json").write_text(json.dumps(summary, indent=1)); print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
