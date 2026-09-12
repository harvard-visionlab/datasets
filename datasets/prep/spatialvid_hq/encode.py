"""Stage 3: re-encode one group of source clips to each target resolution and write slipstream shards.

For every `videos/group_XXXX.tar.gz` (streamed sequentially from the raw mirror), each clip is decoded once by
ffmpeg and encoded to all requested resolutions in one pass (filter split), HEVC x265 crf 29, keyframe every
1 s, timestamps preserved (frame count asserted equal to the source). Records are written in tar order to a
slipstream-compatible shard per resolution: shards/<res>/group_XXXX/{<field>.bin,.meta.npy,.offsets.npy,.npy,
_shard_manifest.json, records.parquet}. Stage 4 (merge) concatenates the shards into one store per resolution.

    uv run python -m datasets.prep.spatialvid_hq.encode --raw <hf mirror> --out <work dir> \
        [--res 640x360,456x256] [--groups 1-74] [--limit N] [--workers 16] [--ffmpeg <bin or conda env bin dir>]

Requires stage 1 output (index/). Idempotent: a group with a finished shard for every requested resolution is
skipped, so the full run can be stopped and resumed; a partially written group is redone from scratch.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (FFMPEG_THREADS, FIELD_TYPES, GOP_SECONDS, GROUP_FMT, RES, X265_CRF, X265_PRESET, Layout,
                     ffprobe_stream, find_ffmpeg, iter_tar_members, npy_bytes, parse_groups, resolve_res)

# ----------------------------------------------------------------------------- shard writer (slipstream layout)

def _variable_meta_dtype():
    from slipstream.cache import VARIABLE_METADATA_DTYPE
    return VARIABLE_METADATA_DTYPE


class ShardWriter:
    """Writes records sequentially in slipstream's on-disk field formats; trims to the number actually written."""

    def __init__(self, shard_dir: Path, field_types: dict[str, str], capacity: int, image_format: str = "mp4"):
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True)
        self.dir, self.ft, self.cap, self.image_format = shard_dir, field_types, capacity, image_format
        self.n = 0
        self.files, self.meta, self.offs, self.num, self.ptr, self.max_size = {}, {}, {}, {}, {}, {}
        vm = _variable_meta_dtype()
        for f, t in field_types.items():
            if t in ("bytes", "ImageBytes"):
                self.files[f] = open(shard_dir / f"{f}.bin", "wb"); self.meta[f] = np.zeros(capacity, dtype=vm); self.ptr[f] = 0; self.max_size[f] = 0
            elif t == "str":
                self.files[f] = open(shard_dir / f"{f}.bin", "wb"); self.offs[f] = np.zeros((capacity, 2), dtype=np.uint64); self.ptr[f] = 0
            else:
                self.num[f] = np.zeros(capacity, dtype=np.float64 if t in ("float", "float32", "float64") else np.int64)
        self.records: list[dict] = []

    def add(self, sample: dict, record: dict) -> None:
        i = self.n
        if i >= self.cap:
            raise RuntimeError("shard capacity exceeded")
        for f, t in self.ft.items():
            v = sample[f]
            if t in ("bytes", "ImageBytes"):
                b = bytes(v); self.files[f].write(b); m = self.meta[f][i]
                m["data_ptr"], m["data_size"], m["height"], m["width"] = self.ptr[f], len(b), 0, 0
                self.ptr[f] += len(b); self.max_size[f] = max(self.max_size[f], len(b))
            elif t == "str":
                b = str(v).encode("utf-8"); self.files[f].write(b); self.offs[f][i] = (self.ptr[f], len(b)); self.ptr[f] += len(b)
            else:
                self.num[f][i] = v
        self.records.append(dict(local_idx=i, **record)); self.n += 1

    def finalize(self, extra: dict | None = None) -> dict:
        n = self.n; fields = {}
        for f, t in self.ft.items():
            if t in ("bytes", "ImageBytes"):
                self.files[f].close(); np.save(self.dir / f"{f}.meta.npy", self.meta[f][:n])
                fields[f] = {"type": t, "num_samples": n, "max_size": int(self.max_size[f] * 1.2), "image_format": self.image_format}
            elif t == "str":
                self.files[f].close(); np.save(self.dir / f"{f}.offsets.npy", self.offs[f][:n]); fields[f] = {"type": "str", "num_samples": n}
            else:
                np.save(self.dir / f"{f}.npy", self.num[f][:n]); fields[f] = {"type": t, "num_samples": n}
        from slipstream.cache import _get_expected_files
        file_sizes = {fn: os.path.getsize(self.dir / fn) for f, t in self.ft.items() for fn in _get_expected_files(f, t) if (self.dir / fn).exists()}
        manifest = {"worker_id": 0, "start_idx": 0, "end_idx": n, "num_samples": n, "fields": fields, "file_sizes": file_sizes, **(extra or {})}
        pd.DataFrame(self.records).to_parquet(self.dir / "records.parquet", index=False)
        (self.dir / "_shard_manifest.json").write_text(json.dumps(manifest, indent=1))
        return manifest


# ----------------------------------------------------------------------------- per-clip worker (process pool)

def encode_clip(args: tuple) -> dict:
    """Decode once, encode to every resolution. Returns {'ok', 'src': probe, 'out': {res: {'bytes','probe'}}, 'error'}."""
    clip_id, src_bytes, res_names, ffmpeg, ffprobe, tmp_root = args
    res_names = list(res_names)
    with tempfile.TemporaryDirectory(dir=tmp_root) as td:
        src = Path(td) / "src.mp4"; src.write_bytes(src_bytes)
        try:
            sp = ffprobe_stream(ffprobe, src)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"probe: {e}"}
        k = max(1, int(round(sp["fps"] * GOP_SECONDS)))
        n = len(res_names)
        fc = f"[0:v]split={n}" + "".join(f"[s{i}]" for i in range(n)) + ";" + ";".join(
            f"[s{i}]scale={RES[r][0]}:{RES[r][1]}:flags=lanczos[o{i}]" for i, r in enumerate(res_names))
        cmd = [ffmpeg, "-v", "error", "-y", "-threads", str(FFMPEG_THREADS), "-i", str(src), "-filter_complex", fc]
        outs = []
        for i, r in enumerate(res_names):
            o = Path(td) / f"{r}.mp4"; outs.append(o)
            cmd += ["-map", f"[o{i}]", "-c:v", "libx265", "-crf", str(X265_CRF), "-preset", X265_PRESET,
                    "-x265-params", f"keyint={k}:min-keyint={k}:scenecut=0:log-level=error", "-tag:v", "hvc1",
                    "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-an", "-movflags", "+faststart", str(o)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            return {"ok": False, "error": f"ffmpeg: {p.stderr.strip()[-300:]}", "src": sp}
        out = {}
        for r, o in zip(res_names, outs):
            op = ffprobe_stream(ffprobe, o)
            if op["nb_frames"] != sp["nb_frames"]:
                return {"ok": False, "error": f"frame count {op['nb_frames']} != source {sp['nb_frames']} at {r}", "src": sp}
            out[r] = {"bytes": o.read_bytes(), "probe": op}
        return {"ok": True, "src": sp, "out": out}


# ----------------------------------------------------------------------------- group driver

def build_sample(res: str, cid: str, clip: pd.Series, ann: pd.Series, result: dict) -> dict:
    op = result["out"][res]["probe"]; sp = result["src"]
    poses = np.asarray(ann["poses"], np.float32).reshape(-1, 7); intr = np.asarray(ann["intrinsics"], np.float32).reshape(-1, 4)
    fidx = np.asarray(ann["annot_frame_idx"], np.int32)
    s = dict(video=result["out"][res]["bytes"], clip_id=cid, source_id=str(clip.get("source_id", "")),
             group_id=int(clip["group_id"]), width=op["width"], height=op["height"], fps=float(sp["fps"]),
             num_frames=int(op["nb_frames"]), duration_s=float(op["nb_frames"] / sp["fps"]),
             src_start_us=int(clip.get("src_start_us", -1) if pd.notna(clip.get("src_start_us", np.nan)) else -1),
             src_end_us=int(clip.get("src_end_us", -1) if pd.notna(clip.get("src_end_us", np.nan)) else -1),
             n_annot=int(len(poses)), annot_frame_idx=npy_bytes(fidx), poses=npy_bytes(poses), intrinsics=npy_bytes(intr),
             instructions=ann.get("instructions", "") or "", caption=ann.get("caption", "") or "")
    for f in ("scene_type", "motion_tags", "brightness", "time_of_day", "weather", "crowd_density"):
        s[f] = str(clip.get(f, "") or "")
    for f in ("aesthetic_score", "luminance_score", "motion_score", "ocr_score", "move_dist", "rot_angle", "traj_turns", "dynamic_ratio"):
        s[f] = float(clip.get(f, np.nan))
    s["dist_level"] = int(clip.get("dist_level", -1))
    assert set(s) == set(FIELD_TYPES), set(FIELD_TYPES) ^ set(s)
    return s


def encode_group(lay: Layout, gid: int, res_names: list[str], workers: int, limit: int | None, ffmpeg: str, ffprobe: str,
                 tmp_root: str, log_every: int = 100) -> dict:
    gname = GROUP_FMT.format(gid=gid)
    done = all((lay.shard_dir(r, gid) / "_shard_manifest.json").exists() for r in res_names)
    if done:
        print(f"[{gname}] shards exist for {res_names}, skipping"); return {"group": gid, "skipped": True}
    clips = pd.read_parquet(lay.index_dir / "clips.parquet"); clips = clips[clips.group_id == gid].set_index("clip_id")
    ann = pd.read_parquet(lay.index_dir / "annotations" / f"{gname}.parquet").set_index("clip_id")
    eligible = clips.index[clips["annot_ok"]].intersection(ann.index)
    cap = min(len(eligible), limit) if limit else len(eligible)
    writers = {r: ShardWriter(lay.shard_dir(r, gid), FIELD_TYPES, cap) for r in res_names}
    stats = dict(group=gid, eligible=int(len(eligible)), submitted=0, written=0, failed=0, not_in_index=0, bytes={r: 0 for r in res_names}, src_bytes=0)
    t0 = time.time(); errors: list[dict] = []
    pending: deque = deque()
    window = 2 * workers

    def drain(block: bool) -> None:
        while pending and (block or pending[0][0].done()):
            fut, cid = pending.popleft(); res = fut.result()
            if not res["ok"]:
                stats["failed"] += 1; errors.append({"clip_id": cid, "error": res["error"]}); continue
            for r in res_names:
                writers[r].add(build_sample(r, cid, clips.loc[cid], ann.loc[cid], res),
                               dict(clip_id=cid, group_id=gid, num_frames=res["out"][r]["probe"]["nb_frames"], video_bytes=len(res["out"][r]["bytes"])))
                stats["bytes"][r] += len(res["out"][r]["bytes"])
            stats["written"] += 1
            if stats["written"] % log_every == 0:
                el = time.time() - t0; rate = stats["written"] / el
                print(f"[{gname}] {stats['written']}/{cap} clips, {rate:.2f} clips/s, eta {(cap - stats['written']) / max(rate, 1e-6) / 60:.1f} min, "
                      + ", ".join(f"{r} {stats['bytes'][r] / max(stats['src_bytes'], 1) * 100:.0f}%" for r in res_names), flush=True)

    with ProcessPoolExecutor(workers) as ex:
        for name, data in iter_tar_members(lay.video_tar(gid), suffixes=(".mp4",)):
            cid = Path(name).stem
            if cid not in clips.index or cid not in ann.index or not clips.loc[cid, "annot_ok"]:
                stats["not_in_index"] += 1; continue
            if limit and stats["submitted"] >= limit:
                break
            while len(pending) >= window:
                drain(block=False); time.sleep(0.01) if len(pending) >= window else None
            pending.append((ex.submit(encode_clip, (cid, data, tuple(res_names), ffmpeg, ffprobe, tmp_root)), cid))
            stats["submitted"] += 1; stats["src_bytes"] += len(data)
        drain(block=True)
    extra = dict(group=gname, encode=dict(codec="libx265", crf=X265_CRF, preset=X265_PRESET, gop_seconds=GOP_SECONDS), errors=errors)
    for r in res_names:
        writers[r].finalize(extra={**extra, "res": r})
    stats["seconds"] = round(time.time() - t0, 1); stats["errors"] = errors[:20]
    print(f"[{gname}] done: {stats['written']} written, {stats['failed']} failed, {stats['not_in_index']} skipped (not in index / no annotations), "
          f"{stats['seconds']} s; size vs source: " + ", ".join(f"{r} {stats['bytes'][r] / max(stats['src_bytes'], 1) * 100:.1f}%" for r in res_names), flush=True)
    (lay.shards_dir / f"{gname}.stats.json").write_text(json.dumps(stats, indent=1, default=str))
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--res", default=",".join(RES)); ap.add_argument("--groups", default="all")
    ap.add_argument("--limit", type=int, default=None, help="max clips per group (test runs)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // FFMPEG_THREADS))
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg binary or a bin/ directory (default: PATH or $SPATIALVID_FFMPEG)")
    ap.add_argument("--tmp", default=None, help="fast local scratch for per-clip temp files (default: system tmp)")
    a = ap.parse_args(argv)
    lay = Layout(a.raw, a.out); res_names = [resolve_res(r) for r in a.res.split(",")]
    ffmpeg, ffprobe = find_ffmpeg(a.ffmpeg)
    enc = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "libx265" not in enc:
        print(f"{ffmpeg} lacks libx265 (conda: conda create -p <env> -c conda-forge 'ffmpeg>=7,<8')", file=sys.stderr); return 2
    clips = pd.read_parquet(lay.index_dir / "clips.parquet", columns=["group_id"])
    groups = parse_groups(a.groups, sorted(clips.group_id.unique().tolist()))
    print(f"encode {len(groups)} groups -> {res_names}, {a.workers} workers x {FFMPEG_THREADS} ffmpeg threads, ffmpeg={ffmpeg}")
    lay.shards_dir.mkdir(parents=True, exist_ok=True)
    for gid in groups:
        encode_group(lay, gid, res_names, a.workers, a.limit, ffmpeg, ffprobe, a.tmp or tempfile.gettempdir())
    return 0


if __name__ == "__main__":
    sys.exit(main())
