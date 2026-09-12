"""Shared constants, paths and helpers for the SpatialVID-HQ pipeline."""
from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

DATASET = "spatialvid-hq"
RAW_METADATA_CSV = "data/train/SpatialVID_HQ_metadata.csv"
RAW_SOURCE_CSV = "metadata_short_duration.csv"          # from SpatialVID/SpatialVID-RAW
GROUP_FMT = "group_{gid:04d}"

# Resolution presets: name -> (width, height). Names are the `res` axis of the registry.
RES = {"640x360": (640, 360), "456x256": (456, 256)}
RES_ALIASES = {"640": "640x360", "360p": "640x360", "456": "456x256", "256p": "456x256"}

# Encode settings settled by measurement (slipstream/experiments/spatialvid_inspection/DECISIONS.md):
# x265 crf 29 matches the source bitrate at 720p; keyframe every 1 s (+8.8 % bytes, 2.3x faster random windows).
X265_CRF = 29
X265_PRESET = "medium"
GOP_SECONDS = 1.0
FFMPEG_THREADS = 4

# Field layout of one record in the h265 store (see README.md "Data model").
FIELD_TYPES: dict[str, str] = {
    "video": "bytes",           # MP4 container, HEVC Main, yuv420p, hvc1 tag, faststart, no audio
    "clip_id": "str",
    "source_id": "str",         # YouTube id (SpatialVID-RAW)
    "group_id": "int",
    "width": "int",
    "height": "int",
    "fps": "float",
    "num_frames": "int",
    "duration_s": "float",
    "src_start_us": "int",      # clip position inside the source video
    "src_end_us": "int",
    "n_annot": "int",           # number of annotated (pose) frames
    "annot_frame_idx": "bytes", # np.save int32 (n_annot,)   video frame index of each pose row
    "poses": "bytes",           # np.save float32 (n_annot, 7) [tx ty tz qx qy qz qw], world->camera, OpenCV axes
    "intrinsics": "bytes",      # np.save float32 (n_annot, 4) normalized [fx fy cx cy]
    "instructions": "str",      # instructions.json verbatim
    "caption": "str",           # caption.json verbatim
    "scene_type": "str", "motion_tags": "str", "brightness": "str", "time_of_day": "str",
    "weather": "str", "crowd_density": "str",
    "aesthetic_score": "float", "luminance_score": "float", "motion_score": "float", "ocr_score": "float",
    "move_dist": "float", "dist_level": "int", "rot_angle": "float", "traj_turns": "float", "dynamic_ratio": "float",
}
CSV_TO_FIELD = {
    "sceneType": "scene_type", "motionTags": "motion_tags", "brightness": "brightness", "timeOfDay": "time_of_day",
    "weather": "weather", "crowdDensity": "crowd_density", "aesthetic score": "aesthetic_score",
    "luminance score": "luminance_score", "motion score": "motion_score", "ocr score": "ocr_score",
    "moveDist": "move_dist", "distLevel": "dist_level", "rotAngle": "rot_angle", "trajTurns": "traj_turns",
    "dynamicRatio": "dynamic_ratio",
}


@dataclass
class Layout:
    """Directory layout. `raw` is the read-only HF mirror; `out` is the working tree (local NVMe)."""
    raw: Path
    out: Path

    @property
    def index_dir(self) -> Path: return self.out / "index"
    @property
    def splits_dir(self) -> Path: return self.out / "splits"
    @property
    def shards_dir(self) -> Path: return self.out / "shards"
    @property
    def stores_dir(self) -> Path: return self.out / "stores"
    def video_tar(self, gid: int) -> Path: return self.raw / "videos" / f"{GROUP_FMT.format(gid=gid)}.tar.gz"
    def annotation_tar(self, gid: int) -> Path: return self.raw / "annotations" / f"{GROUP_FMT.format(gid=gid)}.tar.gz"
    def store_dir(self, res: str, fmt: str = "h265") -> Path: return self.stores_dir / f"{DATASET}-{fmt}-{res}"
    def shard_dir(self, res: str, gid: int) -> Path: return self.shards_dir / res / GROUP_FMT.format(gid=gid)


def resolve_res(name: str) -> str:
    name = RES_ALIASES.get(str(name), str(name))
    if name not in RES:
        raise ValueError(f"unknown res {name!r}; choose from {sorted(RES)} (aliases {sorted(RES_ALIASES)})")
    return name


def parse_groups(spec: str | None, available: list[int]) -> list[int]:
    if not spec or spec == "all":
        return list(available)
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return [g for g in available if g in out]


def iter_tar_members(path: Path, suffixes: tuple[str, ...] | None = None) -> Iterator[tuple[str, bytes]]:
    """Stream a .tar.gz sequentially (no random access), yielding (member_name, bytes) for regular files."""
    with open(path, "rb", buffering=0) as raw:
        buf = io.BufferedReader(raw, 16 << 20)
        with tarfile.open(fileobj=buf, mode="r|gz") as tf:
            for m in tf:
                if not m.isfile() or m.name.rsplit("/", 1)[-1].startswith("._"):   # skip macOS AppleDouble sidecars
                    continue
                if suffixes and not m.name.endswith(suffixes):
                    continue
                f = tf.extractfile(m)
                if f is not None:
                    yield m.name, f.read()


ANNOT_FILES = ("poses.npy", "intrinsics.npy", "indexes.txt", "instructions.json", "caption.json")


def read_group_annotations(tar_path: Path) -> dict[str, dict]:
    """Read all annotation files for a group (skipping dyn_masks.npz). Returns {clip_id: {name: parsed}}."""
    clips: dict[str, dict] = {}
    for name, data in iter_tar_members(tar_path, suffixes=ANNOT_FILES):
        parts = name.split("/")
        clip_id, fname = parts[-2], parts[-1]
        d = clips.setdefault(clip_id, {})
        if fname.endswith(".npy"):
            try:
                arr = np.load(io.BytesIO(data), allow_pickle=False)
            except ValueError:  # a few files are pickled object arrays; coerce or flag
                try:
                    arr = np.asarray(np.load(io.BytesIO(data), allow_pickle=True), dtype=np.float32)
                    d.setdefault("notes", []).append(f"{fname}: pickled, coerced")
                except Exception:  # noqa: BLE001
                    d.setdefault("notes", []).append(f"{fname}: unreadable"); continue
            d[fname[:-4]] = arr
        elif fname == "indexes.txt":
            arr = np.loadtxt(io.StringIO(data.decode()), dtype=np.int64).reshape(-1, 2)
            d["annot_row"], d["annot_frame_idx"] = arr[:, 0], arr[:, 1]
        else:
            d[fname[:-5]] = data.decode("utf-8", errors="replace")
    return clips


def npy_bytes(arr: np.ndarray) -> bytes:
    b = io.BytesIO(); np.save(b, np.ascontiguousarray(arr)); return b.getvalue()


def npy_from_bytes(data: bytes | memoryview | np.ndarray) -> np.ndarray:
    return np.load(io.BytesIO(bytes(data)), allow_pickle=False)


def hms_to_us(s: str) -> int:
    """'00:00:27.628' -> microseconds (SpatialVID-RAW timestamps)."""
    h, m, sec = s.split(":")
    return int(round((int(h) * 3600 + int(m) * 60 + float(sec)) * 1_000_000))


def ffprobe_stream(ffprobe: str, path: str | Path) -> dict:
    out = subprocess.run([ffprobe, "-v", "quiet", "-print_format", "json", "-show_streams", "-select_streams", "v:0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = st["avg_frame_rate"].split("/")
    return {"width": st["width"], "height": st["height"], "fps": int(num) / int(den),
            "nb_frames": int(st.get("nb_frames", 0)), "codec": st["codec_name"], "tag": st.get("codec_tag_string")}


def find_ffmpeg(explicit: str | None) -> tuple[str, str]:
    """Return (ffmpeg, ffprobe) paths. Accepts a binary path or a directory (e.g. a conda env's bin)."""
    cand = Path(explicit) if explicit else None
    if cand and cand.is_dir():
        return str(cand / "ffmpeg"), str(cand / "ffprobe")
    if cand:
        return str(cand), str(cand.with_name("ffprobe"))
    env = os.environ.get("SPATIALVID_FFMPEG")
    if env:
        return find_ffmpeg(env)
    return "ffmpeg", "ffprobe"
