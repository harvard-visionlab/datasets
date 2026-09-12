"""Read-side helpers for SpatialVID-style slipstream video stores (h265 format).

This is the interim access layer used by notebooks and tests until slipstream grows a native window sampler
for `bytes` video records. It opens a store built by `datasets.prep.spatialvid_hq`, returns fully decoded
per-record metadata, decodes frame windows with torchcodec straight from the mmap'd bytes, and interpolates
poses to arbitrary frames.

    store = VideoStore("/path/to/stores/spatialvid-hq-h265-640x360")
    rec = store.record(0)                       # dict: clip_id, fps, poses (n,7), annot_frame_idx (n,), caption, ...
    frames = store.frames(0, start=30, count=8, stride=2)      # uint8 [T, 3, H, W], with .pts_seconds
    poses = store.poses_at(0, frame_idx=[30, 32, 34])          # (T, 7) interpolated world->camera poses
    rel = relative_motion(poses[0], poses[1])                  # (6,) dpos in camera-t axes + rotvec (rad)
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

POSE_COLS = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]

_NV_PRELOADED = False


def preload_nvidia_libs() -> list[str]:
    """Make pip-installed NVIDIA runtime libs (NPP, cuda runtime) visible to torchcodec's CUDA core library.

    torchcodec dlopens libnppicc/libnppig by soname; pip wheels put them under site-packages/nvidia/*/lib, which is
    not on the default search path. Loading them RTLD_GLOBAL first avoids needing LD_LIBRARY_PATH. No-op off Linux.
    FFmpeg shared libraries themselves must come from the system (e.g. `apt-get install ffmpeg`).
    """
    global _NV_PRELOADED
    if _NV_PRELOADED or sys.platform != "linux":
        return []
    import ctypes, glob, site
    loaded = []
    roots = {p for p in site.getsitepackages() + [site.getusersitepackages()] if p}
    for root in roots:
        for pattern in ("nvidia/cuda_runtime/lib/libcudart.so.*", "nvidia/npp/lib/libnppc.so.*", "nvidia/npp/lib/libnppicc.so.*", "nvidia/npp/lib/libnppig.so.*"):
            for lib in sorted(glob.glob(str(Path(root) / pattern))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL); loaded.append(lib)
                except OSError:
                    pass
    _NV_PRELOADED = True
    return loaded


def import_torchcodec():
    """Import torchcodec after preloading NVIDIA libs; raises with a setup hint if FFmpeg libs are missing."""
    preload_nvidia_libs()
    try:
        from torchcodec.decoders import VideoDecoder
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "torchcodec could not load its FFmpeg-backed core. Install FFmpeg shared libraries on the system "
            "(Debian/Ubuntu: apt-get install ffmpeg; macOS: brew install ffmpeg and export DYLD_LIBRARY_PATH=/opt/homebrew/lib)."
        ) from e
    return VideoDecoder


def _npy(data) -> np.ndarray:
    return np.load(io.BytesIO(bytes(data)), allow_pickle=False)


class VideoStore:
    """Thin wrapper over slipstream's OptimizedCache for h265 video stores."""

    def __init__(self, path: str | Path, device: str = "cpu"):
        from slipstream.cache import OptimizedCache
        self.path = Path(path); self.device = device
        self.cache = OptimizedCache.load(self.path, verbose=False)
        self.field_types = self.cache.field_types
        rp = self.path / "records.parquet"
        self.records = None
        if rp.exists():
            import pandas as pd
            self.records = pd.read_parquet(rp)
        self._decoders: dict[int, Any] = {}

    def __len__(self) -> int:
        return self.cache.num_samples

    def raw(self, idx: int, field: str):
        out = self.cache.fields[field].load_batch(np.array([idx], dtype=np.int64), parallel=False)
        data = out["data"]
        if self.field_types[field] in ("bytes", "ImageBytes"):
            return bytes(data[0][: int(out["sizes"][0])])
        return data[0]

    def record(self, idx: int, with_video: bool = False) -> dict[str, Any]:
        rec: dict[str, Any] = {}
        for f, t in self.field_types.items():
            if f == "video" and not with_video:
                continue
            v = self.raw(idx, f)
            if f in ("poses", "intrinsics", "annot_frame_idx"):
                v = _npy(v)
            elif f in ("caption", "instructions") and v:
                try: v = json.loads(v)
                except Exception: pass
            elif t in ("int", "float") and hasattr(v, "item"):
                v = v.item()
            rec[f] = v
        return rec

    def video_bytes(self, idx: int) -> bytes:
        return self.raw(idx, "video")

    def decoder(self, idx: int):
        """torchcodec VideoDecoder over the record's bytes (cached per record; call close_decoders() to free)."""
        d = self._decoders.get(idx)
        if d is None:
            VideoDecoder = import_torchcodec()
            d = VideoDecoder(self.video_bytes(idx), device=self.device, seek_mode="exact")
            self._decoders[idx] = d
        return d

    def close_decoders(self) -> None:
        self._decoders.clear()

    def frames(self, idx: int, start: int, count: int, stride: int = 1):
        """Decode `count` frames starting at frame `start` with `stride` -> torchcodec FrameBatch (uint8 [T,3,H,W])."""
        d = self.decoder(idx)
        idxs = list(range(start, start + count * stride, stride))
        return d.get_frames_at(indices=idxs)

    def frames_at_seconds(self, idx: int, seconds):
        return self.decoder(idx).get_frames_played_at(seconds=list(seconds))

    def poses_at(self, idx: int, frame_idx, rec: dict | None = None) -> np.ndarray:
        """Interpolate world->camera poses to arbitrary video frame indices (linear t, slerp q)."""
        rec = rec or self.record(idx)
        return interpolate_poses(rec["poses"], rec["annot_frame_idx"], np.asarray(frame_idx))


# ---------------------------------------------------------------- pose math (numpy only, xyzw quaternions)

def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True); q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1); dot = np.abs(dot).clip(-1, 1)
    theta = np.arccos(dot); s = np.sin(theta)
    t = t[..., None]
    lin = (1 - t) * q0 + t * q1
    w0 = np.where(s > 1e-6, np.sin((1 - t) * theta) / np.where(s > 1e-6, s, 1), 1 - t)
    w1 = np.where(s > 1e-6, np.sin(t * theta) / np.where(s > 1e-6, s, 1), t)
    out = np.where(s > 1e-6, w0 * q0 + w1 * q1, lin)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def interpolate_poses(poses: np.ndarray, annot_frame_idx: np.ndarray, frame_idx: np.ndarray) -> np.ndarray:
    """poses (n,7) at annot_frame_idx (n,) -> (m,7) at frame_idx; clamps outside the annotated range."""
    poses = np.asarray(poses, np.float64); af = np.asarray(annot_frame_idx, np.float64); fi = np.asarray(frame_idx, np.float64)
    fi = fi.clip(af[0], af[-1])
    j = np.searchsorted(af, fi, side="right").clip(1, len(af) - 1); i = j - 1
    t = np.where(af[j] > af[i], (fi - af[i]) / np.maximum(af[j] - af[i], 1e-9), 0.0)
    pos = (1 - t)[:, None] * poses[i, :3] + t[:, None] * poses[j, :3]
    q = quat_slerp(poses[i, 3:], poses[j, 3:], t)
    return np.concatenate([pos, q], axis=1).astype(np.float32)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return axis * angle


def camera_center(pose: np.ndarray) -> np.ndarray:
    """World->camera pose -> camera center in world coordinates."""
    R = quat_to_rotmat(pose[3:]); return -R.T @ pose[:3]


def relative_motion(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Motion from frame 0 to frame 1 expressed in camera-0 axes (OpenCV: x right, y down, z forward).

    Returns (6,): [dx, dy, dz, rx, ry, rz] with translation in the (non-metric) pose units and rotation as a
    rotation vector in radians. Convention matches the authors' get_instructions.py.
    """
    R0, R1 = quat_to_rotmat(p0[3:]), quat_to_rotmat(p1[3:])
    dpos = R0 @ (camera_center(p1) - camera_center(p0))
    drot = rotmat_to_rotvec(R1 @ R0.T)
    return np.concatenate([dpos, drot]).astype(np.float32)
