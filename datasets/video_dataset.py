"""Video datasets in the registry: one slipstream store per (fmt, res, fps), splits and subsets as parquet index sets.

    from visionlab.datasets import load
    ds = load("spatialvid-hq", split="train", subset="person_carried_v0", rate_hz=15)
    ds.cache              # slipstream OptimizedCache of the chosen store (h265 bytes per clip)
    ds.clips              # DataFrame of the selected clips (subset ∩ split ∩ where ∩ channel_cap) with record_idx

`subset` defaults to the config's `default_subset` (spatialvid-hq: person_carried_v0, the training population); the
split table itself labels *every* store clip, so `subset="all"` gives the whole store's split members (e.g. all 16,932
v3 val clips rather than the population's 15,029).
    loader = SlipstreamLoader(ds, indices=ds.window_sampler(8.0).recs, pipelines={"video": [DecodeVideoWindow(...)]},
                              after_batch_transforms=[ds.ego_motion_transform()], ...)
    batch["video"], batch["poses"], batch["ego"]                        # [B,T,3,H,W] uint8, [B,T,7], [B,T-1,6]
    poses = ds.poses_at(batch["video_rec"], batch["video_t_sec"])      # the same poses, by hand

Store choice: `fps=` picks a store exactly (`fps="native"` = the un-decimated store); otherwise `rate_hz` (default
`default_rate_hz`) picks the sparsest store whose fps is an integer multiple of the rate (30 fps for 10/15/30 Hz, 15 fps
for 5/15 Hz, native = fallback). Stores listed in the config
but not built yet are skipped with a warning. Local resolution order: `$SLIPSTREAM_CACHE_DIR/<store>` (a node-local
copy), `$VISIONLAB_DATASETS_ROOT/<tree>`, the lab's QNAP mounts (QNAP_ROOTS), then a download from S3 into
`$SLIPSTREAM_CACHE_DIR/<store>` (only the store a job uses).
"""
from __future__ import annotations

import os
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

StoreKey = tuple[str, str, "int | None"]     # (fmt, res, fps); fps None = native source frame rate

QNAP_ROOTS = (
    "~/work/DataExactitudeFlash/DataSets/VideoDatasets",              # machina container
    "~/work/DataRemote/qnap/exactitude/Flash/DataSets/VideoDatasets",  # fleet containers
    "/mnt/QNAP/exactitude/Flash/DataSets/VideoDatasets",              # hosts
)
ROOT_ENV = "VISIONLAB_DATASETS_ROOT"
_WARNED: set[str] = set()


# ----------------------------------------------------------------------------- store selection

def store_name(remote: str) -> str:
    return remote.rstrip("/").rsplit("/", 1)[-1]


def rank_stores(stores: dict[StoreKey, str], fmt: str, res: str, rate_hz: float | None, fps: int | None) -> list[StoreKey]:
    """Candidate store keys in preference order (best first).

    fps given -> exactly that store. Else every store whose fps is an integer multiple of `rate_hz`, sparsest first,
    with the native store (fps None) last; `rate_hz` None -> native first, then densest.
    """
    cands = [k for k in stores if k[0] == fmt and k[1] == res]
    if not cands:
        raise KeyError(f"no store for fmt={fmt!r} res={res!r}; available: {sorted(set((k[0], k[1]) for k in stores))}")
    if fps is not None:
        if (fmt, res, fps) not in stores:
            raise KeyError(f"no store for fmt={fmt!r} res={res!r} fps={fps}; available fps: {[k[2] for k in cands]}")
        return [(fmt, res, fps)]
    if rate_hz is None:
        return sorted(cands, key=lambda k: (k[2] is not None, -(k[2] or 0)))
    ok = []
    for k in cands:
        if k[2] is None:
            ok.append((float("inf"), k)); continue
        ratio = k[2] / rate_hz
        if ratio >= 1 - 1e-9 and abs(ratio - round(ratio)) < 1e-6:
            ok.append((k[2], k))
    if not ok:
        raise ValueError(f"no store whose fps is a multiple of rate_hz={rate_hz}; available fps: {[k[2] for k in cands]}")
    return [k for _, k in sorted(ok, key=lambda x: x[0])]


# ----------------------------------------------------------------------------- local / remote resolution

def local_roots(tree: str) -> list[Path]:
    roots = []
    env = os.environ.get(ROOT_ENV)
    if env:
        roots.append(Path(env).expanduser() / tree)
    roots += [Path(r).expanduser() / tree for r in QNAP_ROOTS]
    return [r for r in roots if r.is_dir()]


def _s5cmd_cp(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(["s5cmd", "cp", remote, str(local)], capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("s5cmd is required to fetch dataset index files from S3 (pip install s5cmd)") from e
    return r.returncode == 0 and local.exists()


def resolve_file(rel: str, remote: str, tree: str, cache_base: Path) -> Path:
    """A small index file (split / subset parquet): shared tree if mounted, else download once into the cache dir."""
    for root in local_roots(tree):
        p = root / rel
        if p.exists():
            return p
    p = cache_base / tree / rel
    if not p.exists() and not _s5cmd_cp(remote, p):
        raise FileNotFoundError(f"{rel} not found locally and could not be fetched from {remote}")
    return p


def resolve_store(remote: str, tree: str, cache_base: Path, download: bool = True) -> Path | None:
    """Store directory: shared tree if mounted, else `$SLIPSTREAM_CACHE_DIR/<store>` (downloaded from S3 when missing).
    Returns None when the store exists nowhere (not built / not synced yet)."""
    from slipstream.cache import MANIFEST_FILE
    name = store_name(remote)
    p = cache_base / name                      # a node-local copy beats the network share (CIFS drops page cache per process)
    if (p / MANIFEST_FILE).exists():
        return p
    for root in local_roots(tree):
        q = root / "stores" / name
        if (q / MANIFEST_FILE).exists():
            return q
    if not download:
        return None
    from slipstream.s3_sync import download_s3_cache, s3_path_exists
    if not s3_path_exists(remote.rstrip("/") + "/" + MANIFEST_FILE):
        return None
    print(f"Downloading video store {name} from S3 ...")
    if not download_s3_cache(remote, p):
        raise RuntimeError(f"Failed to download {remote}")
    return p


# ----------------------------------------------------------------------------- window sampler

@dataclass
class WindowSampler:
    """Seeded (record, t0) anchors: `anchors_per_clip` uniform window starts per clip that fits the window."""
    recs: np.ndarray            # eligible record indices
    max_t0: np.ndarray          # per record: latest allowed window start (s)
    window_s: float
    anchors_per_clip: int = 1
    seed: int = 0

    def __len__(self) -> int:
        return len(self.recs) * self.anchors_per_clip

    def sample(self, epoch: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng([self.seed, epoch])
        recs = np.repeat(self.recs, self.anchors_per_clip); mx = np.repeat(self.max_t0, self.anchors_per_clip)
        return recs.astype(np.int64), (rng.random(len(recs)) * mx).astype(np.float32)


# ----------------------------------------------------------------------------- vectorised .npy parsing

_NPY_PREFIX = {np.dtype("<f4"): b"{'descr': '<f4', 'fortran_order': False, 'shape': (",
               np.dtype("<i4"): b"{'descr': '<i4', 'fortran_order': False, 'shape': ("}


def _npy_rows(data: np.ndarray, sizes: np.ndarray, dtype: np.dtype, row: int) -> tuple[np.ndarray, np.ndarray]:
    """B `.npy` blobs of dtype `dtype` and trailing dim `row`, packed in a (B, W) uint8 buffer with `sizes` -> values
    (B, N, row) [padding undefined] and counts (B,). Vectorised when every blob has the same v1.0 header length and
    the expected descr (np.save of small C arrays always does); otherwise falls back to np.load per blob."""
    data = np.asarray(data); sizes = np.asarray(sizes, dtype=np.int64); B = len(sizes)
    prefix = np.frombuffer(_NPY_PREFIX[dtype], dtype=np.uint8)
    hl = data[:, 8].astype(np.int64) | (data[:, 9].astype(np.int64) << 8)             # v1.0 HEADER_LEN (little-endian)
    ok = (B > 0 and (data[:, :6] == np.frombuffer(b"\x93NUMPY", np.uint8)).all() and (data[:, 6] == 1).all()
          and (hl == hl[0]).all() and (data[:, 10:10 + len(prefix)] == prefix).all())
    if ok:
        H = 10 + int(hl[0]); item = dtype.itemsize * row
        n = (sizes - H) // item
        if ((sizes - H) % item == 0).all() and H + int(n.max()) * item <= data.shape[1]:
            vals = np.ascontiguousarray(data[:, H:H + int(n.max()) * item]).view(dtype).reshape(B, -1, row)
            return vals, n
    from .video import _npy
    arrs = [_npy(data[i][: int(sizes[i])]).reshape(-1, row) for i in range(B)]
    n = np.fromiter((len(a) for a in arrs), dtype=np.int64, count=B)
    vals = np.zeros((B, int(n.max()), row), dtype=dtype)
    for i, a in enumerate(arrs):
        vals[i, : n[i]] = a
    return vals, n


# ----------------------------------------------------------------------------- the dataset object

@dataclass
class VideoDataset:
    name: str
    store_key: StoreKey
    store_dir: Path
    cache: Any                          # slipstream OptimizedCache
    clips: Any                          # pandas DataFrame
    split: str
    subset: str | None
    rate_hz: float | None
    stats: dict | None = None
    _annot_cache: dict = field(default_factory=dict, repr=False)

    # --- basics
    @property
    def indices(self) -> np.ndarray:
        return self.clips["record_idx"].to_numpy(np.int64)

    def __len__(self) -> int:
        return len(self.clips)

    @property
    def fps(self) -> int | None:
        return self.store_key[2]

    @property
    def cache_path(self) -> Path:
        """Store directory, so `SlipstreamLoader(ds, ...)` accepts the dataset directly (the loader duck-types on it)."""
        return self.store_dir

    def video_store(self, device: str = "cpu"):
        from .video import VideoStore
        return VideoStore(self.store_dir, device=device)

    def record(self, record_idx: int, with_video: bool = False) -> dict:
        return self.video_store().record(int(record_idx), with_video=with_video)

    # --- windows
    def window_sampler(self, window_s: float, anchors_per_clip: int = 1, seed: int = 0, end_margin_frames: int = 3) -> WindowSampler:
        """Clips shorter than the window are dropped; t0 ∈ [0, duration − window − margin]."""
        dur = self.clips["duration_s"].to_numpy(np.float64); fps = self.clips["fps"].to_numpy(np.float64)
        max_t0 = dur - window_s - end_margin_frames / fps
        ok = max_t0 > 0
        return WindowSampler(self.indices[ok], max_t0[ok], window_s, anchors_per_clip, seed)

    # --- poses
    def _annot(self, rec: int) -> tuple[np.ndarray, np.ndarray]:
        """(poses (n,7), annotation times (n,) in seconds) of one record, cached."""
        a = self._annot_cache.get(rec)
        if a is None:
            from .video import _npy
            f = self.cache.fields
            def raw(name):
                out = f[name].load_batch(np.array([rec], dtype=np.int64), parallel=False)
                return bytes(out["data"][0][: int(out["sizes"][0])])
            poses = _npy(raw("poses")).astype(np.float32).reshape(-1, 7)
            fidx = _npy(raw("annot_frame_idx")).astype(np.float64)
            fps_field = "src_fps" if "src_fps" in f else "fps"
            src_fps = float(f[fps_field].load_batch(np.array([rec], dtype=np.int64), parallel=False)["data"][0])
            a = (poses, fidx / src_fps)
            if len(self._annot_cache) > 8192:
                self._annot_cache.clear()
            self._annot_cache[rec] = a
        return a

    def _annot_batch(self, recs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Annotations of B records, padded to the longest: poses (B, N, 7) float64, times (B, N) seconds (padding
        = +inf), n_annot (B,). One store read per field for the whole batch and no per-record Python: the .npy blobs
        share a fixed header, so the values are a strided view of the read buffer (`_npy_rows`)."""
        f = self.cache.fields
        recs = np.asarray(recs, dtype=np.int64)
        B = len(recs)
        out = f["poses"].load_batch(recs, parallel=False)
        pose_vals, n_pose = _npy_rows(out["data"], out["sizes"], np.dtype("<f4"), 7)
        out = f["annot_frame_idx"].load_batch(recs, parallel=False)
        fidx, n = _npy_rows(out["data"], out["sizes"], np.dtype("<i4"), 1)
        fidx = fidx.reshape(B, -1)
        fps_field = "src_fps" if "src_fps" in f else "fps"
        src_fps = np.asarray(f[fps_field].load_batch(recs, parallel=False)["data"], dtype=np.float64).reshape(B)
        if not np.array_equal(n, n_pose):
            raise ValueError("poses / annot_frame_idx row counts differ (records not in a consistent store)")
        N = fidx.shape[1]
        valid = np.arange(N)[None, :] < n[:, None]
        P = pose_vals.astype(np.float64); P[~valid] = 0.0
        T = np.where(valid, fidx.astype(np.float64) / src_fps[:, None], np.inf)
        return P, T, n

    def poses_at(self, recs, t_sec) -> np.ndarray:
        """World->camera poses [B, T, 7] ([tx ty tz qx qy qz qw]) interpolated (linear t, slerp q) to the frame times
        `t_sec` [B, T] of records `recs` [B] (e.g. `batch["video_rec"]`, `batch["video_t_sec"]`). Vectorised over
        the batch (padded annotations, batched searchsorted); no per-record Python math."""
        from .video import interpolate_poses_batched
        recs = np.asarray(recs).reshape(-1); t = np.asarray(t_sec, dtype=np.float64).reshape(len(recs), -1)
        P, T, n = self._annot_batch(recs)
        return interpolate_poses_batched(P, T, n, t)

    def poses_at_loop(self, recs, t_sec) -> np.ndarray:
        """Reference implementation of `poses_at`: one `interpolate_poses` call per record (kept for tests)."""
        from .video import interpolate_poses
        recs = np.asarray(recs).reshape(-1); t = np.asarray(t_sec, dtype=np.float64).reshape(len(recs), -1)
        out = np.empty((len(recs), t.shape[1], 7), dtype=np.float32)
        for i, r in enumerate(recs.tolist()):
            poses, t_annot = self._annot(int(r))
            out[i] = interpolate_poses(poses, t_annot, t[i])
        return out

    def ego_motion_at(self, recs, t_sec) -> tuple[np.ndarray, np.ndarray]:
        """(poses [B, T, 7], deltas [B, T-1, 6]) at the frame times: the interpolated world->camera poses and the
        frame-to-frame ego-motion `[dx dy dz rx ry rz]` in camera-t axes (see `video.ego_motion`). Delta t pairs with
        frame t+1: the motion the camera made *since the last frame*."""
        from .video import ego_motion
        poses = self.poses_at(recs, t_sec)
        return poses, ego_motion(poses)

    def ego_motion_transform(self, field: str = "video", poses_key: str = "poses", ego_key: str = "ego",
                             pad_first: bool = False) -> "EgoMotionTransform":
        """A `SlipstreamLoader(after_batch_transforms=[...])` callable that adds `poses` [B, T, 7] and `ego` [B, T-1, 6]
        (or [B, T, 6] zero-padded at frame 0 with `pad_first`) to every batch, from `<field>_rec` / `<field>_t_sec`."""
        return EgoMotionTransform(self, field, poses_key, ego_key, pad_first)

    def __repr__(self) -> str:
        fmt, res, fps = self.store_key
        return (f"VideoDataset({self.name!r}, split={self.split!r}, subset={self.subset!r}, store={fmt}/{res}/{fps or 'native'}fps, "
                f"clips={len(self):,}, rate_hz={self.rate_hz})")


@dataclass
class EgoMotionTransform:
    """After-batch transform: interpolated poses + frame-to-frame ego-motion as torch tensors in the batch dict.

        loader = SlipstreamLoader(ds, ..., pipelines={"video": [DecodeVideoWindow(...)]},
                                  after_batch_transforms=[ds.ego_motion_transform()])
        batch["poses"]  # [B, T, 7] float32 world->camera at the decoded frame times
        batch["ego"]    # [B, T-1, 6] float32 [dx dy dz rx ry rz] in the previous frame's axes (delta t-1 -> frame t)
    """
    ds: VideoDataset
    field: str = "video"
    poses_key: str = "poses"
    ego_key: str = "ego"
    pad_first: bool = False          # True: ego is [B, T, 6] with a zero row for frame 0 (aligns with frames)

    def __call__(self, batch: dict) -> dict:
        import torch
        poses, ego = self.ds.ego_motion_at(batch[f"{self.field}_rec"], batch[f"{self.field}_t_sec"])
        if self.pad_first:
            ego = np.concatenate([np.zeros((ego.shape[0], 1, 6), dtype=ego.dtype), ego], axis=1)
        batch[self.poses_key] = torch.from_numpy(poses)
        batch[self.ego_key] = torch.from_numpy(ego)
        return batch


# ----------------------------------------------------------------------------- selection

def select_clips(records, split_df, split: str, subset_df, where: str | None, channel_cap: float | None, seed: int):
    """records (record_idx, clip_id, ...) ∩ split ∩ subset, then `where`, then per-channel cap (random thinning)."""
    import pandas as pd
    df = records
    if subset_df is not None:
        # the store's own columns win (record_idx / fps / duration_s are store-specific: fps stores are decimated)
        sub = subset_df[[c for c in subset_df.columns if c == "clip_id" or c not in df.columns]]
        df = df.merge(sub, on="clip_id", how="inner")
    if split_df is not None:
        cols = [c for c in split_df.columns if c == "clip_id" or c not in df.columns]
        df = df.merge(split_df[cols], on="clip_id", how="left")
        df["split"] = df["split"].fillna("excluded")
        if split != "all":
            df = df[df["split"] == split]
    if where:
        df = df.query(where)
    if channel_cap is not None:
        if "channel_id" not in df:
            raise ValueError("channel_cap needs a channel_id column (a subset or split table with channels)")
        cap = int(channel_cap * len(df)); rng = np.random.default_rng(seed)
        keep = []
        for _c, g in df.groupby("channel_id", dropna=False, sort=False):
            keep.append(g.index.to_numpy() if len(g) <= cap else rng.choice(g.index.to_numpy(), cap, replace=False))
        df = df.loc[np.sort(np.concatenate(keep))] if keep else df
    return df.sort_values("record_idx").reset_index(drop=True)


def _store_scalars(store_dir: Path, names: tuple[str, ...]):
    """Whole-store scalar fields (`<field>.npy` written by the merge) as a DataFrame column set."""
    out = {}
    for n in names:
        p = store_dir / f"{n}.npy"
        if p.exists():
            out[n] = np.load(p, mmap_mode="r")
    return out


def load_video(config, split: str | None = None, fmt: str | None = None, res: str | None = None, rate_hz: float | None = None,
               fps: int | None = None, subset: str | None = None, split_version: str | None = None, where: str | None = None,
               channel_cap: float | None = None, seed: int = 0, download: bool = True) -> VideoDataset:
    import pandas as pd
    from slipstream.cache import OptimizedCache
    from .runtime_platform import configure_slipstream_cache

    meta = config.metadata or {}
    tree = meta.get("tree", config.name)
    fmt = fmt or meta.get("default_fmt", "h265")
    res = meta.get("res_aliases", {}).get(str(res), res) if res else meta.get("default_res")
    split = split or meta.get("default_split", "train")
    native = fps == "native"                       # the un-decimated store, whatever the default rate says
    if native:
        fps = None
    if rate_hz is None and fps is None and not native:
        rate_hz = meta.get("default_rate_hz")
    split_version = split_version or meta.get("default_split_version")
    cache_base = Path(configure_slipstream_cache())

    # store
    store_dir = key = None
    for k in rank_stores(config.stores, fmt, res, rate_hz, fps):
        store_dir = resolve_store(config.stores[k], tree, cache_base, download=download)
        if store_dir is not None:
            key = k; break
        if store_name(config.stores[k]) not in _WARNED:      # once per process per store
            _WARNED.add(store_name(config.stores[k]))
            warnings.warn(f"{config.name}: store {store_name(config.stores[k])} is registered but not built/synced yet; trying the next one", stacklevel=3)
    if store_dir is None:
        raise FileNotFoundError(f"{config.name}: no usable store for fmt={fmt} res={res} rate_hz={rate_hz} fps={fps}")
    cache = OptimizedCache.load(store_dir, verbose=False)

    # clip table: records.parquet + per-clip scalars from the store
    records = pd.read_parquet(store_dir / "records.parquet")
    for n, arr in _store_scalars(store_dir, ("fps", "duration_s", "src_fps", "src_num_frames")).items():
        records[n] = np.asarray(arr)[records["record_idx"].to_numpy()]
    # the split table is always joined (also for split="all"): it carries the channel / carrier / strata columns for `where`
    if split_version not in config.splits:
        raise KeyError(f"unknown split version {split_version!r}; available: {sorted(config.splits)}")
    split_df = pd.read_parquet(resolve_file(f"splits/{split_version}.parquet", config.splits[split_version], tree, cache_base))
    subset_df = None
    if subset is None:
        subset = meta.get("default_subset")          # the population; subset="all" = every clip in the store
    if subset == "all":
        subset = None
    if subset is not None:
        if subset not in config.subsets:
            raise KeyError(f"unknown subset {subset!r}; available: {sorted(config.subsets)}")
        subset_df = pd.read_parquet(resolve_file(f"subsets/{subset}.parquet", config.subsets[subset], tree, cache_base))
    clips = select_clips(records, split_df, split, subset_df, where, channel_cap, seed)
    stats = (meta.get("stats") or {}).get(store_name(config.stores[key])) or (meta.get("stats") or {}).get("rgb")
    return VideoDataset(config.name, key, store_dir, cache, clips, split, subset, rate_hz, stats)
