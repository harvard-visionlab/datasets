"""Offline tests for the video side of the registry: store ranking, clip selection, window sampling, pose interpolation."""
import numpy as np
import pandas as pd
import pytest

from visionlab.datasets.registry import get_config
from visionlab.datasets.video import camera_center, camera_centers, ego_motion, interpolate_poses, relative_motion
from visionlab.datasets.video_dataset import VideoDataset, WindowSampler, rank_stores, select_clips

STORES = get_config("spatialvid-hq").stores


def test_config_is_video():
    cfg = get_config("spatialvid-hq")
    assert cfg.is_video and not get_config("imagenet1k").is_video
    assert "v3" in cfg.splits and "person_carried_v0" in cfg.subsets
    assert cfg.metadata["default_subset"] in cfg.subsets          # load() filters to the population unless subset="all"


@pytest.mark.parametrize("rate,expected_fps", [(15, [15, 30, None]), (10, [30, None]), (5, [15, 30, None]), (30, [30, None]), (7.5, [15, 30, None]), (60, [None])])
def test_rank_stores_by_rate(rate, expected_fps):
    keys = rank_stores(STORES, "h265", "456x256", rate, None)
    assert [k[2] for k in keys] == expected_fps


def test_rank_stores_explicit_fps_and_errors():
    assert rank_stores(STORES, "h265", "640x360", 15, 30) == [("h265", "640x360", 30)]
    assert [k[2] for k in rank_stores(STORES, "h265", "456x256", None, None)] == [None, 30, 15]
    with pytest.raises(KeyError):
        rank_stores(STORES, "h265", "999x999", 15, None)
    with pytest.raises(KeyError):
        rank_stores(STORES, "h265", "456x256", 15, 24)
    with pytest.raises(ValueError):
        rank_stores({("h265", "456x256", 30): "s3://x/a"}, "h265", "456x256", 7, None)


def _tables():
    records = pd.DataFrame({"record_idx": np.arange(8), "clip_id": [f"c{i}" for i in range(8)], "duration_s": [3.0, 9.0, 12.0, 8.5, 15.0, 2.0, 10.0, 11.0], "fps": [30.0] * 8})
    split = pd.DataFrame({"clip_id": [f"c{i}" for i in range(7)], "split": ["train", "train", "val", "test", "train", "train", "val"],
                          "channel_id": ["A", "A", "B", "C", "A", "A", "B"], "carrier": ["walk", "rig", "walk", "walk", "walk", "walk", "walk"]})
    subset = pd.DataFrame({"clip_id": ["c0", "c1", "c2", "c3", "c4", "c6", "c7"], "record_idx": [99] * 7, "fps": [60.0] * 7, "duration_s": [-1.0] * 7,
                           "carrier": ["walk", "rig", "walk", "walk", "walk", "walk", "walk"]})
    return records, split, subset


def test_select_clips_split_subset_where():
    records, split, subset = _tables()
    train = select_clips(records, split, "train", subset, None, None, 0)
    assert train["clip_id"].tolist() == ["c0", "c1", "c4"]            # c5 not in subset
    assert train["record_idx"].tolist() == [0, 1, 4]                  # store record_idx wins over the subset's column
    assert train["fps"].tolist() == [30.0] * 3 and train["duration_s"].tolist() == [3.0, 9.0, 15.0]   # store fps / duration win, no _x/_y suffixes
    assert not [c for c in train.columns if c.endswith(("_x", "_y"))]
    allc = select_clips(records, split, "all", subset, None, None, 0)
    assert allc.loc[allc.clip_id == "c7", "split"].item() == "excluded"   # not in the split table
    walk = select_clips(records, split, "train", subset, "carrier == 'walk'", None, 0)
    assert walk["clip_id"].tolist() == ["c0", "c4"]
    assert select_clips(records, split, "test", None, None, None, 0)["clip_id"].tolist() == ["c3"]


def test_select_clips_channel_cap():
    records, split, subset = _tables()
    df = select_clips(records, split, "all", None, None, 0.3, 0)     # 7 clips in the split table -> cap 2 per channel
    assert df.groupby("channel_id").size().max() <= 2
    df2 = select_clips(records, split, "all", None, None, 0.3, 0)
    assert df["clip_id"].tolist() == df2["clip_id"].tolist()          # seeded
    with pytest.raises(ValueError):
        select_clips(records, None, "all", None, None, 0.3, 0)


def test_window_sampler_seeded_and_filters_short_clips():
    records, split, subset = _tables()
    ds = VideoDataset("x", ("h265", "456x256", None), None, None, records, "all", None, 15)
    ws = ds.window_sampler(window_s=8.0, anchors_per_clip=2, seed=1)
    assert set(ws.recs.tolist()) == {1, 2, 3, 4, 6, 7}                 # 3 s, 2 s clips dropped; 8.5 s kept (8 + 3/30 < 8.5)
    r1, t1 = ws.sample(0); r2, t2 = ws.sample(0); r3, t3 = ws.sample(1)
    assert len(r1) == 12 and np.array_equal(t1, t2) and not np.array_equal(t1, t3)
    assert (t1 >= 0).all() and (t1 + 8.0 <= np.repeat(ws.max_t0 + 8.0, 2) + 1e-6).all()


def test_interpolate_poses_in_seconds():
    t_annot = np.array([0.0, 0.2, 0.4])
    poses = np.array([[0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 1]], np.float32)
    out = interpolate_poses(poses, t_annot, np.array([0.1, 0.3, 0.9]))
    assert np.allclose(out[:, 0], [0.5, 1.5, 2.0])                      # linear, clamped after the last annotation
    assert np.allclose(np.linalg.norm(out[:, 3:], axis=1), 1)


def _random_poses(rng, shape):
    q = rng.normal(size=shape + (4,)); q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return np.concatenate([rng.normal(size=shape + (3,)), q], axis=-1).astype(np.float32)


def test_ego_motion_matches_pairwise_reference():
    rng = np.random.default_rng(0)
    p = _random_poses(rng, (2, 5))                                       # [B=2, T=5, 7]
    d = ego_motion(p)
    assert d.shape == (2, 4, 6) and d.dtype == np.float32
    p64 = p.astype(np.float64)                                           # the pairwise reference computes in the input dtype
    for b in range(2):
        for t in range(4):
            assert np.allclose(d[b, t], relative_motion(p64[b, t], p64[b, t + 1]), atol=1e-5)
    assert np.allclose(camera_centers(p)[1, 3], camera_center(p[1, 3]), atol=1e-6)
    assert np.allclose(ego_motion(np.repeat(p[:, :1], 3, axis=1)), 0)   # no motion → zero deltas (incl. the angle≈0 branch)


def test_ego_motion_forward_walk_is_positive_dz():
    # camera moving along its own +z (forward) with identity rotation: world->camera t = -c
    c = np.stack([np.zeros(6), np.zeros(6), np.arange(6) * 0.1], axis=1)
    p = np.concatenate([-c, np.tile([0, 0, 0, 1], (6, 1))], axis=1).astype(np.float32)
    d = ego_motion(p)
    assert np.allclose(d[:, 2], 0.1, atol=1e-6) and np.allclose(d[:, [0, 1, 3, 4, 5]], 0, atol=1e-6)


def test_poses_at_batched(monkeypatch):
    records, *_ = _tables()
    ds = VideoDataset("x", ("h265", "456x256", None), None, None, records, "all", None, 15)
    monkeypatch.setattr(ds, "_annot", lambda rec: (np.array([[rec, 0, 0, 0, 0, 0, 1], [rec + 1, 0, 0, 0, 0, 0, 1]], np.float32), np.array([0.0, 1.0])))
    out = ds.poses_at(np.array([3, 5]), np.array([[0.0, 0.5], [0.25, 1.0]]))
    assert out.shape == (2, 2, 7) and np.allclose(out[:, :, 0], [[3.0, 3.5], [5.25, 6.0]])
