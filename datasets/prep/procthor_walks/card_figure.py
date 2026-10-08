"""Card figure: one paired val walk, furnished (objects) vs empty, same 61-frame window, plus the top-down path
(arrows = facing direction).

    python -m datasets.prep.procthor_walks.card_figure --store-root <dir> --out datasets/cards/figures/procthor-walks-pair.png

Reads the two val stores (stage 2). The walk is the first val record with 4+ rooms and 30-50 objects (deterministic).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import store_name

LEN_WINDOW, START, SHOW = 61, 300, (0, 12, 24, 36, 48, 60)


def facing(head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit facing direction in (x, y) from heading. Generator convention (meta coord_convention): forward motion gives
    egomotion (dX, dY) = (0, -1) under the student's rotation (dX c - dY s, dX s + dY c), so forward = (-sin h, -cos h);
    right = (cos h, -sin h), which is clockwise of forward only with the y axis pointing down."""
    return -np.sin(head), -np.cos(head)


def load_walk(store: Path, i: int):
    from slipstream.cache import OptimizedCache
    from torchcodec.decoders import VideoDecoder
    c = OptimizedCache.load(store, verbose=False)
    r = np.array([i], dtype=np.int64)
    v = c.fields["video"].load_batch(r, parallel=False)
    video = bytes(v["data"][0][: int(v["sizes"][0])])
    pos = np.array(c.fields["positions"].load_batch(r)["data"][0]).reshape(-1, 2)
    head = np.array(c.fields["headings"].load_batch(r)["data"][0]).reshape(-1)
    frames = VideoDecoder(video, dimension_order="NHWC").get_frames_in_range(START, START + LEN_WINDOW).data.numpy()
    return frames, pos, head


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obj = a.store_root / store_name("procthor-walks-objects", "val")
    emp = a.store_root / store_name("procthor-walks-empty", "val")
    rec = pd.read_parquet(obj / "records.parquet")
    row = rec[(rec.n_rooms >= 4) & rec.n_objects.between(30, 50)].iloc[0]
    i = int(row.record_idx)
    assert pd.read_parquet(emp / "records.parquet").clip_id.iloc[i] == row.clip_id
    f_obj, pos, head = load_walk(obj, i)
    f_emp, pos_e, _ = load_walk(emp, i)
    assert np.array_equal(pos, pos_e)

    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(2, len(SHOW) + 2, width_ratios=[1] * len(SHOW) + [0.08, 2.1], wspace=0.05, hspace=0.08)
    for r, (frames, label) in enumerate([(f_obj, "objects"), (f_emp, "empty")]):
        for k, t in enumerate(SHOW):
            ax = fig.add_subplot(gs[r, k])
            ax.imshow(frames[t])
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(f"step {START + t}", fontsize=9)
            if k == 0:
                ax.set_ylabel(label, fontsize=10)
    ax = fig.add_subplot(gs[:, -1])
    ax.plot(pos[:, 0], pos[:, 1], lw=0.6, color="0.65", label="whole walk (1,000 steps)")
    w = slice(START, START + LEN_WINDOW)
    ax.plot(pos[w, 0], pos[w, 1], lw=2, color="C1", label=f"window shown ({START}-{START + LEN_WINDOW - 1})")
    q = np.array(SHOW) + START
    fx, fy = facing(head[q])
    ax.quiver(pos[q, 0], pos[q, 1], fx, fy, color="C1", angles="xy", scale=12, width=0.008)
    ax.set_aspect("equal")
    ax.invert_yaxis()                                    # y down = top-down view, not mirrored (see facing())
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m, down)")
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"{row.clip_id}: {row.n_rooms} rooms, {row.n_objects} objects", fontsize=9)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"wrote {a.out} ({row.clip_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
