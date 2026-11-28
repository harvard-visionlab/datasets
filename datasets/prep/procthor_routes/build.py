"""Lossless stores + split/subset tables from rendered route windows (pilot.py output). datasets env (slipstream, av,
torchcodec).

    python -m datasets.prep.procthor_routes.build store --render <dir> --split val --cell objects --store-root <dir>
    python -m datasets.prep.procthor_routes.build index --render <dir> --split val --tree-root <dir>   # both cells
    python -m datasets.prep.procthor_routes.build pair --store-root <dir> --split val                  # cross-cell check
    python -m datasets.prep.procthor_routes.build records --render <dir> --split val --cell objects --store-root <dir>
        # rewrite records.parquet of a built store (new columns; clip_ids and existing columns must be unchanged)

One store per (cell, split): `procthor-routes-<cell>-lossless-160x120-<split>`, one record per window, both lengths
(load() picks a store by split; the subset selects the route type and length). Records in sorted clip_id order;
clip_id `<split>/house_XXXXX/<subset>_<k>`, identical in both cells (same camera path):
    video        bytes            mp4: H.264 RGB lossless (libx264rgb qp 0, rgb24), 30 fps, frame i at pts i
    positions    float64[125,2]   Habitat (x, z) metres per frame; rows >= num_frames are NaN (61-frame windows)
    headings     float64[125]     Habitat heading (rad) per frame; NaN-padded likewise
    fps / num_frames / duration_s 30 / T+4 / (T+4)/30. T+4 = the window (20 warm-up + rollout + 1 target) + 4 held frames,
                                  so `eval_windows(T)` serves exactly one window per record, from frame 0.
records.parquet: record_idx, clip_id, split, subset, route, window_frames (T), house_idx, k, n_rooms, n_objects,
time_of_day, seed, event_start / event_end (frame index; None for Brownian), turn_rate, p_keep, info / metrics (JSON),
door_crossings / door_crossings_rollout / rooms_visited (room changes over frames 0..T-1 / at frames >= 20, distinct rooms;
`rooms()`: the camera's room per frame from the house's floor polygons; a doorway or open archway between two rooms
counts, and stepping back and forth across a threshold counts each time).
Tables (`<tree-root>/procthor-routes-<cell>/`): splits/v1.parquet (clip_id, split), subsets/<subset>.parquet (clip_id).

Verification (every record; exit 1 on failure): all frames decoded with torchcodec == rendered frames byte for byte;
the eval window (slipstream DecodeVideoWindow, T frames, t0 = 0.5/30) == frames[:T]; positions / headings == rendered
(float64 bytes) and NaN beyond num_frames.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..procthor_walks.lossless import ENCODE, encode_lossless
from ..procthor_walks.common import software_versions

FPS = 30
RES = "160x120"
TMAX = 121 + 4
CELLS = ("objects", "empty")
ROUTE = {"goal": "goal_directed", "turn": "full_turn", "outback": "out_and_back", "loop": "loop", "brownian": "brownian",
         "brownian_inertia80": "brownian_inertia80", "brownian_inertia95": "brownian_inertia95"}
EVENT = {"turn": ("turn_start", "turn_end"), "outback": ("out_start", "back_end"), "loop": ("loop_start", "loop_end")}


def dataset(cell: str) -> str:
    return f"procthor-routes-{cell}"


def store_name(cell: str, split: str) -> str:
    return f"{dataset(cell)}-lossless-{RES}-{split}"


ROLLOUT_START = 20


def houses_file(render: Path, split: str) -> Path:
    """The ProcTHOR house file the render used: <render>/../source/<split>.json.gz."""
    return render.parent / "source" / f"{split}.json.gz"


def _inside(x: float, z: float, poly: np.ndarray) -> bool:
    px, pz = poly[:, 0], poly[:, 1]; qx, qz = np.roll(px, -1), np.roll(pz, -1)
    hit = ((pz > z) != (qz > z)) & (x < (qx - px) * (z - pz) / np.where(qz == pz, 1e-12, qz - pz) + px)
    return bool(hit.sum() % 2)


def rooms(pos_u: np.ndarray, polys: list[np.ndarray]) -> list[int | None]:
    """Room index per frame (Unity x, z vs floor polygons); a point in no room keeps the previous room."""
    out, prev = [], None
    for x, z in pos_u:
        r = next((i for i, p in enumerate(polys) if _inside(x, z, p)), None)
        prev = prev if r is None else r
        out.append(prev)
    return out


def door_stats(pos_u: np.ndarray, T: int, polys: list[np.ndarray]) -> dict:
    lab = rooms(np.asarray(pos_u)[:T], polys)
    ch = [t for t in range(1, T) if lab[t] is not None and lab[t - 1] is not None and lab[t] != lab[t - 1]]
    return dict(door_crossings=len(ch), door_crossings_rollout=sum(t >= ROLLOUT_START for t in ch),
                rooms_visited=len({r for r in lab if r is not None}))


def records(render: Path, split: str) -> pd.DataFrame:
    """One row per rendered window (both cells share it), sorted by clip_id."""
    houses = json.load(gzip.open(houses_file(render, split), "rt"))
    rows = []
    for hj in sorted(render.glob("house_*/house.json")):
        m = json.loads(hj.read_text())
        polys = [np.array([[p["x"], p["z"]] for p in r["floorPolygon"]], float) for r in houses[m["house_idx"]]["rooms"]]
        assert len(polys) == m["n_rooms"], (hj, len(polys), m["n_rooms"])
        for w in m["windows"]:
            if w["kind"] not in ROUTE:
                continue
            sub = f"{ROUTE[w['kind']]}_{w['T']}"; info = w["info"]; w["k"] = int(w["name"].rsplit("_", 1)[1])
            ev = EVENT.get(w["kind"])
            rows.append(dict(clip_id=f"{split}/house_{m['house_idx']:05d}/{sub}_{w['k']}", split=split, subset=sub,
                             route=ROUTE[w["kind"]], window_frames=int(w["T"]), house_idx=int(m["house_idx"]), k=int(w["k"]),
                             n_rooms=int(m["n_rooms"]), n_objects=int(m["n_objects"]), time_of_day=m["time_of_day"],
                             seed=json.dumps(w["seed"]),
                             event_start=int(info[ev[0]]) if ev else None, event_end=int(info[ev[1]]) if ev else None,
                             turn_rate=float(info["turn_rate"]) if "turn_rate" in info else None,
                             p_keep=float(info["p_keep"]) if "p_keep" in info else None,
                             info=json.dumps(info), metrics=json.dumps(w["metrics"]),
                             **door_stats(np.load(hj.parent / f"{w['name']}.npz")["pos_u"], int(w["T"]), polys),
                             src_path=str(hj.parent / f"{w['name']}.npz")))
    df = pd.DataFrame(rows).sort_values("clip_id").reset_index(drop=True)
    df.insert(0, "record_idx", range(len(df)))
    return df


def read_window(path: str, cell: str):
    z = np.load(path)
    return z[cell], z["pos_h"], z["head_h"]


def padded(a: np.ndarray) -> np.ndarray:
    out = np.full((TMAX, *a.shape[1:]), np.nan, np.float64); out[: len(a)] = a
    return out


class Source:
    def __init__(self, paths: list[str], cell: str, out: Path):
        self.paths, self.cell, self.out = paths, cell, out

    @property
    def cache_path(self) -> Path:
        return self.out

    @property
    def field_types(self) -> dict[str, str]:
        return {"video": "bytes", "positions": f"float64[{TMAX},2]", "headings": f"float64[{TMAX}]",
                "fps": "float", "num_frames": "int", "duration_s": "float"}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        frames, pos, head = read_window(self.paths[i], self.cell)
        n = len(frames)
        if pos.dtype != np.float64 or head.dtype != np.float64 or len(pos) != n or len(head) != n:
            raise ValueError(f"{self.paths[i]}: poses {pos.shape} {pos.dtype} / {head.shape} {head.dtype}, {n} frames")
        return {"video": encode_lossless(frames), "positions": padded(pos), "headings": padded(head),
                "fps": float(FPS), "num_frames": n, "duration_s": n / FPS}


def verify(store: Path, rec: pd.DataFrame, cell: str, threads: int) -> dict:
    import torch
    torch.set_num_threads(1)
    from slipstream.cache import OptimizedCache
    from slipstream.decoders import DecodeVideoWindow
    from torchcodec.decoders import VideoDecoder

    cache = OptimizedCache.load(store, verbose=False)
    fv, fp, fh = cache.fields["video"], cache.fields["positions"], cache.fields["headings"]
    stages = {T: DecodeVideoWindow(T=T, rate_hz=FPS, t0_key="t0", device="cpu", num_workers=2) for T in (61, 121)}

    def read(field, i):
        out = field.load_batch(np.array([i], dtype=np.int64), parallel=False)
        return {k: np.array(v, copy=True) for k, v in out.items()}

    def one(i: int) -> dict:
        frames, pos, head = read_window(rec["src_path"].iloc[i], cell)
        n, T = len(frames), int(rec["window_frames"].iloc[i])
        v = read(fv, i); blob = bytes(v["data"][0][: int(v["sizes"][0])])
        dec = VideoDecoder(blob, dimension_order="NHWC")
        allf = dec.get_frames_in_range(0, n).data.numpy()
        frames_ok = dec.metadata.num_frames == n and allf.shape == frames.shape and np.array_equal(allf, frames)
        sp = np.asarray(read(fp, i)["data"][0]); sh = np.asarray(read(fh, i)["data"][0])
        pos_ok = sp.dtype == np.float64 and sp[:n].tobytes() == pos.tobytes() and np.isnan(sp[n:]).all()
        head_ok = sh.dtype == np.float64 and sh[:n].tobytes() == head.tobytes() and np.isnan(sh[n:]).all()
        out = stages[T]({"data": v["data"][:1], "sizes": v["sizes"][:1], "field": "video",
                         "sample_data": {"t0": np.array([0.5 / FPS], np.float32)}})
        win = out["video"].permute(0, 1, 3, 4, 2).cpu().numpy()[0]
        return dict(i=i, frames=frames_ok, pos=pos_ok, head=head_ok, window=np.array_equal(win, frames[:T]),
                    mb=len(blob) / 1e6)

    t = time.perf_counter()
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(one, range(len(rec))))
    bad = [rec["clip_id"].iloc[r["i"]] for r in res if not (r["frames"] and r["pos"] and r["head"] and r["window"])]
    out = dict(records=len(rec), frames_byte_identical=sum(r["frames"] for r in res),
               positions_identical=sum(r["pos"] for r in res), headings_identical=sum(r["head"] for r in res),
               eval_window_byte_identical=sum(r["window"] for r in res), failures=bad[:20],
               decode="torchcodec get_frames_in_range (all frames) + slipstream DecodeVideoWindow (T=window_frames, "
                      f"rate {FPS} Hz, t0=0.5/{FPS})",
               video_mb_per_record=round(float(np.mean([r["mb"] for r in res])), 3),
               seconds=round(time.perf_counter() - t), verified=time.strftime("%Y-%m-%dT%H:%M:%S"))
    out["ok"] = not bad
    return out


def cmd_store(a) -> int:
    from slipstream.cache import OptimizedCache
    rec = records(a.render, a.split)
    name = store_name(a.cell, a.split); store = a.store_root / name
    if not a.verify_only:
        if store.exists() and any(store.iterdir()):
            sys.exit(f"{store} exists and is not empty; remove it to rebuild")
        print(f"{name}: {len(rec)} windows from {a.render}, {ENCODE}", flush=True)
        t = time.perf_counter()
        cache = OptimizedCache.build(Source(rec["src_path"].tolist(), a.cell, store), output_dir=store, verbose=True,
                                     num_workers=a.workers)
        if len(cache) != len(rec):
            sys.exit(f"store has {len(cache)} records, expected {len(rec)}")
        rec.to_parquet(store / "records.parquet", index=False)
        vbytes = sum(p.stat().st_size for p in store.glob("video*") if p.is_file())
        sm = dict(dataset=dataset(a.cell), store=name, split=a.split, num_records=len(rec), cell=a.cell,
                  render=str(a.render), subsets=rec["subset"].value_counts().sort_index().to_dict(), encode=ENCODE,
                  fields=Source([], a.cell, store).field_types, software=software_versions(), video_bytes=vbytes,
                  encode_seconds=round(time.perf_counter() - t), built=time.strftime("%Y-%m-%dT%H:%M:%S"))
        (store / "store_manifest.json").write_text(json.dumps(sm, indent=1, default=int))
        print(f"BUILT {name}: {len(rec)} records, video {vbytes / 1e9:.2f} GB", flush=True)
    v = verify(store, pd.read_parquet(store / "records.parquet"), a.cell, a.workers)
    sm = json.loads((store / "store_manifest.json").read_text()); sm["verify"] = v
    (store / "store_manifest.json").write_text(json.dumps(sm, indent=1, default=int))
    print("VERIFY " + ("OK " if v["ok"] else "FAILED ") + json.dumps(v, default=int), flush=True)
    return 0 if v["ok"] else 1


def cmd_records(a) -> int:
    """New records.parquet for a built store: same clip_ids in the same order, existing columns unchanged."""
    store = a.store_root / store_name(a.cell, a.split)
    old, new = pd.read_parquet(store / "records.parquet"), records(a.render, a.split)
    if old["clip_id"].tolist() != new["clip_id"].tolist():
        sys.exit("clip_ids differ from the store's records")
    diff = [c for c in old.columns if c not in new.columns or not old[c].equals(new[c])]
    if diff:
        sys.exit(f"existing columns changed: {diff}")
    added = [c for c in new.columns if c not in old.columns]
    new.to_parquet(store / "records.parquet", index=False)
    sm = json.loads((store / "store_manifest.json").read_text())
    sm.setdefault("records_updates", []).append(dict(added=added, at=time.strftime("%Y-%m-%dT%H:%M:%S")))
    (store / "store_manifest.json").write_text(json.dumps(sm, indent=1, default=int))
    print(f"RECORDS {store.name}: {len(new)} rows, added {added}", flush=True)
    return 0


def cmd_index(a) -> int:
    rec = records(a.render, a.split)
    for cell in CELLS:
        d = a.tree_root / dataset(cell)
        (d / "splits").mkdir(parents=True, exist_ok=True); (d / "subsets").mkdir(exist_ok=True)
        sp = d / "splits" / "v1.parquet"
        old = pd.read_parquet(sp) if sp.exists() else pd.DataFrame(columns=["clip_id", "split"])
        new = pd.concat([old[old["split"] != a.split], rec[["clip_id", "split"]]]).sort_values("clip_id")
        new.reset_index(drop=True).to_parquet(sp, index=False)
        for sub, g in rec.groupby("subset"):
            f = d / "subsets" / f"{sub}.parquet"
            old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["clip_id"])
            ids = pd.concat([old[~old["clip_id"].str.startswith(f"{a.split}/")], g[["clip_id"]]]).sort_values("clip_id")
            ids.reset_index(drop=True).to_parquet(f, index=False)
        print(f"{dataset(cell)}: split {a.split} {len(rec)} clips; subsets " +
              json.dumps(rec["subset"].value_counts().sort_index().to_dict()), flush=True)
    return 0


def cmd_pair(a) -> int:
    from slipstream.cache import OptimizedCache
    st = {c: a.store_root / store_name(c, a.split) for c in CELLS}
    r = {c: pd.read_parquet(st[c] / "records.parquet") for c in CELLS}
    same_ids = r["objects"]["clip_id"].tolist() == r["empty"]["clip_id"].tolist()
    f = {c: OptimizedCache.load(st[c], verbose=False).fields for c in CELLS}
    idx = np.arange(len(r["objects"]), dtype=np.int64)
    pos = {c: np.asarray(f[c]["positions"].load_batch(idx, parallel=False)["data"]) for c in CELLS}
    head = {c: np.asarray(f[c]["headings"].load_batch(idx, parallel=False)["data"]) for c in CELLS}
    ok = same_ids and pos["objects"].tobytes() == pos["empty"].tobytes() and head["objects"].tobytes() == head["empty"].tobytes()
    print(f"PAIR {'OK' if ok else 'FAILED'}: {len(idx)} records, same clip_ids/order {same_ids}, "
          f"identical poses {pos['objects'].tobytes() == pos['empty'].tobytes()}", flush=True)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("store"); s.add_argument("--render", type=Path, required=True); s.add_argument("--split", required=True)
    s.add_argument("--cell", choices=CELLS, required=True); s.add_argument("--store-root", type=Path, required=True)
    s.add_argument("--workers", type=int, default=8); s.add_argument("--verify-only", action="store_true")
    i = sub.add_parser("index"); i.add_argument("--render", type=Path, required=True); i.add_argument("--split", required=True)
    i.add_argument("--tree-root", type=Path, required=True)
    p = sub.add_parser("pair"); p.add_argument("--store-root", type=Path, required=True); p.add_argument("--split", required=True)
    r = sub.add_parser("records"); r.add_argument("--render", type=Path, required=True); r.add_argument("--split", required=True)
    r.add_argument("--cell", choices=CELLS, required=True); r.add_argument("--store-root", type=Path, required=True)
    a = ap.parse_args(argv)
    return {"store": cmd_store, "index": cmd_index, "pair": cmd_pair, "records": cmd_records}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
