"""procthor-goalwalks stores from the goalwalks render (goalwalks.py): same format, encoder, record schema and split
names as procthor-walks. datasets env (slipstream, av, torchcodec).

    python -m datasets.prep.procthor_routes.goalwalks_build index  --plan <P> --render <RW> --houses-root <src> --work <W>
    python -m datasets.prep.procthor_routes.goalwalks_build encode --work <W> --cell objects --folder train --store-root <ST> [--workers 16]
    python -m datasets.prep.procthor_routes.goalwalks_build verify --work <W> --cell objects --folder train --store-root <ST> [--windows 256]
    python -m datasets.prep.procthor_routes.goalwalks_build pair   --store-root <ST> --folder train
    python -m datasets.prep.procthor_routes.goalwalks_build stats  --work <W> --render <RW>        # RGB + ego-motion (train)
    python -m datasets.prep.procthor_routes.goalwalks_build publish --work <W> --cell objects --folder train --store-root <ST>

index: <W>/<dataset>/index/walks.parquet (one row per walk, store order: folder train/val/test, house, seed; clip_id
`<folder>/house_XXXXX/seed_SSSS` as procthor-walks), splits/v1.parquet (folder) and splits/rtq-r160.parquet (procthor-walks'
table, same clip_ids, copied after checking the clip_id sets are equal). Columns: procthor-walks' (house facts, camera,
mean_brightness, ...; motion = "goal_directed", step_std / angle_std = NaN) plus walkable_m2, goal_min_m, goals, legs,
replans, uturns, fixes, path_m, door_crossings.
encode: one store per (cell, folder) `procthor-goalwalks-<cell>-h264-160x120-<folder>`: video (procthor_walks.common.encode_walk:
h264 yuv444p crf 10, GOP 60, 30 fps), positions float32[1000,2], headings float32[1000] (habitat convention), fps,
num_frames, duration_s; records.parquet; store_manifest.json.
verify: every record's mp4 metadata (1000 frames, 160x120, 30 fps); `--windows` random 61-frame training windows decoded
with torchcodec vs the rendered frames (PSNR mean >= 40 dB, min >= 33 dB); positions / headings of EVERY record equal
the plan (float32). Writes the result into store_manifest.json["verify"]; exit 1 on failure.
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..procthor_walks.common import ENCODE, FPS, H, N_STEPS, W, encode_walk, software_versions
from .build import door_stats

CELLS = ("objects", "empty")
FOLDERS = ("train", "val", "test")
S3_BASE = "s3://visionlab-datasets/slipstream-cache"
LAB_BASE = Path("/n/lab_storage/alvarez_lab/Lab/datasets/slipstream")
WALKS_RTQ = LAB_BASE / "procthor-walks-objects" / "splits" / "rtq-r160.parquet"
LEN_WINDOW = 61


def dataset(cell: str) -> str:
    return f"procthor-goalwalks-{cell}"


def store_name(cell: str, folder: str) -> str:
    return f"{dataset(cell)}-h264-160x120-{folder}"


def cmd_index(a) -> int:
    rows = {c: [] for c in CELLS}
    for folder in FOLDERS:
        houses = json.load(gzip.open(a.houses_root / f"{folder}.json.gz", "rt"))
        for hd in sorted((a.render / folder).glob("house_*")):
            rj = hd / "render.json"
            if not rj.exists():
                sys.exit(f"{hd}: not rendered")
            rm = json.loads(rj.read_text()); pm = json.loads((a.plan / folder / hd.name / "plan.json").read_text())
            hi = rm["house_idx"]
            polys = [np.array([[p["x"], p["z"]] for p in r["floorPolygon"]], float) for r in houses[hi]["rooms"]]
            plan = np.load(a.plan / folder / hd.name / "plan.npz")
            for j, (pw, rw) in enumerate(zip(pm["walks"], rm["walks"], strict=True)):
                assert pw["seed"] == rw["seed"] == int(plan["seeds"][j])
                ck, st = pw["check"], pw["stats"]
                base = dict(clip_id=f"{folder}/{hd.name}/seed_{pw['seed']}", folder=folder, house=hd.name, seed=int(pw["seed"]),
                            src_path=str(hd / f"seed_{pw['seed']}.npz"), source="procthor-10k", house_idx=int(hi),
                            motion="goal_directed", step_std=np.nan, angle_std=np.nan, camera_height_m=1.454, hfov_deg=90.0,
                            vfov_deg=73.739795, time_of_day=rm["time_of_day"], skybox=rm["skybox"], n_rooms=int(rm["n_rooms"]),
                            room_types=json.dumps(rm["room_types"]), n_objects=int(rm["n_objects"]), n_windows=int(rm["n_windows"]),
                            n_doors=int(rm["n_doors"]), wall_materials=json.dumps(rm["wall_materials"]),
                            floor_materials=json.dumps(rm["floor_materials"]), n_reachable=int(pm["n_reachable"]),
                            reachable_set="furnished house, shared by both cells", paired_trajectories=True,
                            stalls=int((np.linalg.norm(np.diff(plan["pos_u"][j], axis=0), axis=1) < 1e-6).sum()),
                            ai2thor="5.0.0", n_steps=N_STEPS,
                            walkable_m2=float(pm["walkable_m2"]), goal_min_m=float(pw["goal_min_m"]), goals=int(st["goals"]),
                            legs=int(st["legs"]), replans=int(st["replans"]), uturns=int(st["uturns"]), fixes=int(st["fixes"]),
                            path_m=float(ck["path_m"]),
                            door_crossings=door_stats(plan["pos_u"][j], N_STEPS, polys)["door_crossings"])
                for c in CELLS:
                    rows[c].append(base | dict(mean_brightness=float(rw["mean_brightness"][c]),
                                               env_name=f"procthor10k_{folder}_{hi:05d}_goalwalks_{c}"))
        print(f"{folder}: {sum(r['folder'] == folder for r in rows['objects'])} walks", flush=True)
    rtq = pd.read_parquet(WALKS_RTQ)
    for c in CELLS:
        df = pd.DataFrame(rows[c]); df.insert(0, "record_idx", range(len(df)))
        out = a.work / dataset(c); (out / "index").mkdir(parents=True, exist_ok=True); (out / "splits").mkdir(exist_ok=True)
        df.to_parquet(out / "index" / "walks.parquet", index=False)
        df[["clip_id"]].assign(split=df["folder"]).to_parquet(out / "splits" / "v1.parquet", index=False)
        if set(rtq["clip_id"]) != set(df["clip_id"]):
            sys.exit(f"{dataset(c)}: clip_ids differ from procthor-walks rtq-r160 "
                     f"({len(set(rtq['clip_id']) - set(df['clip_id']))} missing, {len(set(df['clip_id']) - set(rtq['clip_id']))} extra)")
        rtq.set_index("clip_id").loc[df["clip_id"]].reset_index().to_parquet(out / "splits" / "rtq-r160.parquet", index=False)
        print(f"{dataset(c)}: {len(df)} walks; splits v1 + rtq-r160 (copied from procthor-walks)", flush=True)
    return 0


class Source:
    def __init__(self, paths: list[str], cell: str, out: Path):
        self.paths, self.cell, self.out = paths, cell, out

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
        z = np.load(self.paths[i]); fr = z[self.cell]
        if fr.shape != (N_STEPS, H, W, 3):
            raise ValueError(f"{self.paths[i]}: {fr.shape}")
        return {"video": encode_walk(fr), "positions": z["pos_h"].astype(np.float32), "headings": z["head_h"].astype(np.float32),
                "fps": float(FPS), "num_frames": N_STEPS, "duration_s": N_STEPS / FPS}


def _records(a, folder) -> pd.DataFrame:
    w = pd.read_parquet(a.work / dataset(a.cell) / "index" / "walks.parquet")
    w = w[w["folder"] == folder].rename(columns={"record_idx": "walk_idx"}).reset_index(drop=True)
    w.insert(0, "record_idx", range(len(w)))
    return w


def cmd_encode(a) -> int:
    from slipstream.cache import OptimizedCache
    rec = _records(a, a.folder); store = a.store_root / store_name(a.cell, a.folder)
    if store.exists() and any(store.iterdir()):
        sys.exit(f"{store} exists and is not empty; remove it to rebuild")
    print(f"{store.name}: encoding {len(rec):,} walks, {a.workers} workers, {ENCODE}", flush=True)
    t = time.perf_counter()
    cache = OptimizedCache.build(Source(rec["src_path"].tolist(), a.cell, store), output_dir=store, verbose=True, num_workers=a.workers)
    if len(cache) != len(rec):
        sys.exit(f"store has {len(cache)} records, expected {len(rec)}")
    rec.to_parquet(store / "records.parquet", index=False)
    vbytes = sum(p.stat().st_size for p in store.glob("video*") if p.is_file())
    sm = dict(dataset=dataset(a.cell), store=store.name, folder=a.folder, num_records=len(rec), cell=a.cell,
              source="procthor_routes.goalwalks render", encode=ENCODE, fields=Source([], a.cell, store).field_types,
              software=software_versions(), video_bytes=vbytes, encode_seconds=round(time.perf_counter() - t),
              built=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (store / "store_manifest.json").write_text(json.dumps(sm, indent=1))
    print(f"BUILT {store.name}: {len(rec):,} records, video {vbytes / 1e9:.2f} GB", flush=True)
    return 0


def _psnr(x, y) -> float:
    mse = float(np.mean((x.astype(np.float32) - y.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def cmd_verify(a) -> int:
    import torch
    torch.set_num_threads(1)
    from slipstream.cache import OptimizedCache
    from torchcodec.decoders import VideoDecoder
    store = a.store_root / store_name(a.cell, a.folder)
    cache = OptimizedCache.load(store, verbose=False); rec = pd.read_parquet(store / "records.parquet")
    fails = [] if len(cache) == len(rec) else [f"store {len(cache)} records, records.parquet {len(rec)}"]
    if rec["clip_id"].tolist() != _records(a, a.folder)["clip_id"].tolist():
        fails.append("records.parquet order differs from the index")
    vf, pf, hf = cache.fields["video"], cache.fields["positions"], cache.fields["headings"]

    def read(f, i):
        out = f.load_batch(np.array([i], dtype=np.int64), parallel=False)
        return {k: np.array(v, copy=True) for k, v in out.items()}

    def blob(i):
        o = read(vf, i)
        return bytes(o["data"][0][: int(o["sizes"][0])])

    def one(i):
        m = VideoDecoder(blob(i), seek_mode="exact").metadata
        if (m.num_frames, m.width, m.height) != (N_STEPS, W, H) or abs((m.average_fps or 0) - FPS) > 1e-3:
            return f"record {i}: {m.num_frames} frames {m.width}x{m.height} fps {m.average_fps}"
        z = np.load(rec["src_path"][i])
        if read(pf, i)["data"][0].tobytes() != z["pos_h"].astype(np.float32).tobytes() or \
                read(hf, i)["data"][0].tobytes() != z["head_h"].astype(np.float32).tobytes():
            return f"record {i}: poses differ"
        return None

    with ThreadPoolExecutor(a.threads) as ex:
        fails += [r for r in ex.map(one, range(len(rec))) if r][:10]
    rng = np.random.default_rng(0); pick = rng.choice(len(rec), min(a.windows, len(rec)), replace=False)
    starts = rng.integers(0, 16, len(pick)) * 60

    def win(args):
        i, s0 = args
        got = VideoDecoder(blob(i), seek_mode="exact").get_frames_in_range(int(s0), int(s0) + LEN_WINDOW).data.permute(0, 2, 3, 1).numpy()
        return _psnr(got, np.load(rec["src_path"][i])[a.cell][s0:s0 + LEN_WINDOW])

    with ThreadPoolExecutor(a.threads) as ex:
        ps = np.array(list(ex.map(win, zip(pick, starts))))
    if ps.mean() < 40 or ps.min() < 33:
        fails.append(f"PSNR mean {ps.mean():.2f} min {ps.min():.2f}")
    v = dict(ok=not fails, records=len(rec), poses_checked=len(rec), windows=len(ps), psnr_mean=float(ps.mean()),
             psnr_min=float(ps.min()), failures=fails, verified=time.strftime("%Y-%m-%dT%H:%M:%S"))
    sm = json.loads((store / "store_manifest.json").read_text()); sm["verify"] = v
    (store / "store_manifest.json").write_text(json.dumps(sm, indent=1))
    print("VERIFY " + ("OK " if v["ok"] else "FAILED ") + json.dumps(v), flush=True)
    return 0 if v["ok"] else 1


def cmd_pair(a) -> int:
    from slipstream.cache import OptimizedCache
    st = {c: a.store_root / store_name(c, a.folder) for c in CELLS}
    r = {c: pd.read_parquet(st[c] / "records.parquet") for c in CELLS}
    same = r["objects"]["clip_id"].tolist() == r["empty"]["clip_id"].tolist()
    f = {c: OptimizedCache.load(st[c], verbose=False).fields for c in CELLS}
    idx = np.arange(len(r["objects"]), dtype=np.int64)
    eq = {k: np.asarray(f["objects"][k].load_batch(idx, parallel=False)["data"]).tobytes()
          == np.asarray(f["empty"][k].load_batch(idx, parallel=False)["data"]).tobytes() for k in ("positions", "headings")}
    ok = same and all(eq.values())
    print(f"PAIR {'OK' if ok else 'FAILED'} {a.folder}: {len(idx)} records, same clip_ids {same}, identical {eq}", flush=True)
    return 0 if ok else 1


def ego(pos: np.ndarray, head: np.ndarray) -> np.ndarray:
    """procthor-walks card convention: (dx_ego, dy_ego, dtheta) between consecutive frames."""
    step = pos[1:] - pos[:-1]; c, s = np.cos(head[:-1]), np.sin(head[:-1])
    return np.stack([step[:, 0] * c - step[:, 1] * s, step[:, 0] * s + step[:, 1] * c,
                     (np.diff(head) + np.pi) % (2 * np.pi) - np.pi], -1)


def cmd_stats(a) -> int:
    """RGB mean/std per cell and ego-motion mean/std over all train walks (float32 poses, as stored)."""
    acc = {c: [0, np.zeros(3), np.zeros(3)] for c in CELLS}; egos = []
    for rj in sorted((a.render / "train").glob("house_*/render.json")):
        m = json.loads(rj.read_text())
        for w in m["walks"]:
            for c in CELLS:
                acc[c][0] += w["rgb"][c]["n"]; acc[c][1] += w["rgb"][c]["sum"]; acc[c][2] += w["rgb"][c]["sumsq"]
            z = np.load(rj.parent / f"seed_{w['seed']}.npz")
            egos.append(ego(z["pos_h"].astype(np.float32).astype(np.float64), z["head_h"].astype(np.float32).astype(np.float64)))
    e = np.concatenate(egos)
    out = {"rgb": {dataset(c): {"mean": (acc[c][1] / acc[c][0]).tolist(),
                                "std": np.sqrt(acc[c][2] / acc[c][0] - (acc[c][1] / acc[c][0]) ** 2).tolist(), "pixels": int(acc[c][0])}
                   for c in CELLS},
           "egomotion": {"mean": e.mean(0).tolist(), "std": e.std(0).tolist(), "steps": int(len(e))}}
    (a.work / "stats_train.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)
    return 0


def cmd_publish(a) -> int:
    store = a.store_root / store_name(a.cell, a.folder); ds = dataset(a.cell); splits = a.work / ds / "splits"
    sm = json.loads((store / "store_manifest.json").read_text())
    if not (store / "manifest.json").exists() or not sm.get("verify", {}).get("ok"):
        sys.exit(f"{store}: not a finished, verified store")

    def run(cmd):
        print("+", " ".join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=True)
    s3 = f"{S3_BASE}/{ds}"
    run(["rsync", "-a", "--chmod=Dg+rwxs,Fg+rw", f"{store}/", f"{LAB_BASE / store.name}/"])
    run(["mkdir", "-p", str(LAB_BASE / ds / "splits")])
    run(["rsync", "-a", "--chmod=Dg+rwxs,Fg+rw", f"{splits}/", f"{LAB_BASE / ds / 'splits'}/"])
    run([a.s5cmd, "sync", "--concurrency", "8", f"{store}/", f"{s3}/{store.name}/"])
    run([a.s5cmd, "sync", f"{splits}/", f"{s3}/splits/"])
    url = f"https://visionlab-datasets.s3.amazonaws.com/slipstream-cache/{ds}/{store.name}/manifest.json"
    if not a.dry_run:
        code = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url], capture_output=True, text=True).stdout
        print(f"anonymous GET {url} -> HTTP {code} ({'private, OK' if code in ('403', '301') else 'CHECK ACLS'})", flush=True)
        if code not in ("403", "301"):
            return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("index")
    for k in ("plan", "render", "houses-root", "work"):
        i.add_argument(f"--{k}", type=Path, required=True)
    for name in ("encode", "verify", "publish"):
        p = sub.add_parser(name); p.add_argument("--work", type=Path, required=True); p.add_argument("--cell", choices=CELLS, required=True)
        p.add_argument("--folder", choices=FOLDERS, required=True); p.add_argument("--store-root", type=Path, required=True)
        p.add_argument("--workers", type=int, default=16); p.add_argument("--threads", type=int, default=16)
        p.add_argument("--windows", type=int, default=256); p.add_argument("--s5cmd", default="s5cmd"); p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("pair"); p.add_argument("--store-root", type=Path, required=True); p.add_argument("--folder", choices=FOLDERS, required=True)
    s = sub.add_parser("stats"); s.add_argument("--work", type=Path, required=True); s.add_argument("--render", type=Path, required=True)
    a = ap.parse_args(argv)
    return {"index": cmd_index, "encode": cmd_encode, "verify": cmd_verify, "pair": cmd_pair, "stats": cmd_stats,
            "publish": cmd_publish}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
