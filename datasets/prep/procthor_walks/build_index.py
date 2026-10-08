"""Stage 1: index the source walks of one dataset and write its split tables.

    python -m datasets.prep.procthor_walks.build_index --dataset procthor-walks-objects --out <work>

Writes
  <work>/<dataset>/index/walks.parquet   one row per walk, in store order (record_idx): clip_id, folder, house, seed,
                                         src_path, and the generator metadata of every walk (rooms, objects, materials, ...)
  <work>/<dataset>/splits/v1.parquet     clip_id -> split = source folder (train / val / test)
  <work>/<dataset>/splits/rtq-r160.parquet  clip_id -> the split the student's r160 runs used:
      train (18,000 walks) and val (2,000 held-out walks) both come from the train folder (their window lists),
      val_unseen = the first 400 walks (sorted paths) of the val folder (their held-out-houses set),
      test = the test folder, unused = the rest of the val folder.
  plus a .report.md per split table.
Checks that every run of the cell (4 settings x {active, passive}) used the same window lists, and that every listed
walk is windowed at starts 0, 60, ..., 900 (16 windows of 61 frames).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .common import DATASETS, FOLDERS, RUN_SETTINGS, SRC_DATA, SRC_RUNS, clip_id

META_COLS = ("source", "house_idx", "motion", "step_std", "angle_std", "camera_height_m", "hfov_deg", "vfov_deg",
             "time_of_day", "skybox", "n_rooms", "room_types", "n_objects", "n_windows", "n_doors", "wall_materials",
             "floor_materials", "n_reachable", "reachable_set", "paired_trajectories", "stalls", "mean_brightness", "ai2thor")


def walk_meta(path: Path) -> dict:
    import h5py
    import hdf5plugin  # noqa: F401
    from .common import read_meta
    with h5py.File(path, "r") as f:
        m = read_meta(f)
        n = f["trajectory"]["agent_positions"].shape[0]
        env = f["meta"]["env_name"][()]
    out = {k: (json.dumps(m[k]) if isinstance(m.get(k), (list, dict)) else m.get(k)) for k in META_COLS}
    out["n_steps"] = int(n)
    out["env_name"] = env.decode() if isinstance(env, bytes) else str(env)
    return out


def read_windows(run: str, split: str) -> list[tuple[str, int]]:
    with open(SRC_RUNS / run / "active_reproducibility" / f"{split}_windows.tsv") as fh:
        return [(r["filepath"], int(r["start"])) for r in csv.DictReader(fh, delimiter="\t")]


def report(df: pd.DataFrame, name: str, note: str) -> str:
    c = df["split"].value_counts().reindex(["train", "val", "val_unseen", "test", "unused"]).dropna().astype(int)
    rows = "\n".join(f"| {k} | {v:,} |" for k, v in c.items())
    return f"# {name}\n\n{note}\n\n| split | walks |\n| --- | --- |\n{rows}\n| total | {len(df):,} |\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    cell = DATASETS[a.dataset]["cell"]
    out = a.out / a.dataset
    (out / "index").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    rows = []
    for folder in FOLDERS:
        d = SRC_DATA / cell / folder
        for house in sorted(p.name for p in d.iterdir() if p.is_dir()):
            for f in sorted((d / house).glob("seed_*.h5")):
                rows.append(dict(clip_id=clip_id(folder, house, f.stem), folder=folder, house=house,
                                 seed=int(f.stem.split("_")[1]), src_path=str(f)))
    print(f"{a.dataset}: {len(rows):,} walks; reading metadata with {a.threads} threads", flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        metas = list(ex.map(lambda r: walk_meta(Path(r["src_path"])), rows))
    df = pd.DataFrame([{**r, **m} for r, m in zip(rows, metas)])
    df.insert(0, "record_idx", range(len(df)))
    assert (df["n_steps"] == 1000).all(), df.loc[df["n_steps"] != 1000, "clip_id"].tolist()[:5]
    df.to_parquet(out / "index" / "walks.parquet", index=False)

    v1 = df[["clip_id"]].assign(split=df["folder"])
    v1.to_parquet(out / "splits" / "v1.parquet", index=False)
    (out / "splits" / "v1.report.md").write_text(report(v1, f"{a.dataset} splits v1", "Split = the source folder."))

    # the student's r160 runs: every run of this cell must share one window list
    lists = {}
    for s in RUN_SETTINGS:
        for twin in ("", "__passive"):
            run = f"{cell}__{s}_r160{twin}"
            lists[run] = {sp: read_windows(run, sp) for sp in ("train", "val")}
    ref_run = f"{cell}__bridge_r160"
    ref = {sp: set(w) for sp, w in lists[ref_run].items()}
    for run, w in lists.items():
        for sp in ("train", "val"):
            if set(w[sp]) != ref[sp]:
                sys.exit(f"window list of {run}/{sp} differs from {ref_run}/{sp}")
    by_path = {p: c for p, c in zip(df["src_path"], df["clip_id"])}
    split = {}
    for sp in ("train", "val"):
        starts: dict[str, list[int]] = {}
        for p, s in lists[ref_run][sp]:
            starts.setdefault(p, []).append(s)
        for p, ss in starts.items():
            if sorted(ss) != list(range(0, 901, 60)):
                sys.exit(f"{ref_run}/{sp}: unexpected window starts for {p}: {sorted(ss)[:5]}...")
            split[by_path[p]] = sp
    val_folder = df[df["folder"] == "val"].sort_values("src_path")
    for c in val_folder["clip_id"].iloc[:400]:
        split[c] = "val_unseen"
    for c in df.loc[df["folder"] == "test", "clip_id"]:
        split[c] = "test"
    rtq = df[["clip_id"]].assign(split=df["clip_id"].map(split).fillna("unused"))
    rtq.to_parquet(out / "splits" / "rtq-r160.parquet", index=False)
    note = (f"The windows of the student's r160 runs ({', '.join(sorted(lists))}; identical lists, checked). train/val: "
            "walks of the train folder, windowed at starts 0, 60, ..., 900 (16 x 61 frames per walk). val_unseen: first "
            "400 sorted val-folder walks (held-out houses). test: test folder. unused: the other val-folder walks.")
    (out / "splits" / "rtq-r160.report.md").write_text(report(rtq, f"{a.dataset} splits rtq-r160", note))
    print(report(v1, "v1", "") + "\n" + report(rtq, "rtq-r160", ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
