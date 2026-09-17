"""Run the encode stage on the lab fleet: every host claims groups from the shared QNAP work dir (`encode --claim`).

    uv run python -m datasets.prep.spatialvid_hq.fleet launch --fps 30 [--res 640x360,456x256] [--hosts thrace,vesper,...]
    uv run python -m datasets.prep.spatialvid_hq.fleet status --fps 30          # runs status.py on machina
    uv run python -m datasets.prep.spatialvid_hq.fleet ps                       # encode processes per host
    uv run python -m datasets.prep.spatialvid_hq.fleet stop [--hosts ...]       # kill encode on hosts (claims of unfinished groups must be removed by hand)
    uv run python -m datasets.prep.spatialvid_hq.fleet tail [--hosts ...]       # last log lines per host

Runs from any machine with ssh access to the hosts. Each host runs the `jupyter-grez72` container with the repo at
~/work/GitHub/datasets and the QNAP Flash share mounted (path differs per host, see HOSTS). Per host: `git pull`
(clone + `uv sync --group video` if missing), then `nohup encode --groups all --claim --fps F --tmp <local NVMe>`
with the log on the QNAP (`<out>/logs/encode_<host>_<axis>.log`), so `status.py` sees every host. Stopping and
relaunching is safe: finished groups are skipped, a group claimed by a killed host stays claimed until its
`.claim` file is removed (status.py lists 'claimed but unfinished').
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

CONTAINER = "jupyter-grez72"
REPO = "~/work/GitHub/datasets"
REPO_URL = "git@github.com:harvard-visionlab/datasets.git"
DATASETS_REL = "DataSets/VideoDatasets"
# host -> QNAP Flash root inside the container
HOSTS = {
    "machina": "~/work/DataExactitudeFlash",
    "thrace": "~/work/DataRemote/qnap/exactitude/Flash",
    "vesper": "~/work/DataRemote/qnap/exactitude/Flash",
    "leeloo": "~/work/DataRemote/qnap/exactitude/Flash",
    "stelline": "~/work/DataRemote/qnap/exactitude/Flash",
}
TMP = "~/work/DataLocal/tmp/spatialvid"


def ssh(host: str, script: str, detach: bool = False, timeout: int = 600) -> subprocess.CompletedProcess:
    inner = f"docker exec {'-d ' if detach else ''}{CONTAINER} bash -lc {shlex.quote(script)}"
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, inner], capture_output=True, text=True, timeout=timeout)


def paths(host: str) -> tuple[str, str]:
    root = HOSTS[host]
    return f"{root}/{DATASETS_REL}/SpatialVID-HQ", f"{root}/{DATASETS_REL}/SpatialVID-HQ-slipstream"


def axis(res: str, fps: int | None) -> str:
    return "-".join(r for r in res.split(",")) + (f"-{fps}fps" if fps else "")


def launch(hosts: list[str], fps: int | None, res: str, workers: int | None, extra: str) -> None:
    for h in hosts:
        raw, out = paths(h)
        log = f"{out}/logs/encode_{h}_{axis(res, fps)}.log"
        bootstrap = (f"mkdir -p {TMP} {out}/logs && "
                     f"if [ ! -d {REPO}/.git ]; then git clone -q {REPO_URL} {REPO}; fi && cd {REPO} && git pull -q --ff-only && "
                     f"if [ ! -d .venv ]; then uv sync -q --group video; fi && git log --oneline -1")
        r = ssh(h, bootstrap)
        if r.returncode:
            print(f"[{h}] bootstrap FAILED: {r.stderr.strip()[-400:]}"); continue
        print(f"[{h}] repo at {r.stdout.strip().splitlines()[-1]}")
        cmd = (f"cd {REPO} && nohup uv run --no-sync --group video python -m datasets.prep.spatialvid_hq.encode --raw {raw} --out {out} "
               f"--res {res} --groups all --claim --tmp {TMP}" + (f" --fps {fps}" if fps else "") + (f" --workers {workers}" if workers else "")
               + (f" {extra}" if extra else "") + f" > {log} 2>&1 &")
        r = ssh(h, cmd, detach=True)
        print(f"[{h}] {'launched' if r.returncode == 0 else 'launch FAILED: ' + r.stderr.strip()[-300:]} -> {log}")


def ps(hosts: list[str]) -> None:
    for h in hosts:
        r = ssh(h, "pgrep -af 'spatialvid_hq.encode' | grep -v pgrep | head -3; echo \"ffmpeg procs: $(pgrep -c -x ffmpeg)\"; uptime")
        print(f"[{h}]\n{(r.stdout or r.stderr).strip()}")


def stop(hosts: list[str]) -> None:
    for h in hosts:
        r = ssh(h, "pkill -f 'spatialvid_hq.encode'; sleep 1; pkill -x ffmpeg; echo stopped")
        print(f"[{h}] {(r.stdout or r.stderr).strip()}")


def tail(hosts: list[str], fps: int | None, res: str, n: int) -> None:
    for h in hosts:
        _raw, out = paths(h)
        r = ssh(h, f"tail -n {n} {out}/logs/encode_{h}_{axis(res, fps)}.log 2>&1")
        print(f"[{h}]\n{(r.stdout or r.stderr).strip()}")


def status(fps: int | None, res: str) -> None:
    _raw, out = paths("machina")
    r = ssh("machina", f"cd {REPO} && uv run --no-sync --group video python -m datasets.prep.spatialvid_hq.status --out {out} --res {res}" + (f" --fps {fps}" if fps else ""))
    print((r.stdout or "") + (r.stderr or ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["launch", "status", "ps", "stop", "tail"])
    ap.add_argument("--hosts", default=",".join(HOSTS)); ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--res", default="640x360,456x256"); ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--extra", default="", help="extra args passed to encode (e.g. '--limit 5')"); ap.add_argument("-n", type=int, default=5)
    a = ap.parse_args(argv)
    hosts = [h for h in a.hosts.split(",") if h]
    unknown = [h for h in hosts if h not in HOSTS]
    if unknown:
        print(f"unknown hosts {unknown}; known {list(HOSTS)}", file=sys.stderr); return 2
    if a.cmd == "launch": launch(hosts, a.fps, a.res, a.workers, a.extra)
    elif a.cmd == "ps": ps(hosts)
    elif a.cmd == "stop": stop(hosts)
    elif a.cmd == "tail": tail(hosts, a.fps, a.res, a.n)
    else: status(a.fps, a.res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
