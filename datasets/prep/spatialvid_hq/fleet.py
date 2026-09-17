"""Run the encode stage on the lab fleet: every host claims groups from the shared QNAP work dir (`encode --claim`).

    uv run python -m datasets.prep.spatialvid_hq.fleet launch --fps 30 [--res 640x360,456x256] [--hosts thrace,vesper,...]
    uv run python -m datasets.prep.spatialvid_hq.fleet sync [--hosts ...]       # fallback: copy this checkout into the containers as jovyan (launch normally git-pulls)
    uv run python -m datasets.prep.spatialvid_hq.fleet launch --fps 15 --after-fps 30   # per host: start once its 30 fps encode exits
    uv run python -m datasets.prep.spatialvid_hq.fleet launch --fps 30 --extra --retry-failed --log-tag _retry   # patch pass for failed clips (loops until covered)
    uv run python -m datasets.prep.spatialvid_hq.fleet finish --fps-list 30,15   # on machina: merge + S3 sync each axis when its 74 shards exist and failures are patched
    uv run python -m datasets.prep.spatialvid_hq.fleet status --fps 30          # runs status.py on machina
    uv run python -m datasets.prep.spatialvid_hq.fleet ps                       # encode processes per host
    uv run python -m datasets.prep.spatialvid_hq.fleet stop [--hosts ...]       # kill encode on hosts (claims of unfinished groups must be removed by hand)
    uv run python -m datasets.prep.spatialvid_hq.fleet tail [--hosts ...]       # last log lines per host

Runs from any machine with ssh access to the hosts. Each host runs the `jupyter-grez72` container with the repo at
~/work/GitHub/datasets and the QNAP Flash share mounted (path differs per host, see HOSTS). Everything runs as the
container user `jovyan` (`docker exec -u jovyan`): the fleet hosts use docker userns-remap, so a bare `docker exec`
is root-in-namespace, which owns nothing (DataLocal, the repo, GitHub keys all belong to jovyan). Per host: `git pull`
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
CONTAINER_USER = "jovyan"
REPO = "~/work/GitHub/datasets"
REPO_URL = "git@github.com:harvard-visionlab/datasets.git"
DATASETS_REL = "DataSets/VideoDatasets"
# host -> (QNAP Flash root inside the container, local scratch for per-clip temp files). machina is the older
# machine with its own mount layout; the fleet hosts share one.
FLEET_FLASH = "~/work/DataRemote/qnap/exactitude/Flash"
SCRATCH = "~/work/DataLocal/tmp/spatialvid"
HOSTS = {
    "machina": ("~/work/DataExactitudeFlash", SCRATCH),
    "thrace": (FLEET_FLASH, SCRATCH),
    "vesper": (FLEET_FLASH, SCRATCH),
    "leeloo": (FLEET_FLASH, SCRATCH),
    "stelline": (FLEET_FLASH, SCRATCH),
}


def ssh(host: str, script: str, detach: bool = False, timeout: int = 600) -> subprocess.CompletedProcess:
    inner = f"docker exec {'-d ' if detach else ''}-u {CONTAINER_USER} {CONTAINER} bash -lc {shlex.quote(script)}"
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, inner], capture_output=True, text=True, timeout=timeout)


def paths(host: str) -> tuple[str, str]:
    root = HOSTS[host][0]
    return f"{root}/{DATASETS_REL}/SpatialVID-HQ", f"{root}/{DATASETS_REL}/SpatialVID-HQ-slipstream"


def axis(res: str, fps: int | None) -> str:
    return "-".join(r for r in res.split(",")) + (f"-{fps}fps" if fps else "")


def launch(hosts: list[str], fps: int | None, res: str, workers: int | None, extra: str, after_fps: int | None = None, log_tag: str = "") -> None:
    """`after_fps`: start only once this host's running encode for that fps axis has exited (chains passes per host)."""
    for h in hosts:
        raw, out = paths(h); tmp = HOSTS[h][1]
        log = f"{out}/logs/encode_{h}_{axis(res, fps)}.log"
        # pull is best-effort (a missing nbstripout filter makes git refuse to touch notebooks); `fleet sync` rsyncs this tree as a fallback
        bootstrap = (f"ls {raw}/videos/group_0001.tar.gz {out}/index/clips.parquet > /dev/null && mkdir -p {tmp} {out}/logs && "
                     f"if [ ! -d {REPO}/.git ]; then git clone -q {REPO_URL} {REPO}; fi && cd {REPO} && (git pull -q --ff-only 2>/dev/null || echo 'pull failed; using the tree as is') && "
                     f"if [ ! -d .venv ]; then uv sync -q --group video; fi && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx265 && git log --oneline -1")
        r = ssh(h, bootstrap, timeout=1800)
        if r.returncode:
            print(f"[{h}] bootstrap FAILED: {(r.stderr or r.stdout).strip()[-400:]}"); continue
        print(f"[{h}] repo at {r.stdout.strip().splitlines()[-1]}")
        wait = (f"while pgrep -f 'spatialvid_hq.encode .*--fps {after_fps}( |$)' > /dev/null; do sleep 60; done; " if after_fps else "")
        cmd = (f"cd {REPO} && {wait}FLEET_HOST={h} nohup uv run --no-sync --group video python -m datasets.prep.spatialvid_hq.encode --raw {raw} --out {out} "
               f"--res {res} --groups all --claim --tmp {tmp}" + (f" --fps {fps}" if fps else "") + (f" --workers {workers}" if workers else "")
               + (f" {extra}" if extra else "") + f" > {log} 2>&1 &")
        if after_fps:
            cmd = f"nohup bash -c {shlex.quote(cmd)} > /dev/null 2>&1 &"
        r = ssh(h, cmd, detach=True)
        print(f"[{h}] {'launched' if r.returncode == 0 else 'launch FAILED: ' + r.stderr.strip()[-300:]} -> {log}")


def finish(fps_list: str, res: str) -> None:
    """Start finish.py on machina: merges + S3-syncs each axis once all its shards exist."""
    _raw, out = paths("machina")
    cmd = (f"cd {REPO} && nohup uv run --no-sync --group video python -m datasets.prep.spatialvid_hq.finish --out {out} --res {res} --fps {fps_list} "
           f"> {out}/logs/finish.log 2>&1 &")
    r = ssh("machina", cmd, detach=True)
    print(f"[machina] finisher {'launched' if r.returncode == 0 else 'FAILED: ' + r.stderr[-300:]} -> {out}/logs/finish.log")


def sync(hosts: list[str]) -> None:
    """Fallback when `git pull` cannot run in the container: rsync this checkout (minus .venv) into the container as jovyan
    (`docker exec -i -u jovyan ... tar x`), so ownership stays with the container user. Prefer `git pull` (launch does it)."""
    import os
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    tar = ["tar", "-C", src, "--exclude", ".venv", "--exclude", "__pycache__", "--exclude", "*.egg-info", "--exclude", ".pytest_cache", "-cf", "-", "."]
    for h in hosts:
        untar = f"docker exec -i -u {CONTAINER_USER} {CONTAINER} bash -lc {shlex.quote(f'mkdir -p {REPO} && tar -C {REPO} -xf -')}"
        p1 = subprocess.Popen(tar, stdout=subprocess.PIPE)
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", h, untar], stdin=p1.stdout, capture_output=True, text=True); p1.wait()
        print(f"[{h}] sync {'ok' if r.returncode == 0 else 'FAILED: ' + r.stderr[-300:]}")


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
    ap.add_argument("cmd", choices=["launch", "sync", "finish", "status", "ps", "stop", "tail"])
    ap.add_argument("--after-fps", type=int, default=None, help="launch: wait until the running encode of this fps axis exits (chain passes)")
    ap.add_argument("--fps-list", default="30,15", help="finish: axes to merge + sync when complete")
    ap.add_argument("--log-tag", default="", help="launch: suffix for the log file name (e.g. _retry), so a second job does not truncate the first job's log")
    ap.add_argument("--hosts", default=",".join(HOSTS)); ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--res", default="640x360,456x256"); ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--extra", default="", help="extra args passed to encode (e.g. '--limit 5')"); ap.add_argument("-n", type=int, default=5)
    a = ap.parse_args(argv)
    hosts = [h for h in a.hosts.split(",") if h]
    unknown = [h for h in hosts if h not in HOSTS]
    if unknown:
        print(f"unknown hosts {unknown}; known {list(HOSTS)}", file=sys.stderr); return 2
    if a.cmd == "launch": launch(hosts, a.fps, a.res, a.workers, a.extra, a.after_fps, a.log_tag)
    elif a.cmd == "finish": finish(a.fps_list, a.res)
    elif a.cmd == "sync": sync(hosts)
    elif a.cmd == "ps": ps(hosts)
    elif a.cmd == "stop": stop(hosts)
    elif a.cmd == "tail": tail(hosts, a.fps, a.res, a.n)
    else: status(a.fps, a.res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
