---
name: visionlab-datasets
description: Lab dataset caches (slipstream) with the visionlab-datasets CLI. Where caches live on each machine (FAS cluster netscratch / lab_storage, workstations, S3), how to point SLIPSTREAM_CACHE_DIR at them, how to check a cache (path, status) and refill it (sync), and what is safe on a cluster login node. Use when someone sets up or debugs dataset loading, asks which command checks or downloads a cache, or when S3 traffic/costs from dataset reads come up (ImageNet, imagenet100, procthor-walks, SpatialVID, "slipstream cache", "visionlab-datasets").
---

# visionlab-datasets

Canonical copy: `skills/visionlab-datasets/SKILL.md` in harvard-visionlab/datasets. Every `visionlab-datasets` command
in this file and in the repo README is parsed by the test suite (`tests/test_docs.py`), so the syntax here is correct
for the release it ships with. For anything not below: the README (section "On the FAS cluster: quickstart", then
"Command line") or `visionlab-datasets <command> --help`.

## Where caches live

| place | dir | notes |
|---|---|---|
| FAS cluster, training | `/n/netscratch/alvarez_lab/Lab/datasets/slipstream` | shared by the lab; holds every registered cache; purged after ~90 days unused |
| FAS cluster, master | `/n/lab_storage/alvarez_lab/Lab/datasets/slipstream` | durable but slow NFS; the source for refills, never train from it |
| machina | `~/work/DataScratch/cache/slipstream-cache` | set in jovyan's `.profile` |
| other workstations | `~/.slipstream` | per-platform default |
| S3 | `s3://visionlab-datasets/slipstream-cache/` | private; egress costs money (~$56 per ImageNet-1k pass) |

The cache dir is `SLIPSTREAM_CACHE_DIR` if set, else a per-platform default. On the cluster the default is the
lab_storage master, so **set the variable** (in `~/.bashrc`; sbatch jobs inherit the submitting shell's environment):

```bash
export SLIPSTREAM_CACHE_DIR=/n/netscratch/alvarez_lab/Lab/datasets/slipstream
```

A job that reads from S3 every epoch, or syncs into a per-job `/tmp`, means this variable is missing.

Purges are handled by `load()` (since 0.21.0): each call size-checks the cache (milliseconds) and refills only what's
missing, from the master copy (lab_storage on the cluster; `VISIONLAB_DATASETS_MASTER` overrides, empty = none),
else from S3 with a printed size and cost estimate. A "downloading ... from S3" line on the cluster means the master
lacks that cache.

## Commands

```bash
uv run --no-sync visionlab-datasets path imagenet1k train,val jpeg       # ✓/✗ per cache + its path (fast, offline)
uv run --no-sync visionlab-datasets status imagenet1k --no-remote        # fuller report for one dataset
uv run --no-sync visionlab-datasets status                               # every dataset + S3 checks (slow-ish)
uv run --no-sync visionlab-datasets list                                 # registered datasets and S3 caches
# refill on the cluster from the master (in a job):
uv run --no-sync visionlab-datasets sync imagenet1k train,val jpeg \
    --dest /n/netscratch/alvarez_lab/Lab/datasets/slipstream \
    --source /n/lab_storage/alvarez_lab/Lab/datasets/slipstream
```

- Arguments: `<dataset> <splits> <fmt>`. Splits: `train`, `val`, comma lists or `all`. Formats: `jpeg`, `yuv420`,
  `original` (ImageNet val as distributed; not in `all`, which means `jpeg,yuv420`), comma lists or `all`. The format
  goes positionally or as `--fmt` (since 0.20.3; older releases need `path ... --fmt jpeg` and `sync ... jpeg`).
  Aliases: `in10`, `in100`, `in1k`, `in100_s292`.
- `sync` copies only missing or wrong-size files, is safe on shared dirs (lock, stage, verify, manifest last), and
  falls back to S3 per cache when the `--source` copy is damaged. `--deep` also sha256-checks what is present.
- Python: `from visionlab.datasets import load, cache_identity, sync_store`.

## Cluster login node

- OK: `path`, `list`, `status <dataset> --no-remote`.
- In a job (`shared`, 1-2 cores; it waits on I/O): `sync`, anything with `--deep`, full `status` if it is slow.
- Always `uv run --no-sync` (plain `uv run` may install). If `uv run` hangs before any output, the node's NFS
  locking is broken (uv locks its cache dir first; all login nodes 2026-11-04, fixed 2026-11-07; tell rchelp if it
  recurs): call `$LAB_SCRATCH/venvs/<repo>/bin/visionlab-datasets` directly or use a job.
- Don't wrap `uv` in `timeout`/`xargs`: that bypasses the lab's `uv` shell function (venv on netscratch) and creates
  `.venv` in the holylabs checkout.

## Owner

The `datasets` Claude session (repo harvard-visionlab/datasets) owns the registry, the caches and their sync and
hashing. New datasets, rebuilt caches, or cache problems go there.
