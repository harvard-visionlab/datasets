# visionlab-datasets

Visionlab dataset utilities for streaming datasets.

## Installation

datasets requires `visionlab-slipstream>=0.9,<0.17` but doesn't pin a slipstream tag, and slipstream
isn't on PyPI. **Pin slipstream yourself in the same install** (any tag in range); installing without
it fails to resolve.

```bash
pip install "visionlab-slipstream @ git+https://github.com/harvard-visionlab/slipstream.git@v0.16.0" \
            "visionlab-datasets @ git+https://github.com/harvard-visionlab/datasets.git@v0.21.0"
```

Or with [uv](https://github.com/astral-sh/uv), in a project's `pyproject.toml`:

```toml
dependencies = ["visionlab-datasets", "visionlab-slipstream"]

[tool.uv.sources]
visionlab-datasets = { git = "https://github.com/harvard-visionlab/datasets.git", tag = "v0.21.0" }
visionlab-slipstream = { git = "https://github.com/harvard-visionlab/slipstream.git", tag = "v0.16.0" }
```

uv applies a git dependency's own `[tool.uv.sources]`, so datasets' torch/torchvision sources (the
`pytorch-cu126` / `pytorch-cpu` indexes) reach your lock: declare those indexes and torch sources in
your project too (copy them from this repo's `pyproject.toml`), or `uv lock` fails on conflicting indexes.

A slipstream tag outside the range (e.g. v0.8.0) fails resolution. The range only moves in a datasets
minor release, because each slipstream minor so far has changed seeded streams.

**Note:** This package automatically installs GPU-enabled PyTorch on Linux x86_64 systems (CUDA 12.1) and CPU-only PyTorch on macOS and other platforms.

**Prerequisite: TurboJPEG.** slipstream (>= 0.9.2) builds its C++ decoder at install time and the
install *fails* if the TurboJPEG API (libturbojpeg + `turbojpeg.h`) is missing. Install it first:
`brew install jpeg-turbo` (macOS), `apt install libturbojpeg0-dev` (Debian/Ubuntu), or
`conda install -c conda-forge libjpeg-turbo`. For a non-standard prefix set `TURBOJPEG_ROOT=<prefix>`
($CONDA_PREFIX and ~/.local are searched automatically). On CLI-only machines that never decode,
set `SLIPSTREAM_SKIP_EXT=1`. `slipstream status` shows whether the decoder built.

## Usage

### On the FAS cluster: quickstart

The lab keeps every registered cache on the cluster, in a shared dir on netscratch. **Point your cache there; never
stream or re-download from S3 for training** (S3 egress costs ~$56 per ImageNet-1k pass and is slow from the cluster).

```bash
# 1. Once: in ~/.bashrc (and in sbatch scripts, to be safe; jobs inherit it from the shell you submit from)
export SLIPSTREAM_CACHE_DIR=/n/netscratch/alvarez_lab/Lab/datasets/slipstream

# 2. Check (fast, offline, login node OK): one ✓ line per cache, under the dir above
uv run --no-sync visionlab-datasets path imagenet1k train,val jpeg

# 3. Nothing else: load() checks the cache on every call (a size check, milliseconds) and refills anything
#    netscratch purged from the lab_storage master (free), using S3 only if the master lacks it.
#    To refill ahead of time (e.g. before a big sweep), in a job (shared partition, 1-2 cores):
uv run --no-sync visionlab-datasets sync imagenet1k train,val jpeg \
    --dest /n/netscratch/alvarez_lab/Lab/datasets/slipstream \
    --source /n/lab_storage/alvarez_lab/Lab/datasets/slipstream
```

Login node rules: `path`, `list`, and `status <dataset> --no-remote` are fine; `sync` and anything with `--deep`
(sha256 of hundreds of GB) go in a job. Always `uv run --no-sync`: a plain `uv run` may install packages.

If `uv run` hangs before printing anything, the login node's network-file locking is broken (uv locks its cache dir
first; all login nodes 2026-11-04, fixed by FASRC 2026-11-07; report it to rchelp if it recurs). Meanwhile call the
venv's binary directly (`$LAB_SCRATCH/venvs/<repo>/bin/visionlab-datasets ...`) or run in a job. Don't wrap `uv` in
`timeout`/`xargs`/etc.: that runs the real binary, bypassing the lab's `uv` shell function, and puts a `.venv` in the
holylabs checkout.

### Command line

Check where the slipstream cache lives on this machine, which lab datasets are
already present (and where), and whether you can read the local cache and the
S3 bucket:

```bash
uv run visionlab-datasets status              # cache dir + permissions, S3 access, per-dataset table
uv run visionlab-datasets status --paths      # ...plus full local path of every present cache
uv run visionlab-datasets status --no-remote  # offline (skip S3 checks)
uv run visionlab-datasets status imagenet1k --no-remote   # only some datasets (faster)
uv run visionlab-datasets list                # registered datasets and their S3 caches
uv run visionlab-datasets path imagenet100 val   # local cache path(s) for a dataset (✓ present, ✗ missing)
uv run visionlab-datasets path imagenet1k val original   # one format
```

Download caches: `sync <dataset> <splits> <fmt>`, where splits/formats are
comma lists or `all`. Formats: `jpeg`, `yuv420`, `original` (ImageNet val as distributed; `all` means `jpeg,yuv420`, so name
`original` explicitly). `path` and `sync`
take the format either positionally or as `--fmt`:

```bash
uv run visionlab-datasets sync imagenet100 train,val jpeg
uv run visionlab-datasets sync imagenet1k val all --dry-run
uv run visionlab-datasets sync imagenet1k train jpeg --deep   # also sha256 what's present, re-fetch mismatches
uv run visionlab-datasets status --deep --json                # sha256 check of present caches, machine-readable
# FAS cluster: copy from the lab_storage master instead of S3 (falls back to S3 per cache if the master copy is damaged)
uv run visionlab-datasets sync imagenet1k train,val jpeg --dest /n/netscratch/alvarez_lab/Lab/datasets/slipstream \
    --source /n/lab_storage/alvarez_lab/Lab/datasets/slipstream
```

`sync` (and `load()` when a cache is missing) is safe on a directory shared by several users
or jobs:

- **Lock.** Each cache has a lock file `.<cache>.sync.lock` next to it, recording owner, host,
  pid and start time, with a heartbeat every 60 s. A second sync of the same cache exits 1 and
  changes nothing, while `load()` waits for the other sync and uses its result. A lock is stale
  if its process is gone (same host) or its heartbeat is over 15 min old; the next sync breaks
  it automatically. `--break-lock` forces a takeover.
- **Stage, verify, commit.** Files download into `.<cache>.sync.partial/` on the same filesystem.
  Each is checked for size, and for sha256 if the manifest has `file_sha256`. Then they're renamed
  into the cache, `manifest.json` last. Nothing already in the cache is deleted.
- **Failure.** The staging dir is kept, with the reason in `SYNC_INCOMPLETE.json`, and the next
  sync resumes from it. `status` shows it.
- **Repair.** Only missing or wrong-size files are fetched, so a partly purged cache (e.g. on
  netscratch) re-downloads just what's gone. `--deep` also re-fetches right-size files whose
  sha256 doesn't match.
- **Only the cache's own files.** Sync copies manifest.json, the files it names, the
  `<field>_index.npy` indexes, and datasets' video-store sidecars (`records.parquet`,
  `store_manifest.json`). Anything else in the source, e.g. a leftover `slipcache/` copy, is
  skipped, and `status` flags such extras in local caches.
- **Rebuilt caches.** If the remote `manifest.json` differs from the local one, the cache was
  rebuilt upstream; sync refuses to mix versions unless given `--force`.
- **Shared permissions.** In a group-writable cache dir, everything sync creates is made
  group-writable whatever the caller's umask: the lock, the staging dir and its marker, files
  and dirs. So any group member can repair or re-sync the cache. `status` flags items without
  group write and prints the `chmod -R g+w` the owner should run. Personal cache dirs follow the umask.
- **Sources.** From S3, big files download in parallel parts: `--concurrency auto` (the default)
  uses 32 parts for files of 256 MB and up, 1 for the rest. Use `--concurrency 1` when writing to
  NFS that collapses under parallel part writes (e.g. /n/lab_storage). `--source DIR` copies from
  another directory holding the same caches (e.g. the lab_storage master) with `--readers`
  parallel ranged reads (default 16; `--chunk-mb`, default 64, sets the size of each read). It uses the same lock, verify and manifest-last rules, and
  falls back to S3 for any cache that's missing or damaged there.

Dataset names are the registry names shown by `list`; short aliases `in10`, `in100`,
`in1k`, `in100_s292` are accepted too. Also available as `python -m visionlab.datasets`.

This CLI is the one lab members should use: visionlab-datasets owns the registry of
lab-supported datasets (names, splits, formats, S3 cache locations, per-platform cache
dir). `slipstream status` from the `visionlab-slipstream` package is only a low-level
plumbing check (cache dir, s5cmd, AWS credentials) and knows nothing about lab datasets.
Adding a dataset means adding a config under `datasets/_configs/`; it then appears in
`status`, `list`, `path`, and `sync` automatically.

Cache directory resolution: `SLIPSTREAM_CACHE_DIR` if set, else a per-platform default
(`/n/lab_storage/alvarez_lab/Lab/datasets/slipstream` on the FAS cluster, `~/.slipstream`
on workstations/devboxes/devcontainers, `/tmp/slipstream_cache` on Lightning Studio).

**Purges and refills (`load()`, since 0.21.0).** netscratch deletes files ~90 days after they were written, used or
not, which leaves caches without `manifest.json` (`status`: `empty dir`) or missing data files (`incomplete`). A
missing file is not silent (the first read raises `FileNotFoundError`), but it crashes the job after it got its GPUs.
So every `load()` (image and video datasets) checks the cache first: every file the manifest lists, present at its
size (milliseconds; a purge deletes whole files, so no hashing is needed). If anything is missing it refills only
that, through the same lock/stage/verify path as `sync`:

1. from the **master copy** when the machine has one: `VISIONLAB_DATASETS_MASTER` if set (empty = none), else
   `/n/lab_storage/alvarez_lab/Lab/datasets/slipstream` on the FAS cluster. Free. With a master, the check also
   compares against the master's file list, which catches files the manifest doesn't list (`<field>_index.npy`);
2. else from **S3**, with a printed line giving the file count, size and estimated egress cost.

Several jobs or DDP ranks hitting the same purged cache refill it once; the others wait. `status --deep` still
sha256-checks contents.

### Python

```python
from visionlab.datasets import StreamingDataset
```

### List available streaming datasets

```python
from visionlab.datasets import list_streaming_datasets

datasets = list_streaming_datasets()
```

### Load a streaming dataset

```python
from visionlab.datasets import load_dataset

dataset = load_dataset("imagenet-1k-train")
```

### Decode images

```python
import io
from PIL import Image

sample = dataset[0]

# Decode image bytes with PIL
pil_image = Image.open(io.BytesIO(sample['image']))

# Or use automatic decoding
dataset = load_dataset("imagenet-1k-train", decode_images=True)
sample = dataset[0]
pil_image = sample['image']  # Already decoded
```

### Store identity and sync (any slipstream store)

```python
from visionlab.datasets import cache_identity, sync_store

# "sha256:..." over the manifest's sample count + per-file sha256; also ds.identity after load()
cache_identity("/path/to/store")

# Same lock/stage/verify/commit path as `visionlab-datasets sync`, for stores outside the registry
# (e.g. visionlab.worlds rollouts). From S3, or from a local master copy with source_dir=.
sync_store("s3://visionlab-datasets/<prefix>/<store>", "/n/netscratch/.../<store>")
sync_store("s3://...", "/n/netscratch/.../<store>", source_dir="/n/lab_storage/.../<store>", deep=True)
```

---

## Development Setup

### Prerequisites

-   Python 3.10-3.12
-   [uv](https://github.com/astral-sh/uv) (recommended) or pip
-   AWS credentials configured (for accessing remote datasets)

### Option 1: Local Development

Clone the repository and install in editable mode with dev dependencies:

```bash
git clone https://github.com/harvard-visionlab/datasets.git
cd datasets
uv lock --python 3.10.12
uv sync --dev
```

This installs:

-   CPU PyTorch on macOS
-   GPU PyTorch (CUDA 12.1) on Linux

#### Configure nbstripout

To prevent notebook outputs from being committed to git, run these commands once per local clone:

```bash
git config filter.nbstripout.clean 'uv run nbstripout'
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
```

The `.gitattributes` file is already configured to apply the filter to `*.ipynb` files.

#### Running tests

```bash
uv run pytest
```

### Option 2: Dev Container (Docker)

The repository includes a dev container configuration for VS Code that provides a consistent development environment with JupyterLab.

#### Prerequisites

-   Docker
-   VS Code with the "Dev Containers" extension

#### Getting Started

1. Open the repository in VS Code
2. When prompted, click "Reopen in Container" (or use the command palette: `Dev Containers: Reopen in Container`)
3. Wait for the container to build and start
4. JupyterLab will be available at `http://localhost:8888`

The dev container:

-   Uses Python 3.10
-   Installs all dependencies via `uv sync --dev`
-   Automatically configures `nbstripout`
-   Mounts your AWS credentials from `~/.aws`
-   Persists the `.venv` across container rebuilds
-   Includes libjpeg-turbo and OpenCV built from source

#### Manual Docker build (without VS Code)

```bash
cd .devcontainer
docker build -t visionlab-datasets -f Dockerfile ..
docker run -it --rm \
  -v $(pwd)/..:/workspace \
  -v ~/.aws:/root/.aws:ro \
  -p 8888:8888 \
  visionlab-datasets
```

### Project Structure

```
datasets/
├── datasets/                # Source code
│   ├── __init__.py
│   └── streaming_dataset.py
├── .devcontainer/           # Dev container configuration
│   ├── Dockerfile
│   └── devcontainer.json
├── install_*.sh             # Build scripts for native dependencies
├── pyproject.toml           # Project configuration
└── README.md
```

### Platform-specific PyTorch installation

The `pyproject.toml` uses uv's conditional sources to automatically select the correct PyTorch build:

-   **Linux x86_64**: CUDA 12.1 wheels from `https://download.pytorch.org/whl/cu121`
-   **macOS / other**: CPU-only wheels from `https://download.pytorch.org/whl/cpu`

If you need a different CUDA version, you can override by installing manually:

```bash
# Example: CUDA 11.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
