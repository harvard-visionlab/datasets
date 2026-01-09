# visionlab-datasets

Visionlab dataset utilities for streaming datasets.

## Installation

Install from GitHub:

```bash
pip install git+https://github.com/harvard-visionlab/datasets.git
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install git+https://github.com/harvard-visionlab/datasets.git
```

**Note:** This package automatically installs GPU-enabled PyTorch on Linux x86_64 systems (CUDA 12.1) and CPU-only PyTorch on macOS and other platforms.

## Usage

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
