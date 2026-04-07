"""Platform detection and cache directory resolution.

Detects whether code is running on Lightning Studio, SLURM cluster,
devcontainer, GPU devbox, or plain CPU workstation, and maps each
platform to the appropriate slipstream cache directory.

Ported from lrm_ssl_analyses.utils.runtime_platform with slipstream-specific
cache directory defaults.
"""
import os
import posixpath
import shutil
from enum import Enum
from pathlib import Path


class Platform(Enum):
    LIGHTNING_STUDIO = "lightning_studio"
    FAS_CLUSTER = "fas_cluster"
    DEVCONTAINER = "devcontainer"
    GPU_DEVBOX = "gpu_devbox"
    CPU_WORKSTATION = "cpu_workstation"


# Platform-specific slipstream cache directories.
# Override with SLIPSTREAM_CACHE_DIR env var.
PLATFORM_CACHE_DIRS = {
    Platform.FAS_CLUSTER: "/n/netscratch/alvarez_lab/Lab/datasets/slipstream",
    Platform.LIGHTNING_STUDIO: "/tmp/slipstream_cache",
    Platform.DEVCONTAINER: str(Path.home() / ".slipstream"),
    Platform.GPU_DEVBOX: str(Path.home() / ".slipstream"),
    Platform.CPU_WORKSTATION: str(Path.home() / ".slipstream"),
}


def is_lightning_studio():
    """Check if running inside Lightning Studio."""
    try:
        from litdata.constants import _IS_IN_STUDIO
        return bool(_IS_IN_STUDIO)
    except Exception:
        return False


def is_slurm_available():
    """Check if running on a SLURM-managed cluster.

    Checks (in order): SLURM env vars, SLURM commands in PATH, FAS hostname.
    Note: SLURM env vars are not always propagated (e.g., Jupyter kernels),
    so we also check for SLURM commands and hostname patterns.
    """
    # Fast path: env vars (set inside SLURM jobs)
    if any(var in os.environ for var in ["SLURM_JOB_ID", "SLURM_CLUSTER_NAME"]):
        return True
    # SLURM commands available on the node
    if shutil.which("sbatch") is not None and shutil.which("squeue") is not None:
        return True
    return False


def has_slurm_scheduler():
    """Check whether SLURM client commands are available on this machine."""
    return shutil.which("sbatch") is not None and shutil.which("squeue") is not None


def is_devcontainer():
    """Check if running inside a VS Code devcontainer or Docker."""
    return (
        os.environ.get("REMOTE_CONTAINERS") == "true"
        or os.environ.get("REMOTE_CONTAINERS_IPC") is not None
        or Path("/.dockerenv").exists()
    )


def is_gpu_available():
    """Check if a CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def detect_platform():
    """Detect the current platform.

    Detection order: Lightning Studio -> SLURM -> Devcontainer -> GPU -> CPU.
    """
    if is_lightning_studio():
        return Platform.LIGHTNING_STUDIO
    if is_slurm_available():
        return Platform.FAS_CLUSTER
    if is_devcontainer():
        return Platform.DEVCONTAINER
    if is_gpu_available():
        return Platform.GPU_DEVBOX
    return Platform.CPU_WORKSTATION


def get_platform_cache_dir(platform=None):
    """Get the slipstream cache directory for the given (or detected) platform.

    Resolution order:
      1. SLIPSTREAM_CACHE_DIR env var (user override)
      2. Platform-specific default from PLATFORM_CACHE_DIRS
    """
    override = os.environ.get("SLIPSTREAM_CACHE_DIR")
    if override:
        return override
    if platform is None:
        platform = detect_platform()
    return str(Path(PLATFORM_CACHE_DIRS[platform]).expanduser())


def configure_slipstream_cache():
    """Set SLIPSTREAM_CACHE_DIR env var based on detected platform.

    Call this before creating any SlipstreamDataset to ensure the cache
    directory is set correctly for the current platform. No-op if the
    env var is already set.
    """
    cache_dir = get_platform_cache_dir()
    os.environ.setdefault("SLIPSTREAM_CACHE_DIR", cache_dir)
    return cache_dir


def join_cloud_path(base, *parts):
    """Join cloud storage paths (s3://, gs://, etc.) correctly.

    Preserves protocol prefix for cloud URLs, uses os.path.join for local paths.

    Examples:
        >>> join_cloud_path("s3://bucket/logs", "weights")
        's3://bucket/logs/weights'
        >>> join_cloud_path("/local/path", "weights")
        '/local/path/weights'
    """
    if "://" in base:
        protocol, path = base.split("://", 1)
        joined = posixpath.join(path.rstrip("/"), *parts)
        return f"{protocol}://{joined}"
    return os.path.join(base, *parts)
