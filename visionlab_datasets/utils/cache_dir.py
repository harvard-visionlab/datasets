import os
import warnings
from enum import Enum
from pathlib import Path
from collections import OrderedDict
from litdata.constants import _IS_IN_STUDIO

_DEFAULT_STUDIO_CACHEDIR = "/teamspace/cache/datasets/"
_DEFAULT_DIRS=OrderedDict([
    ('NETSCRATCH', "/n/netscratch/alvarez_lab/Lab/datasets"),
    ('TIER1', "/n/alvarez_lab_tier1/Lab/datasets"),
    ('DEVBOX', os.path.expanduser(path="~/work/DataLocal")),
])

SHARED_DATASET_DIR = os.getenv('SHARED_DATASET_DIR')
if SHARED_DATASET_DIR is not None:
    _DEFAULT_DIRS['SHARED_DATASET_DIR'] = SHARED_DATASET_DIR

# Define the Platform Enum
class Platform(Enum):
    LIGHTNING_STUDIO = "lightning_studio"
    FAS_CLUSTER = "fas_cluster"
    DEVBOX = "devbox"
    
def is_slurm_available():
    return any(var in os.environ for var in ["SLURM_JOB_ID", "SLURM_CLUSTER_NAME"])

def check_platform():
    if _IS_IN_STUDIO:
        return Platform.LIGHTNING_STUDIO
    elif is_slurm_available():
        return Platform.FAS_CLUSTER
    else:
        return Platform.DEVBOX
    
def get_cache_dir():
    platform = check_platform()
    if platform == Platform.LIGHTNING_STUDIO:
        return os.getenv('STUDIO_CACHE', _DEFAULT_STUDIO_CACHEDIR)
    else:
        for folder in _DEFAULT_DIRS.values():
            if os.path.exists(folder):
                return folder
    return None