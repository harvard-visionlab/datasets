'''
    some helper functions to determine the local_path and remote_path for streaming datasets,
    based on whether you are on lightning.ai, the cluster, or some local machine.
'''
import os
import warnings
from enum import Enum
from pathlib import Path
from collections import OrderedDict
from litdata.constants import _IS_IN_STUDIO

S3_MOUNT = os.getenv('S3_LITDATA_MOUNT_DIR', '/teamspace/s3_connections/visionlab-litdata/')
S3_BUCKET = os.getenv('S3_LITDATA_BUCKET', 's3://visionlab-litdata/')
STUDIO_CACHE = os.getenv('STUDIO_CACHE', "/teamspace/cache/datasets/")

_DEFAULT_DIRS=OrderedDict([
    ('VAST', "/n/vast-scratch/kempner_alvarez_lab/datasets"),
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
    
def get_streaming_remote_root_dir():
    platform = check_platform()
    if platform == Platform.LIGHTNING_STUDIO:
        if os.path.exists(S3_MOUNT):            
            return S3_MOUNT
        else:
            warnings.warn(f"You should add {S3_BUCKET} to your teamspace. Ping visionlab slack #compute channel for help.")
    return S3_BUCKET

def get_streaming_local_root_dir(data_dirs=_DEFAULT_DIRS):
    platform = check_platform()    
    if platform == Platform.LIGHTNING_STUDIO:
        Path(STUDIO_CACHE).mkdir(parents=True, exist_ok=True)
        return STUDIO_CACHE
    else:
        for name,path in data_dirs.items():
            if os.path.exists(path):
                return path
    msg = f"Cannot find any of these local_dirs {[STUDIO_CACHE]+list(data_dirs.values())}. You might be storing an extra copy of the dataset somehwere undesirable."
    warnings.warn(msg)
    
def get_streaming_dirs(dataset):
    remote_root = get_streaming_remote_root_dir()
    local_root = get_streaming_local_root_dir()
    remote_dir = os.path.join(remote_root, dataset)
    local_dir = os.path.join(local_root, dataset)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    return dict(local_dir=local_dir, remote_dir=remote_dir)