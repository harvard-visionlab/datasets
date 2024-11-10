import os
import warnings
from enum import Enum
from pathlib import Path
from collections import OrderedDict
from litdata.constants import _IS_IN_STUDIO
from urllib.parse import urlparse

from .s3_auth import get_aws_credentials

_DEFAULT_STUDIO_CACHEDIR = "/teamspace/cache/datasets/"
_DEFAULT_DIRS=OrderedDict([
    ('NETSCRATCH', "/n/netscratch/alvarez_lab/Lab/datasets/cache"),
    ('TIER1', "/n/alvarez_lab_tier1/Lab/datasets/cache"),
    ('DEVBOX', os.path.expanduser(path="~/work/DataLocal/cache")),
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
    
def get_cache_root():
    platform = check_platform()
    if platform == Platform.LIGHTNING_STUDIO:
        return os.getenv('STUDIO_CACHE', _DEFAULT_STUDIO_CACHEDIR)
    else:
        for folder in _DEFAULT_DIRS.values():
            if os.path.exists(folder):
                return folder
    warnings.warn("NO cache directory found!")
    return None

def get_cache_dir(source=None, cache_root=None, profile_name=None):
    '''
        Gets the cache directory for different sources, including s3, http(s), or mnt
    '''
    cache_root = get_cache_root() if cache_root is None else cache_root
    if source is None: return cache_root
    parsed = urlparse(source)
    scheme = parsed.scheme
    netloc = parsed.netloc
    key = parsed.path.lstrip("/")
    
    if scheme == "s3":
        endpoint = get_aws_credentials(profile_name=profile_name)['endpoint_url']
        endpoint = urlparse(endpoint).netloc
        local_dir = os.path.join(cache_root, scheme, endpoint, netloc, key)
    elif scheme in ['https', 'http']:
        local_dir = os.path.join(cache_root, scheme, netloc, key)
    else:
        key = os.path.abspath(key).lstrip("/")
        local_dir = os.path.join(cache_root, 'mnt', netloc, key)
    return local_dir