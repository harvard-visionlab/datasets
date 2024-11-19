import os
import warnings
from typing import Any, Optional
from enum import Enum
from pdb import set_trace
from urllib.parse import urlparse
from torchvision.datasets.folder import is_image_file

zip_extensions = ['.zip']
tar_extensions = ['.tar', '.tar.gz', '.tgz']
ffcv_extensions = ['.beton', '.ffcv']
pth_extensions = ['.pth', '.pth.tar', '.pt']

# source locations
class S3_URI(str): pass
class HTTP_URL(str): pass
class MNT_DIR(str): pass
class MNT_FILE(str): pass

# Define class with data locations
class DatasetRemoteLocationType:
    S3_URI = S3_URI
    HTTP_URL = HTTP_URL
    MNT_DIR = MNT_DIR
    MNT_FILE = MNT_FILE
    
# source file formats
class STREAMING_DATASET(str): pass
class FFCV_FILE(str): pass
class ZIP_FILE(str): pass
class TAR_FILE(str): pass
class MAT_FILE(str): pass
class PTH_FILE(str): pass
class JSON_FILE(str): pass
class IMAGE_DIR(str): pass
class FILE_DIR(str): pass
class UNKNOWN(str): pass

# define class tracking file formats
class DatasetFormat:
    STREAMING = STREAMING_DATASET
    FFCV = FFCV_FILE
    ZIP = ZIP_FILE
    TAR = TAR_FILE
    MAT = MAT_FILE
    PTH = PTH_FILE
    JSON = JSON_FILE
    IMAGE_DIR = IMAGE_DIR
    FILE_DIR = FILE_DIR
    UNKNOWN = UNKNOWN
    
def get_source_location(source):
    if source.startswith("s3://"):
        return S3_URI(source)
    elif source.startswith("http://") or source.startswith("https://"):
        return HTTP_URL(source)
    else:
        if os.path.isdir(source):
            return MNT_DIR(source)
        elif os.path.isfile(source):
            return MNT_FILE(source)
        else:
            raise ValueError(f"Dataset source intepreted as directory or file, but can't be found: {source}")
    
def get_source_format(source):
    '''
        Figure out the type of dataset from currently supported DatasetFileTypes.
        
        The code assumes a litdata StreamingDataset if the sources has "litdata" in the name, 
        or points towards a bucket/directory with an "index.json" file.
            
        returns:
        DatasetFileType(Enum):
            STREAMING = "litdata"
            FFCV = "beton_file"
            ZIP = "zip_archive"
            TAR = "tar_archive"
            IMAGE_DIR = "image_directory"
    
    '''
    source_loc = get_source_location(source)
    if any([source.endswith(ext) for ext in ffcv_extensions]):
        return FFCV_FILE(source)   
    elif source.endswith('.mat'):
        return MAT_FILE(source)
    elif source.endswith('.json'):
        return JSON_FILE(source)
    elif any([source.endswith(ext) for ext in pth_extensions]):
        return PTH_FILE(source)
    elif (
        'litdata' in source
        or (type(source_loc) == MNT_DIR and os.path.isfile(os.path.join(source, 'index.json')))
        or (type(source_loc) == S3_URI and s3_file_exists(os.path.join(source, 'index.json')))
    ):
        return STREAMING_DATASET(source)
    elif any([source.endswith(ext) for ext in tar_extensions]) and not source.endswith(".pth.tar"):
        return TAR_FILE(source)
    elif any([source.endswith(ext) for ext in zip_extensions]):
        return ZIP_FILE(source)
    elif type(source_loc) == MNT_DIR:
        if is_image_directory(source):
            return IMAGE_DIR(source)
        return FILE_DIR(source) 
    else:
        valid_keys = [k for k in DatasetFormat.__dict__.keys() if not k.startswith("__")]
        warnings.warn(f"Unknown dataset format for this source: {source}. Expected source to be one of {valid_keys}")
        return UNKNOWN(source)

def is_image_directory(root_dir):
    """
    Check if a directory contains at least one image file, searching recursively.

    This function works for:
      1. `root_dir` as a single folder with images in it.
      2. `root_dir` with subfolders containing images.

    Parameters:
    root_dir (str): The root directory to search.
    
    Returns:
    bool: True if at least one image file is found, False otherwise.
    """
    # Traverse the root directory and subdirectories
    for root, _, fnames in os.walk(root_dir):
        # Check each file in the current directory level
        for fname in fnames:
            path = os.path.join(root, fname)
            if is_image_file(path):  # If an image file is found, return True
                return True

    # No images were found in any directory
    return False
