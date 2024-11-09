import os
from enum import Enum
from ..utils import s3_file_exists

zip_extensions = ['.zip']
tar_extensions = ['.tar', '.tar.gz', '.tgz']
ffcv_extensions = ['.beton', '.ffcv']

class DatasetSourceType(Enum):
    S3 = "s3_uri"
    HTTP = "http_url"
    DIR = "directory"
    FILE = "file"

class DatasetFileType(Enum):
    STREAMING = "litdata"
    FFCV = "beton_file"
    ZIP = "zip_archive"
    TAR = "tar_archive"
    IMAGE_DIR = "image_directory"
    
def get_dataset_source_type(source):
    if source.startswith("s3://"):
        return DatasetSourceType.S3
    elif source.startswith("http://") or source.startswith("https://"):
        return DatasetSourceType.HTTP
    else:
        if os.path.isdir(source):
            return DatasetSourceType.DIR
        elif os.path.isfile(source):
            return DatasetSourceType.FILE
        else:
            raise ValueError(f"Dataset source intepreted as directory or file, but can't be found: {source}")
            
def get_dataset_filetype(source):
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
    source_type = get_dataset_source_type(source)
    if any([source.endswith(ext) for ext in tar_extensions]):
        return DatasetFileType.TAR
    elif any([source.endswith(ext) for ext in zip_extensions]):
        return DatasetFileType.ZIP
    elif any([source.endswith(ext) for ext in ffcv_extensions]):
        return DatasetFileType.FFCV
    elif (
        'litdata' in source
        or (source_type == DatasetSourceType.DIR and os.path.isfile(os.path.join(source, 'index.json')))
        or (source_type == DatasetSourceType.S3 and s3_file_exists(os.path.join(source, 'index.json')))
    ):
        return DatasetFileType.STREAMING
    else:
        raise ValueError(f"Cannot determine dataset filetype, please pass a DatasetFileType: {DatasetFileType}")
        
def get_dataset_source_and_filetype(source):
    return get_dataset_source_type(source), get_dataset_filetype(source)
        