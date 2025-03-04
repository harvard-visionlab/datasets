import warnings

from ..utils import sync_to_local_cache, has_keyword_arg
from .source_types import get_source_format, DatasetFormat
from .streaming_dataset import StreamingDatasetVisionlab
from .filedir import FileDirDataset
from .jsonfile import JSONFileDataset
from .matfile import MatFileDataset
from .pytorchfile import PyTorchFileDataset
from .subfolder import SubFolderDataset

from pdb import set_trace

dataset_classes = {
        DatasetFormat.STREAMING: StreamingDatasetVisionlab,
        DatasetFormat.IMAGE_DIR: SubFolderDataset,
        DatasetFormat.MAT: MatFileDataset,
        DatasetFormat.PTH: PyTorchFileDataset,
        DatasetFormat.JSON: JSONFileDataset,
        DatasetFormat.FILE_DIR: FileDirDataset,
    }

try:
    import ffcv
except:
    warnings.warn("FFCV not installed, will be unable to use ffcv datasets. Ignore this warning if you are not using ffcv.")

try:
    from .ffcv_dataset import FFCVDataset
    dataset_classes[DatasetFormat.FFCV] = FFCVDataset
except:
    warnings.warn("Error importing FFCVDataset")
    
class RemoteDataset:
    """Abstract base class for datasets loaded from remote sources.
        
        The remote source can be an s3 bucket location, a url (http or https), 
        or even a OS file or directory (e.g., for mounted file systems or s3 buckets).
        
        Data are copied from the source, to the local_cache dir, using
        the default cache_root if None is set (see utils.cache_dir),
        using some reasonable cache-locations to avoid collisions or duplicating data.
        
        Args:
            source (str): s3_uri, http(s)_url, or mounted folder/file
            profile_name: name of aws_profile name used for s3_storage_credentials
        
        Different sources rely on different parent classes, e.g.,
        StreamingDataset will be used for litdata streaming datsets, and
        SubFolderDataset will be used for collections of image files,
        FFCVDataset will be used for .ffcv or .beton files.
        
        What happens with different sources?
        
        StreamingDataset
            s3://bucket/dataset => cache_dir (~/.lightning/chunks/)
            /local_dir/local_dataset => cache_dir (~/.lightning/chunks/)        
    """
    
    def __init__(self, source, dataset_cls=None, cache_root=None, profile_name=None, region='us-east-1', **kwargs):
        self.source = source
        self.cache_root = cache_root
        self.dataset_cls = dataset_cls
        self.profile_name = profile_name
        self.local_path = self._sync_dataset(source, cache_root, profile_name, region)
        set_trace()
        self.dataset = self._initialize_dataset(self.local_path, **kwargs)
    
    def _sync_dataset(self, source, cache_root, profile_name, region):
        local_path = sync_to_local_cache(source, cache_root=cache_root, profile_name=profile_name, region=region)
        return local_path
    
    def _initialize_dataset(self, source, **kwargs):
        """Initializes and returns the appropriate dataset instance."""
        source = get_source_format(source)     
        if type(source) not in dataset_classes and self.dataset_cls is None:
            lines = [f'Dataset class not configured for this type: {type(source)}']
            lines += ['See visionlab_datasets.remote_dataset.dataset_classes) for supported types, or pass `dataset_cls`']
            raise ValueError("\n".join(lines))
        
        print(f"==> Detected dataset format: {type(source)}")
        dataset_cls = dataset_classes[type(source)] if self.dataset_cls is None else self.dataset_cls
                
        if has_keyword_arg(dataset_cls, 'cache_dir'):
            return dataset_cls(source, cache_dir=self.cache_dir, **kwargs)
        else:
            return dataset_cls(source, **kwargs)

    def __getattr__(self, attr):
        """Delegate attribute access to the dataset instance."""
        return getattr(self.dataset, attr)
    
    def __repr__(self):
        """Return the repr of the underlying dataset."""
        return repr(self.dataset)
    
    def __getitem__(self, index):
        """Delegate item access to the underlying dataset."""
        return self.dataset[index]

    def __len__(self):
        """Delegate length access to the underlying dataset."""
        return len(self.dataset)