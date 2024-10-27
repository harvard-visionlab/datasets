from .utils import *
from .version import *

formats = ["streaming", "ffcv", "images"]

def load_dataset(dataset_name, split, fmt='streaming', res=None, quality=None, transform=None, profile='wasabi',
                 **kwargs):
    assert fmt in formats, f"Oops, fmt must be in {formats}, got {fmt}"
    if fmt=="images":
        from . import images as image_datasets
        dataset = image_datasets.__dict__[dataset_name](split, res=res, transform=transform)
    elif fmt=="ffcv":
        from . import ffcv as ffcv_datasets
        dataset = ffcv_datasets.__dict__[dataset_name](split, transform=transform)
    elif fmt=="streaming":
        from . import litdata as streaming_datasets
        dataset = streaming_datasets.__dict__[dataset_name](split, res=res, quality=quality,
                                                            transform=transform, profile=profile,
                                                            **kwargs)
        
    return dataset
