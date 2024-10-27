from .utils import *
from .version import *

def load_dataset(dataset_name, split, fmt='images', res=None, transform=None, profile='wasabi'):
    if fmt=="images":
        from . import images as image_datasets
        dataset = image_datasets.__dict__[dataset_name](split, res=res, transform=transform)
    elif fmt=="ffcv":
        from . import ffcv as ffcv_datasets
        dataset = ffcv_datasets.__dict__[dataset_name](split, transform=transform)
    elif fmt=="streaming":
        from . import litdata as streaming_datasets
        dataset = streaming_datasets.__dict__[dataset_name](split, transform=transform, profile=profile)
        
    return dataset
