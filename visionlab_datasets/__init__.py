from . import ffcv as ffcv_datasets
from . import images as image_datasets
from . import streaming as streaming_datasets
from .utils import *

def load_dataset(dataset_name, split, fmt, res=None, transform=None):
    if fmt=="images":
        dataset = image_datasets.__dict__[dataset_name](split, res=res, transform=transform)
    elif fmt=="ffcv":
        dataset = ffcv_datasets.__dict__[dataset_name](split, transform=transform)
    elif fmt=="streaming":
        dataset = streaming_datasets.__dict__[dataset_name](split, transform=transform)
        
    return dataset
