from . import ffcv as ffcv_datasets
from . import images as image_datasets
from . import streaming as streaming_datasets
from .utils import *

def load_dataset(dataset_name, split, fmt, res=None, transform=None):
    if fmt=="images":
        dataset, output_mapping = image_datasets.__dict__[dataset_name](split, res=res, transform=transform)
    elif fmt=="ffcv":
        dataset, output_mapping = ffcv_datasets.__dict__[dataset_name](split, transform=transform)
    elif fmt=="streaming":
        dataset, output_mapping = streaming_datasets.__dict__[dataset_name](split, transform=transform)
        
    return dataset, output_mapping
