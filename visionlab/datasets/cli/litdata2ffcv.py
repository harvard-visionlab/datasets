'''
    Convert a streaming dataset to an ffcv dataset.

    convert dataset from litdata to ffcv
        from litdata StreamingDataset (https://github.com/Lightning-AI/litdata)
        to ffcv (beton) dataset (https://github.com/facebookresearch/FFCV-SSL)
        
    Demo (write jpg, quality 100, short edge 256, max edge cropped at 512, specifing order/values with field_names):
    
    litdata2ffcv /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata/streaming-s256-l512-jpgbytes-q100/val /n/alvarez_lab_tier1/Users/alvarez/datasets/ffcv/imagenet1k-s256-l512-jpgbytes-q100-val.ffcv 50000 --expected-version 1741031197.5358698 --chunk_size 100 --shuffle_indices False --write_mode jpg --jpeg_quality 100 --min_resolution 256 --max-resolution 512 --field_names "['image', 'label', 'index', 'path']"
    
    litdata2ffcv /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata/streaming-s256-l512-jpgbytes-q100/train /n/alvarez_lab_tier1/Users/alvarez/datasets/ffcv/imagenet1k-s256-l512-jpgbytes-q100-train.ffcv 1281167 --expected-version 1741030773.04552 --chunk_size 100 --shuffle_indices False --min_resolution 256 --max-resolution 512 --field_names "['image', 'label', 'index', 'path']"
    
''' 

import os
import io
import random
import fire
import numpy as np
from collections import OrderedDict
from PIL import Image
from pathlib import Path
from pdb import set_trace

from visionlab.datasets.core import StreamingDatasetVisionlab
from visionlab.datasets.cli.ffcv_writers.custom_fields import RGBImageField

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, JSONField, BytesField

class DataSet(StreamingDatasetVisionlab):
    '''the ffcv writer expects a tuple of outputs, not a dictionary'''    
    def __init__(self, *args, field_names=None, **kwargs):
        # Initialize the parent class with all provided args and kwargs.
        super().__init__(*args, **kwargs)
        # Store the field_names if provided.
        self.field_names = field_names
        # If not provided, get the first sample to establish the order.
        if self.field_names is None:
            first_sample = super().__getitem__(0)
            self.field_names = list(first_sample.keys())
            
    def __getitem__(self, index):
        index = int(index) # major gotcha; DatasetWriter uses np.arange to generate indices, not recognized as "int"!!
        sample = super().__getitem__(index)
        
        # get values in order of field_names
        sample_list = []        
        for key in self.field_names:
            value = sample[key]
            if isinstance(value, (bytes,)):
                try:
                    # are these image bytes? If so, return a pil image
                    value = Image.open(io.BytesIO(value))
                except Exception as e:
                    raise ValueError("Oh no, wrong kind of bytes") from e           
            sample_list.append(value)
            
        return tuple(sample_list)

def get_writer(write_path, values, field_names, num_workers=None, 
               write_mode: str = 'jpg', 
               smart_threshold: int = None,
               min_resolution: int = None, 
               max_resolution: int = None,
               max_enforced_with: str = 'center_crop',
               compress_probability: float = 1.0,
               jpeg_quality: int = 100):
    
    if compress_probability > 0.0 and compress_probability < 1.0: 
        assert write_mode=='proportion', f"Write mode_must be `proportion` for compress_probability > 0, got {write_mode}"
    elif compress_probability == 0.0:
        assert write_mode=='raw', f"Write mode_must be `raw` for compress_probability==0.0, got {write_mode}"

    fields = {}
    for key,value in zip(field_names, values):
        if isinstance(value, Image.Image):
            fields[key] = RGBImageField(write_mode=write_mode,
                                        smart_threshold=smart_threshold,
                                        min_resolution=min_resolution,
                                        max_resolution=max_resolution,
                                        max_enforced_with=max_enforced_with,
                                        compress_probability=compress_probability,
                                        jpeg_quality=jpeg_quality)
        elif isinstance(value, (bytes, np.ndarray)):
            fields[key] = BytesField()
        elif isinstance(value, (int,np.integer)):
            fields[key] = IntField()
        elif isinstance(value, (str,)):
            fields[key] = JSONField()
        else:
            set_trace()
            raise NotImplementedError(f"field {key} type {type(key)} not yet implemented.")        
    print(fields)
    
    num_workers = len(os.sched_getaffinity(0)) - 1 if num_workers is None else num_workers
    writer = DatasetWriter(write_path, fields, num_workers=num_workers)  
    
    return writer

def convert_dataset(input_dir, write_path, num_expected, chunk_size=100, shuffle_indices=True, expected_version=None,
                    min_resolution=256, max_resolution=512, write_mode='jpg', jpeg_quality=100, compress_probability=1.0, 
                    field_names=None, num_workers=None):
    
    Path(write_path).parent.mkdir(parents=True, exist_ok=True)

    dataset = DataSet(input_dir, field_names=field_names, expected_version=expected_version)
    field_names = dataset.field_names
    print(dataset)
    print(field_names)
    
    assert len(dataset)==num_expected, f"Expected {num_expected} samples, got {len(dataset)}"
    print("\n==> Dataset OK\n")    
    
    writer = get_writer(write_path, values=dataset[0], field_names=field_names, num_workers=num_workers,
                        min_resolution=min_resolution, max_resolution=max_resolution,
                        compress_probability=compress_probability,
                        write_mode=write_mode, jpeg_quality=jpeg_quality)
    
    writer.from_indexed_dataset(dataset, chunksize=chunk_size, shuffle_indices=shuffle_indices)
    
def main():
    fire.Fire(convert_dataset)

if __name__ == '__main__':
    fire.Fire(convert_dataset) 