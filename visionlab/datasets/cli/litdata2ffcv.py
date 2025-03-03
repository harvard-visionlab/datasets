'''
    TLDR:
        We abandoned this because ffcv isn't making it easy to go directly from bytes => Image.
        
    convert dataset from litdata to ffcv
        from litdata StreamingDataset (https://github.com/Lightning-AI/litdata)
        to ffcv (beton) dataset (https://github.com/facebookresearch/FFCV-SSL)
        
        litdata2ffcv /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata/streaming-s256-l512-jpgbytes-q100/val /n/alvarez_lab_tier1/Users/alvarez/datasets/ffcv/imagenet1k-s256-l512-jpgbytes-q100-testing123.ffcv 50000 --src_image_format jpgbytes --chunk_size 100 --shuffle_indices False
'''
import os
import fire
from pathlib import Path
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, JSONField, BytesField
from pdb import set_trace

from PIL import Image
import io
import numpy as np
import random

from ..litdata.streaming_dataset import StreamingDatasetVisionlab

class DataSet(StreamingDatasetVisionlab):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return sample['image'], sample['label'], sample['index'], sample['path']
    
class DummDataset():
    def __len__(self):
        return 10000
    
    def __getitem__(self, index):
        # Generate a fake image as a NumPy array (e.g., 64x64 pixels, RGB)
        # fake_image_array = np.random.randint(0, 256, (64, random.choice([64, 124, 256]), 3), dtype=np.uint8)
        fake_image_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        # Convert the array to a PIL Image
        fake_image = Image.fromarray(fake_image_array)
        
        # Save the image to a bytes buffer
        buffer = io.BytesIO()
        fake_image.save(buffer, format='JPEG', quality=95)  # Use 'JPEG' or another format if preferred
        # fake_image_bytes = buffer.getvalue()
        fake_image_bytes = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        
        return fake_image_bytes, 0, index, 'testing'
    
def get_writer(write_path, src_image_format="jpgbytes", num_workers=None):
    
    num_workers = len(os.sched_getaffinity(0)) - 1 if num_workers is None else num_workers
    
    if src_image_format.endswith("bytes"):
        image_field = BytesField()
    else:
        raise NotImplementedError(f"src_image_format {src_image_format} isn't yet supported")
        
    # assuming standard image fields here
    fields = {
        'image': image_field,
        'label': IntField(),
        'index': IntField(),
        'path': JSONField(),        
    }

    writer = DatasetWriter(write_path, fields, num_workers=num_workers)  
    
    return writer

def convert_dataset(input_dir, write_path, num_expected, src_image_format="jpgbytes", chunk_size=100, shuffle_indices=True, expected_version=None):
    Path(write_path).parent.mkdir(parents=True, exist_ok=True)
    
    writer = get_writer(write_path, src_image_format=src_image_format, num_workers=0)
    
    dataset = DataSet(input_dir, expected_version=expected_version)
    print(dataset)
    assert len(dataset)==num_expected, f"Expected {num_expected} samples, got {len(dataset)}"
    print("\n==> Dataset OK\n")
    
    dataset = DummDataset()
    
    writer.from_indexed_dataset(dataset, chunksize=chunk_size, shuffle_indices=shuffle_indices)
    
def main():
    fire.Fire(convert_dataset)

if __name__ == '__main__':
    fire.Fire(convert_dataset)                                             