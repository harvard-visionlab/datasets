'''
    script for generating an ffcv (.beton) dataset (https://github.com/facebookresearch/FFCV-SSL; https://github.com/libffcv/ffcv)
    
    Example:
    generate_ffcv_dataset --root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/rawdata/imagenet1k --split val --write_path /n/alvarez_lab_tier1/Users/alvarez/datasets/ffcv/imagenet1k/imagenet1k-s256-l512-jpg-q100-val.ffcv -short_resize 256 --long_crop 512 --quality 100 --write_mode "jpg" --chunk_size 100 --num_expected 50000
    
'''
import os, io
import fire
from pathlib import Path
from litdata import optimize
from PIL import Image
from torchvision import transforms
from functools import partial
from pdb import set_trace

from . import ffcv_writers as writers 
from .datasets import ImageNet1k, ImageFolder

# ===============================================================
#  get writer
# ===============================================================

def get_writer(writer_type, write_path, short_resize=256, long_crop=512, max_enforced_with='center_crop',
               quality=100, write_mode='jpg', smart_threshold=None, compress_probability=1.0,
               num_workers=None):
    
    num_workers = len(os.sched_getaffinity(0)) - 1 if num_workers is None else num_workers
    
    if compress_probability > 0.0 and compress_probability < 1.0: 
        assert write_mode=='proportion', f"Write mode_must be `proportion` for compress_probability > 0, got {write_mode}"
    elif compress_probability == 0.0:
        assert write_mode=='raw', f"Write mode_must be `raw` for compress_probability==0.0, got {write_mode}"
    
    Path(write_path).parent.mkdir(parents=True, exist_ok=True)
    
    writer = writers.__dict__[writer_type](write_path, 
                                           write_mode=write_mode, 
                                           smart_threshold=smart_threshold,
                                           min_resolution=short_resize,
                                           max_resolution=long_crop,
                                           max_enforced_with=max_enforced_with,
                                           compress_probability=compress_probability,
                                           jpeg_quality=quality)
    
    return writer    

       
# ===============================================================
#  generation function
# ===============================================================
    
def generate_dataset(root_dir, split, write_path, writer_type="image_label_index_path", write_mode='jpg',
                     short_resize=256, long_crop=512, max_enforced_with='center_crop', quality=100, 
                     smart_threshold=None, compress_probability=1.0, chunk_size=100, shuffle_indices=False, num_expected=None):
    
    # load dataset
    if "imagenet1k" in root_dir:
        dataset = ImageNet1k(root_dir, split=split)
    else:
        dataset = ImageFolder(os.path.join(root_dir, split))
    print(dataset)
    if num_expected is not None:
        assert len(dataset)==num_expected, f"oops, expected {num_expected} samples, got {len(dataset)}"
        
    writer = get_writer(writer_type, write_path, short_resize=short_resize, long_crop=long_crop, max_enforced_with=max_enforced_with,
                        quality=quality, write_mode=write_mode, smart_threshold=smart_threshold, compress_probability=compress_probability) 
    print(writer)
        
    writer.from_indexed_dataset(dataset, chunksize=chunk_size, shuffle_indices=shuffle_indices)

    
def main():
    fire.Fire(generate_dataset)

if __name__ == '__main__':
    fire.Fire(generate_dataset)    