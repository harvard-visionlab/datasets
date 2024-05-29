'''imagenet1k

Description:

ILSVRC 2012, commonly known as 'ImageNet' is an image dataset organized according to the WordNet hierarchy. This dataset spans 1000 object classes and contains 1,281,167 training images, 50,000 validation images and 100,000 test images. The classes are drastically imbalanced, with min_count=ASDF, max_count=ASDF, median_count=ASDF.

Citation:

@article{imagenet15russakovsky,
    Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = { {ImageNet Large Scale Visual Recognition Challenge} },
    Year = {2015},
    journal   = {International Journal of Computer Vision (IJCV)},
    doi = {10.1007/s11263-015-0816-y},
    volume={115},
    number={3},
    pages={211-252}
}
'''
import os
from torchvision import datasets
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from ..utils import get_remote_data_file, download_data_from_url, get_signed_s3_url

from pdb import set_trace

__all__ = ['imagenet1k']

# Private Datasets use s3_urls
urls = {
  ("val", "full"): "s3://visionlab-datasets/visionlab-datasets/imagenet1k/val.tar.gz",
}

toolkit_url = "s3://visionlab-datasets/visionlab-datasets/imagenet1k/ILSVRC2012_devkit_t12.tar.gz"

hash_prefixes = {
    ("val", "full"): "2d0d90c3ab"
}

splits = [k[0] for k in urls.keys()]
resolutions = [k[1] for k in urls.keys()]

num_expected = dict(train=1_281_167, val=50_000)

class ImagNetIndex(datasets.ImageNet):
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target, index
    
def download_toolkit_if_needed(root_folder):
    toolkit_filename = os.path.join(root_folder, "ILSVRC2012_devkit_t12.tar.gz")
    if not os.path.isfile(toolkit_filename):
        url = get_signed_s3_url(toolkit_url)
        
        cached_filename = download_data_from_url(
            url = url,
            data_dir = root_folder,
            progress = True,
            check_hash = False,
            hash_prefix = None,
            file_name = None,
        )
        print(cached_filename)
    
def imagenet1k(split, res=None, cache_dir=None, transform=None):
    if res is None: res = 'full'
    assert split in splits, f"Expected split to be in {splits}, got {split}"
    assert res in resolutions, f"Expected res to be one of {resolutions}, got {res}"
    url = urls[(split,res)]
    hash_prefix = hash_prefixes[(split,res)]
    cached_filename, extracted_folder = get_remote_data_file(url, 
                                                             hash_prefix=hash_prefix,
                                                             cache_dir=cache_dir, 
                                                             file_name=None, 
                                                             check_hash=True)
    
    root_folder = Path(extracted_folder).parent
    download_toolkit_if_needed(root_folder)
    
    dataset = ImagNetIndex(root_folder, split=split, transform=transform)
    num_images = len(dataset)
    assert num_images==num_expected[split], f"Oops, expected {num_expected[split]} images, found {num_images}. Check the files at the dataset location: {extracted_folder}"
    
    # no special mapping of output activations needed
    output_mapping = None
    
    return dataset, output_mapping
  
