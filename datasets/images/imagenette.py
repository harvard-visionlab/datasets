'''

From: https://github.com/fastai/imagenette/

Imagenette is a subset of 10 easily classified classes from Imagenet 
(tench, English springer, cassette player, chain saw, church, French horn, 
garbage truck, gas pump, golf ball, parachute).



@misc{imagenette,
  author    = "Jeremy Howard",
  title     = "imagenette",
  url       = "https://github.com/fastai/imagenette/"
}

'''
import os
from torchvision import datasets
from ..utils import get_remote_data_file

__all__ = ['imagenette']

urls = {
  "full": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
  "320": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
  "160": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
}

filename_with_hash = {
    "full": "imagenette2-6cbfac2384.tgz",
    "320": "imagenette2-320-569b4497c9.tgz",
    "160": "imagenette2-160-64d0c4859f.tgz"
}

num_expected = dict(train=9469, val=3925)

def imagenette(split, cache_dir=None, res='320', transform=None):
    assert split in ['train', 'val'], f"Expected split to be `train` or `val`, got {split}"
    assert res in urls.keys(), f"Expected res to be a string, one of {urls.keys()}, got {res}"
    url = urls[res]
    file_name = filename_with_hash[res]
    cached_filename, extracted_folder = get_remote_data_file(url, cache_dir=cache_dir, file_name=file_name, check_hash=True)
    dataset_dir = os.path.join(extracted_folder, split)
    dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    num_images = len(dataset)
    assert num_images==num_expected[split], f"Oops, expected {num_expected[split]} images, found {num_images}. Check the files at the dataset location: {extracted_folder}"
    
    return dataset
  
