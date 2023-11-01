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

urls = {
  "full": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
  "320": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
  "160": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
}

def imagenette(split, cache_dir=None, res='320', transform=None):
  assert split in ['train', 'val'], f"Expected split to be `train` or `val`, got {split}"
  assert res in urls.keys(), f"Expected res to be a string, one of {urls.keys()}, got {res}"
  url = urls[res]
  cached_filename, extracted_folder = get_remote_data_file(url, cache_dir=cache_dir)
  dataset_dir = os.path.join(extracted_folder, split)
  dataset = datasets.ImageFolder(dataset_dir, transform=transform)
  return dataset
  