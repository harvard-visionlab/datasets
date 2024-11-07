'''imagenetV2

github: https://github.com/modestyachts/ImageNetV2
huggingface: https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main

@article{DBLP:journals/corr/abs-1902-10811,
  author       = {Benjamin Recht and
                  Rebecca Roelofs and
                  Ludwig Schmidt and
                  Vaishaal Shankar},
  title        = {Do ImageNet Classifiers Generalize to ImageNet?},
  journal      = {CoRR},
  volume       = {abs/1902.10811},
  year         = {2019},
  url          = {http://arxiv.org/abs/1902.10811},
  eprinttype    = {arXiv},
  eprint       = {1902.10811},
  timestamp    = {Tue, 21 May 2019 18:03:38 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1902-10811.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

'''
import os
from torchvision import datasets
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from ..utils import get_remote_data_file, download_data_from_url, get_signed_s3_url
from ..auth import sign_url_if_needed

from pdb import set_trace

__all__ = ['imagenetV2']

# Private Datasets use s3_urls
urls = {
  ("matched-frequency", "full"): "https://s3.us-east-1.wasabisys.com/visionlab-datasets/imagenetV2/imagenetv2-matched-frequency-f0c37fdf92.tar",
}

hash_prefixes = {
    ("matched-frequency", "full"): "f0c37fdf92"
}

splits = [k[0] for k in urls.keys()]
resolutions = [k[1] for k in urls.keys()]

num_expected = {"matched-frequency": 10000}

class ImageNetV2Folder(datasets.ImageFolder):
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: int(cls_name) for cls_name in classes}
        return classes, class_to_idx
    
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
    

def imagenetV2(split, res=None, cache_dir=None, transform=None):
    if res is None: res = 'full'
    assert split in splits, f"Expected split to be in {splits}, got {split}"
    assert res in resolutions, f"Expected res to be one of {resolutions}, got {res}"
    url = urls[(split,res)]
    url = sign_url_if_needed(url)
    hash_prefix = hash_prefixes[(split,res)]
    cached_filename, extracted_folder = get_remote_data_file(url, 
                                                             hash_prefix=hash_prefix,
                                                             cache_dir=cache_dir, 
                                                             file_name=None, 
                                                             check_hash=True)
    
    dataset = ImageNetV2Folder(extracted_folder, transform=transform)
    dataset.name = 'imagenetV2'
    dataset.split = split
    dataset.hash_id = hash_prefix
    dataset.id = f"{dataset.name}_{dataset.split}_{dataset.hash_id}"
    dataset.citation = """
        @article{DBLP:journals/corr/abs-1902-10811,
          author       = {Benjamin Recht and
                          Rebecca Roelofs and
                          Ludwig Schmidt and
                          Vaishaal Shankar},
          title        = {Do ImageNet Classifiers Generalize to ImageNet?},
          journal      = {CoRR},
          volume       = {abs/1902.10811},
          year         = {2019},
          url          = {http://arxiv.org/abs/1902.10811},
          eprinttype    = {arXiv},
          eprint       = {1902.10811},
          timestamp    = {Tue, 21 May 2019 18:03:38 +0200},
          biburl       = {https://dblp.org/rec/journals/corr/abs-1902-10811.bib},
          bibsource    = {dblp computer science bibliography, https://dblp.org}
        }
    """
    num_images = len(dataset)
    assert num_images==num_expected[split], f"Oops, expected {num_expected[split]} images, found {num_images}. Check the files at the dataset location: {extracted_folder}"

    # make sure the assigned label matches the folder-number
    for path,label in dataset.imgs:
        assert int(Path(path).parent.name) == label, f"Path foldername conflicts with label ({label}), {path}"
    
    return dataset
  
