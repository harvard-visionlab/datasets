import os
import random
from pathlib import Path
from typing import List, Optional, Tuple
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader, is_image_file

from torch.utils.data import Dataset, Subset
from collections import defaultdict
import numpy as np

class SubFolderDataset(Dataset):
    def __init__(self, root_dir: str, subfolders: Optional[List[str]] = None, transform=None):
        """
        Custom dataset that loads images from specified subfolders within the root directory.
        If no subfolders are specified, it loads images from all subfolders (similar to ImageFolder).

        Args:
            root_dir (str): Path to the root directory containing subfolders of images.
            subfolders (Optional[List[str]]): List of subfolder names to include. If None, use all subfolders.
            transform (optional): Optional transform to apply to each image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.loader = default_loader

        # Use specific subfolders or default to all subfolders
        if subfolders is None:
            # If no subfolders are specified, behave like ImageFolder
            self.dataset = ImageFolder(root_dir, transform=transform)
        else:
            # Only include images from specified subfolders
            self.classes = subfolders
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(subfolders)}
            self.samples = []

            # Collect images from each specified subfolder
            for cls_name in subfolders:
                cls_dir = os.path.join(root_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for root, _, fnames in sorted(os.walk(cls_dir)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        if is_image_file(path):
                            self.samples.append((path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        partial_path = os.path.join(Path(path).parent.name, Path(path).name)
        return dict(image=sample, label=target, index=index, filepath=partial_path)
    
    def __repr__(self) -> str:
        _repr_indent = 4
        tab = " " * _repr_indent
        head = "SubFolderDataset"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"root_dir: {self.root_dir}")
            
        body += [f"\nTransforms on image:"]
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)
                     .replace("\n",f"\n{tab}")
                     .replace('\n)', '\n' + ' ' * _repr_indent + ')')]
        else:
            body += ['None']
            
        lines = [head] + [" " * _repr_indent + line for line in body]
        
        lines += ['\nSample Info:']
        sample = self.__getitem__(0)
        any_bytes = False
        if hasattr(sample, 'items'):
            for key,value in sample.items():
                lines += [" " * _repr_indent + f"{key}: {type(value)}"]
                if isinstance(value, bytes): 
                    any_bytes = True
        else:
            for idx,value in enumerate(sample):
                lines += [" " * _repr_indent + f"{idx}: {type(value)}"]
                if isinstance(value, bytes): 
                    any_bytes = True            

        return "\n".join(lines)