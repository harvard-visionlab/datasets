from torchvision import datasets
from pathlib import Path
from typing import Any, Tuple

class ImageNet1k(datasets.ImageNet):
    meta = dict(
        num_train=1281167,
        num_val=50000
    )
    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target, index, path) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        rel_path = path.replace(self.root,'')
        
        return img, target, index, rel_path
    
class ImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target, index, path) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        rel_path = path.replace(self.root,'')
        
        return img, target, index, rel_path