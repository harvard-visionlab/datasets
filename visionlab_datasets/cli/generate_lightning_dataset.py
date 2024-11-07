'''
    script for generating a lightning streaming dataset
    
    Example:
    generate_lightning_dataset --root_dir /n/holyscratch01/alvarez_lab/Lab/datasets/ecoset --split val --output_root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/ecoset-litdata -short_resize 256 --long_crop 512 --quality 100 --image_format "jpgbytes" --chunk_bytes "64MB" --num_expected 
    
'''
import os, io
import fire
from pathlib import Path
from litdata import optimize
from PIL import Image
from torchvision import datasets, transforms
from typing import Any, Tuple
from functools import partial
from pdb import set_trace

format_mapping = {
    "jpgbytes": "JPEG",
    "pngbytes": "PNG",
    "webpbytes": "WEBP",
    "PILImage": "PIL"
}

# ===============================================================
#  Dataset Wrappers
# ===============================================================

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
    
# ===============================================================
#  Transforms
# ===============================================================

class ConvertToRGB:
    """Convert the given image to RGB format."""

    def __call__(self, img):
        return img.convert('RGB')

    def __repr__(self):
        return f'{self.__class__.__name__}()'
    
class ResizeShortWithMaxLong():
    def __init__(self, short_size, longest_size):
        self.short_size = short_size
        self.longest_size = longest_size

        self.resize = transforms.Resize(short_size)
        self.crop_width = transforms.CenterCrop( (short_size, longest_size) )
        self.crop_height = transforms.CenterCrop( (longest_size, short_size) )

    def __call__(self, img):
        img = self.resize(img)

        # Check the image dimensions
        width, height = img.size

        # If width exceeds longest_size, crop width
        if width > self.longest_size:
            img = self.crop_width(img)

        # If height exceeds longest_size, crop height
        if height > self.longest_size:
            img = self.crop_height(img)

        width, height = img.size
        assert width==self.short_size or height==self.short_size, f"Oops, one side should equal {self.short_size}"
        assert width <= self.longest_size, f"Oops, width exceeds {self.longest_size}, {width}"
        assert height <= self.longest_size, f"Oops, height exceeds {self.longest_size}, {height}"

        return img
       
# ===============================================================
#  generation function
# ===============================================================

# helper function that will be called by works with (path,label,index) tuples
def get_sample(item, transform=None, image_format="jpgbytes", quality=100):
    actual_format = format_mapping[image_format]
    path, label, index = item

    img = Image.open(path)
    if transform is not None:
        img = transform(img)

    if image_format.endswith("bytes"):
        image_bytes = io.BytesIO()
        img.save(image_bytes, format=actual_format, quality=quality, optimize=True)
        image_bytes.seek(0)
        image = image_bytes.read()
    else:
        image = img

    data = {
        "index": index,
        "image": image,
        "label": label,
        "path": path,
    }

    return data
    
def generate_dataset(root_dir, split, output_root_dir, short_resize=None, long_crop=None, quality=100, 
                     image_format="jpgbytes", chunk_bytes="64MB", convert_to_rgb=True, num_expected=None):
    
    # make sure image_format is supported
    assert image_format in format_mapping, f"Expected format to be in: {format_mapping}, got {image_format}" 
    
    # set output directory based on args
    folder_name = f"streaming-s{short_resize}-l{long_crop}-{image_format}-q{quality}"
    output_dir = os.path.join(output_root_dir, folder_name, split)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load dataset
    if "imagenet1k" in root_dir:
        dataset = ImageNet1k(root_dir, split=split)
    else:
        dataset = ImageFolder(os.path.join(root_dir, split))
    print(dataset)
    if num_expected is not None:
        assert len(dataset)==num_expected, f"oops, expected {num_expected} samples, got {len(dataset)}"
    inputs = [(path,label,index) for index,(path,label) in enumerate(dataset.imgs)]
    
    # setup transforms
    transform_list = []
    if convert_to_rgb:
        transform_list += [ConvertToRGB()]
    
    if short_resize is not None and long_crop is not None:
        transform_list += [ResizeShortWithMaxLong(int(short_resize), int(long_crop))]
    elif short_resize is not None:
        transform_list += [transforms.Resize(short_size)]
    elif long_crop is not None:
        raise ValueError('You must provide a short_resize value to use long_crop')
                         
    transform = transforms.Compose(transform_list)
    get_sample_fun = partial(get_sample, transform=transform, image_format=image_format, quality=quality)
    
    # store images into the chunks
    optimize(
        fn=get_sample_fun,  # The function applied over each input.
        inputs=inputs,  # Provide any inputs. The fn is applied on each item.
        output_dir=output_dir,  # The directory where the optimized data are stored.
        num_workers=len(os.sched_getaffinity(0)) - 1,  # The number of workers. The inputs are distributed among them.
        chunk_bytes=chunk_bytes # The maximum number of bytes to write into a chunk.
    )
    
def main():
    fire.Fire(generate_dataset)

if __name__ == '__main__':
    fire.Fire(generate_dataset)    