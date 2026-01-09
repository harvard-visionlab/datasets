import io
import os
import sys
import numpy as np
import pathlib
from PIL import Image
from tqdm import tqdm
from litdata import StreamingDataset
from litdata.utilities.dataset_utilities import _read_updated_at
from copy import deepcopy
from pdb import set_trace

import torch
from torchvision.io import decode_image as tv_decode_image, ImageReadMode
from torchvision.transforms import functional as TF

class StreamingDatasetVisionlab(StreamingDataset):
    '''
    Visionlab version of StreamingDataset with pipelines for field transforms,
    automatic image decoding if requested, and pre-determined field type info.
    '''
    def __init__(self, *args, transform=None, pipelines=None, decode_images=False, expected_version=None, profile=None, storage_options={}, max_cache_size='350GB', **kwargs):
        # pipelines: dict mapping field names to transform functions.
        # decode_images: if True, bytes that are valid image bytes are decoded automatically
        ensure_lightning_symlink_on_cluster()
        self.pipelines = pipelines
        if self.pipelines is not None and transform is not None:
            raise ValueError('pipelines and transform cannot both be set. If you only need to transform images, you can use transform. To specify different transforms per field, use pipelines.')
        self.decode_images = decode_images
        storage_options = self.set_storage_options(storage_options, profile)
        super().__init__(*args, 
                         transform=transform,
                         storage_options=storage_options, 
                         max_cache_size=max_cache_size, 
                         **kwargs)
        self._transform = transform
        self.version = _read_updated_at(self.input_dir) if self.input_dir is not None else None
        if expected_version is not None:
            assert self.version == str(expected_version), (
                f"\n==> expected_version={expected_version}, got dataset.version={self.version}"
            )

        # load a sample to determine each field type
        self._set_field_types()

    def set_storage_options(self, storage_options, profile):
        if profile=='wasabi' and storage_options == {}:            
            storage_options = {
                "AWS_NO_SIGN_REQUEST": "yes",
                "S3_ENDPOINT_URL": "https://s3.wasabisys.com",
            }            
        return storage_options
        
    def _set_field_types(self):
        # Determine the type of each field using one raw sample.
        # We bypass any transforms/decoding by directly calling the parent's __getitem__.
        raw_sample = super().__getitem__(0)
        
        self.image_fields = []
        self.field_types = {}
        if hasattr(raw_sample, 'items'):
            for key, value in raw_sample.items():
                if isinstance(value, bytes):
                    # further check if bytes represent an image
                    if is_image_bytes(value):
                        self.field_types[key] = "ImageBytes"
                        self.image_fields.append(key)
                    else:
                        self.field_types[key] = bytes
                else:
                    self.field_types[key] = type(value)
        else:
            for idx, value in enumerate(raw_sample):
                if isinstance(value, bytes):
                    if is_image_bytes(value):
                        self.field_types[idx] = "ImageBytes"
                        self.image_fields.append(idx)
                    else:
                        self.field_types[idx] = bytes
                else:
                    self.field_types[idx] = type(value)
        
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        # If requested, decode any images
        if self.decode_images:
            sample = self._decode_images(sample)

        # Apply image transforms on image fields
        if self._transform is not None:
            if hasattr(sample, 'items'):
                for key, value in sample.items():
                    if key in self.field_types and self.field_types[key] == "ImageBytes":
                        sample[key] = self._transform(value)
            else:
                for i, value in enumerate(sample):
                    if i in self.field_types and self.field_types[i] == "ImageBytes":
                        sample[i] = self._transform(value)
                    
        # Apply pipeline transforms on corresponding fields.
        if self.pipelines is not None:
            for key, transform in self.pipelines.items():
                if key in sample:
                    sample[key] = transform(sample[key])
        
        return sample

    def _decode_images(self, sample):
        if hasattr(sample, 'items'):
            for key, value in sample.items():
                if key in self.field_types and self.field_types[key] == "ImageBytes":
                    sample[key] = decode_image(value, to_pil=True)
        else:
            for i, value in enumerate(sample):
                if i in self.field_types and self.field_types[i] == "ImageBytes":
                    sample[i] = decode_image(value, to_pil=True)

        return sample
        
    def __repr__(self) -> str:
        _repr_indent = 4
        tab = " " * _repr_indent
        head = "StreamingDatasetVisionlab"
        body = [f"Number of datapoints: {len(self)}"]
        if self.input_dir is not None:
            body.append(f"local_dir: {self.input_dir.path}")
            body.append(f"remote_dir: {self.input_dir.url}")
            body.append(f"version: {self.version}")            
            
        body.append(f"\nField Types:")
        for key, field_type in self.field_types.items():
            body.append(f"{key}: {field_type}")
            
        body.append(f"\nPipelines:")
        if self.pipelines is not None:
            for key, pipeline in self.pipelines.items():
                txt = repr(pipeline).replace("\n", f"\n{tab}{tab}")
                body.append(f"{key}: {txt}")
        else:
            body.append(f"None")
    
        body.append(f"\nImage transforms:")
        if self._transform is not None:
            txt = repr(self._transform).replace("\n", f"\n{tab}")
            body.append(f"{txt}")
        else:
            body.append(f"None")
            
        lines = [head] + [tab + line for line in body]

        lines += ['\nUsage Examples:']
        if self.decode_images==False and self.image_fields:            
            lines += [f'{tab}import io']
            lines += [f'{tab}from PIL import Image']            

            lines += [f'\n{tab}# decode image bytes with PIL']
            lines += [f'{tab}sample = dataset[0]']
            for idx in self.image_fields:
                lines += [f"{tab}pil_image = Image.open(io.BytesIO(sample['{idx}']))"]
                    
        elif self.decode_images and self.image_fields:            
            lines += [f'{tab}# image bytes automatically decoded']
            lines += [f'{tab}sample = dataset[0]']
            for idx in self.image_fields:
                lines += [f"{tab}pil_image = sample['{idx}']"]
        else:
            lines += [f'{tab}sample = dataset[0]']
            
        return "\n".join(lines)

def decode_image(image_bytes):
    """Decode image bytes to tensor (CHW format).

    Handles various input types gracefully:
    - torch.Tensor: returned as-is
    - PIL.Image: converted to tensor CHW
    - np.ndarray (HWC): converted to tensor CHW
    - bytes or np.ndarray (1D): decoded using torchvision

    Returns:
        torch.Tensor: RGB image tensor in CHW format, uint8 [0-255]
    """
    # Already a tensor - return as-is
    if isinstance(image_bytes, torch.Tensor):
        return image_bytes

    # Already a PIL Image - convert to tensor
    if isinstance(image_bytes, Image.Image):
        return TF.pil_to_tensor(image_bytes)

    # Already a decoded numpy array (HWC) - convert to tensor (CHW)
    if isinstance(image_bytes, np.ndarray) and image_bytes.ndim > 1:
        return torch.from_numpy(image_bytes).permute(2, 0, 1)

    # Numpy array of bytes - convert to Python bytes
    if isinstance(image_bytes, np.ndarray):
        image_bytes = image_bytes.tobytes()

    # Decode bytes to tensor using torchvision
    img_buffer = torch.frombuffer(image_bytes, dtype=torch.uint8)
    return tv_decode_image(img_buffer, mode=ImageReadMode.RGB)
    
def is_image_bytes(data_bytes: bytes):
    """Check if bytes represent a valid image by attempting to open with PIL."""
    try:
        img = Image.open(io.BytesIO(data_bytes))
        img.verify()
        return True
    except Exception:
        return False

def is_slurm_available():
    return any(var in os.environ for var in ["SLURM_JOB_ID", "SLURM_CLUSTER_NAME"])

def ensure_lightning_symlink_on_cluster():
    if not is_slurm_available():
        return # not on cluster, ignore
    
    home = pathlib.Path.home()
    symlink_path = home / ".lightning"
    target_path = pathlib.Path("/n/netscratch/alvarez_lab/Lab/.lightning")

    # Case 1: symlink already exists and points correctly
    if symlink_path.is_symlink():
        if symlink_path.resolve() == target_path:
            return  # all good
        else:
            raise RuntimeError(
                f"~/.lightning already exists but points to {symlink_path.resolve()}, "
                f"expected {target_path}. Please fix manually."
            )

    # Case 2: ~/.lightning exists but is not a symlink
    if symlink_path.exists():
        raise RuntimeError(
            f"~/.lightning exists but is not a symlink. "
            f"Please remove it and create a symlink to {target_path}."
        )

    # Case 3: doesn’t exist at all → try to create it
    try:
        symlink_path.symlink_to(target_path)
        print(f"Created symlink: {symlink_path} -> {target_path}")
    except Exception as e:
        raise RuntimeError(
            f"Could not create symlink {symlink_path} -> {target_path}. "
            f"Please create it manually."
        ) from e