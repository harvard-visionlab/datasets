import io
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from litdata import StreamingDataset
from litdata.utilities.dataset_utilities import _read_updated_at
from copy import deepcopy
from pdb import set_trace

try:
    # https://github.com/lilohuang/PyTurboJPEG/blob/master/turbojpeg.py
    from turbojpeg import TurboJPEG, TJPF_RGB, TJPF_BGR
except:
    # https://github.com/loopbio/PyTurboJPEG/tree/loopbio
    from turbojpeg import TurboJPEG, TJPF
    TJPF_RGB = TJPF.RGB
    TJPF_BGR = TJPF.BGR
    
turbo = TurboJPEG()

try:
    from litdata.streaming.downloader import get_downloader_cls
except:
    from litdata.streaming.downloader import get_downloader as get_downloader_cls
from visionlab.datasets.utils.s3_sync import s3_sync_data

class StreamingDatasetVisionlab(StreamingDataset):
    '''
    Visionlab version of StreamingDataset with pipelines for field transforms,
    automatic image decoding if requested, and pre-determined field type info.
    '''
    def __init__(self, *args, transform=None, pipelines=None, decode_images=False, expected_version=None, 
                 profile=None, storage_options={}, max_cache_size='350GB', **kwargs):
        # pipelines: dict mapping field names to transform functions.
        # decode_images: if True, bytes that are valid image bytes are decoded automatically
        self.pipelines = pipelines
        self.transform = transform        
        if self.pipelines is not None and self.transform is not None:
            raise ValueError('pipelines and transform cannot both be set. If you only need to transform images, you can use transform. To specify different transforms per field, use pipelines.')
        self.decode_images = decode_images
        storage_options = self.set_storage_options(storage_options, profile)
        super().__init__(*args, storage_options=storage_options, max_cache_size=max_cache_size, **kwargs)
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
    
    def download_all_chunks(self, force_sync=False):
        # get local and remote directories
        local_dir = self.input_dir.path
        remote_dir = self.input_dir.url
        
        # determine if files should be synced
        filepaths = [os.path.join(local_dir, filename) for filename in self.subsampled_files]        
        all_files_exist = all([os.path.isfile(path) for path in filepaths])
        should_sync_files = force_sync or not all_files_exist
        
        if should_sync_files and isinstance(remote_dir, str) and remote_dir.startswith("s3://"):
            s3_sync_data(
                from_dir=remote_dir, 
                to_dir=local_dir, 
                size_only=True, 
                include="*.bin", 
                storage_options=self.cache._reader._storage_options
            )
        elif should_sync_files:
            # all chunk indices
            chunk_indexes = [ChunkedIndex(*self.cache._get_chunk_index_from_index(index)) for index in range(len(self))]
            # index of first item in each chunk 
            unique_chunk_indexes = {}
            for chunk in chunk_indexes:
                if chunk.chunk_index not in unique_chunk_indexes:
                    unique_chunk_indexes[chunk.chunk_index] = chunk
            # Get the filtered list of ChunkedIndex objects
            filtered_chunk_indexes = list(unique_chunk_indexes.values())
            # force download by loading first item in each chunk
            for index in tqdm(filtered_chunk_indexes):
                _ = self[index]
        elif all_files_exist:
            print(f"==> All files in local cache: {local_dir}, skipping sync")
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        # If requested, decode any images
        if self.decode_images:
            sample = self._decode_images(sample)

        # Apply image transforms on image fields
        if self.transform is not None:
            if hasattr(sample, 'items'):
                for key, value in sample.items():
                    if key in self.field_types and self.field_types[key] == "ImageBytes":
                        sample[key] = self.transform(value)
            else:
                for i, value in enumerate(sample):
                    if i in self.field_types and self.field_types[i] == "ImageBytes":
                        sample[i] = self.transform(value)
                    
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
        if self.transform is not None:
            txt = repr(self.transform).replace("\n", f"\n{tab}")
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

def decode_image(image_bytes, to_pil=False, pixel_format=TJPF_RGB):   
    if isinstance(image_bytes, Image.Image):
        return image_bytes

    if isinstance(image_bytes, np.ndarray) and image_bytes.ndim > 1:
        return Image.fromarray(image_bytes) if to_pil else image_bytes
        
    # convert numpy arrays to bytes
    if isinstance(image_bytes, np.ndarray):
        image_bytes = image_bytes.tobytes()

    if is_jpeg_bytes(image_bytes):
        rgb_image = turbo.decode(image_bytes, pixel_format=pixel_format)
    else:
        try:
            rgb_image = Image.open(io.BytesIO(image_bytes))
            rgb_image.verify()
            rgb_image = rgb_image.convert('RGB')
        except:
            # Decode to BGR
            bgr_image = cv2.imdecode(np.array(image_bytes), cv2.IMREAD_COLOR)
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    if to_pil and isinstance(rgb_image, np.ndarray):
        rgb_image = Image.fromarray(rgb_image)

    return rgb_image
    
def is_image_bytes(data_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(data_bytes))
        img.verify()
        return True
    except Exception:
        return False

def is_jpeg_bytes(image_bytes):
    """
    Check if the given bytes represent a JPEG image by examining the magic bytes.
    
    JPEG files typically start with one of these signatures:
    - JFIF: bytes [0xFF, 0xD8, 0xFF, 0xE0] + "JFIF"
    - Exif: bytes [0xFF, 0xD8, 0xFF, 0xE1] + "Exif"
    - Other JPEG: bytes [0xFF, 0xD8, 0xFF] (general marker)
    
    Args:
        image_bytes (bytes): The image data to check
        
    Returns:
        bool: True if the image is a JPEG, False otherwise
    """
    # Check if we have enough bytes to determine the format
    if len(image_bytes) < 3:
        return False
        
    # All JPEG files start with FF D8 FF
    return image_bytes[0:3] == b'\xFF\xD8\xFF'
