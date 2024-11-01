import numpy as np
import cv2
from PIL.Image import Image

from typing import Optional, Callable, TYPE_CHECKING, Tuple, Type

from ffcv.fields.base import Field, ARG_TYPE
from ffcv.fields.rgb_image import SimpleRGBImageDecoder
from ffcv.pipeline.operation import Operation

from numba.typed import Dict

from pdb import set_trace

IMAGE_MODES = Dict()
IMAGE_MODES['jpg'] = 0
IMAGE_MODES['raw'] = 1

def encode_jpeg(numpy_image, quality):
    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    success, result = cv2.imencode('.jpg', numpy_image,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    if not success:
        raise ValueError("Impossible to encode image in jpeg")

    return result.reshape(-1)

def resizer(image, short_resolution_min, long_resolution_max, max_method):
    image = resize_shortest_min(image, short_resolution_min)
    image = resize_longest_max(image, long_resolution_max, max_method)
    
    return image
    
def resize_shortest_min(image, target_resolution):
    '''make the shortest edge this long'''
    if target_resolution is None:
        return image
    
    orig_height,orig_width = image.shape[0], image.shape[1]
    
    # set shortest size to target_resolution, keep aspect ratio
    if orig_width < orig_height:
        new_width = target_resolution
        ratio = target_resolution/orig_width
        new_height = int(round(ratio * orig_height))
    else:
        new_height = target_resolution
        ratio = target_resolution/orig_height
        new_width = int(round(ratio * orig_width))

    new_size = (new_width,new_height)
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

def resize_longest_max(image, target_resolution, max_method):
    '''make the longest edge at most `target_resolution` long by either cropping or squeezing'''
    if target_resolution is None:
        return image
    if max_method == 'center_crop':
        image = center_crop_longest_max(image, target_resolution)
    elif max_method == 'squeeze_resize':
        image = squeeze_resize_longest_max(image, target_resolution)
    return image

def center_crop_longest_max(image, target_resolution):
    if target_resolution is None:
        return image
    
    orig_height,orig_width = image.shape[0], image.shape[1]
    
    # center crop longest edge
    if orig_width > orig_height and orig_width > target_resolution:
        # crop along width
        center = orig_width / 2
        start = round(center - target_resolution/2)
        end = start + target_resolution
        image = image[:, int(start):int(end)]
    elif orig_height > orig_width and orig_height > target_resolution:
        # crop along height
        center = orig_height / 2
        start = round(center - target_resolution/2)
        end = start + target_resolution
        image = image[int(start):int(end),:]
    
    return image

def squeeze_resize_longest_max(image, target_resolution):
    if target_resolution is None:
        return image
    
    orig_height,orig_width = image.shape[0], image.shape[1]
    
    # squeeze longest edge to max (if it's greater than target_resolution)
    if orig_width > orig_height and orig_width > target_resolution:
        # width is longest
        new_size = np.array((target_resolution, orig_height)).astype(int)
        image = cv2.resize(image, tuple(new_size), interpolation=cv2.INTER_AREA)
    elif orig_height > orig_width and orig_height > target_resolution:
        # height is longest, set height to target_resolution
        new_size = np.array((orig_width, target_resolution)).astype(int)
        image = cv2.resize(image, tuple(new_size), interpolation=cv2.INTER_AREA)
        
    return image

class RGBImageField(Field):
    """
    A subclass of :class:`~ffcv.fields.Field` supporting RGB image data.
    Parameters
    ----------
    write_mode : str, optional
        How to write the image data to the dataset file. Should be either 'raw'
        (``uint8`` pixel values), 'jpg' (compress to JPEG format), 'smart'
        (decide between saving pixel values and JPEG compressing based on image
        size), and 'proportion' (JPEG compress a random subset of the data with
        size specified by the ``compress_probability`` argument). By default: 'jpg'.
    min_resolution : int, optional
        If specified, will resize images to have shortest side length equal to
        this value before saving, by default None
    max_resolution : int, optional
        If specified, then the longest edge will be adjusted to this size if 
        the longest edge is greater than this size, by default None        
    max_enforced_with: str, `center_crop` or `squeeze_resize`, how the
        maximum resolution is achieve (either center cropping or squeezing/resizing 
        along the longest edge), default `center_crop`
    smart_threshold : int, optional
        When `write_mode='smart`, will compress an image if it would take more than `smart_threshold` times to use RAW instead of jpeg.
    jpeg_quality : int, optional
        The quality parameter for JPEG encoding (ignored for
        ``write_mode='raw'``), by default 100
    compress_probability : float, optional
        Ignored unless ``write_mode='proportion'``; in the latter case it is the
        probability with which image is JPEG-compressed, by default 0.5.
    """
    def __init__(self, write_mode='jpg', min_resolution: int = None,
                 max_resolution: int = None, max_enforced_with: str = 'center_crop',
                 smart_threshold: int = None,  jpeg_quality: int = 100, 
                 compress_probability: float = 0.5) -> None:
        self.write_mode = write_mode
        self.smart_threshold = smart_threshold
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.max_enforced_with = max_enforced_with
        self.jpeg_quality = int(jpeg_quality)
        self.proportion = compress_probability

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('mode', '<u1'),
            ('width', '<u2'),
            ('height', '<u2'),
            ('data_ptr', '<u8'),
        ])

    def get_decoder_class(self) -> Type[Operation]:
        return SimpleRGBImageDecoder

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        return RGBImageField()

    def to_binary(self) -> ARG_TYPE:
        return np.zeros(1, dtype=ARG_TYPE)[0]

    def encode(self, destination, image, malloc):
        if isinstance(image, Image):
            image = np.array(image)

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Unsupported image type {type(image)}")

        if image.dtype != np.uint8:
            raise ValueError("Image type has to be uint8")

        if image.shape[2] != 3:
            raise ValueError(f"Invalid shape for rgb image: {image.shape}")

        assert image.dtype == np.uint8

        image = resizer(image, self.min_resolution, self.max_resolution, self.max_enforced_with)

        write_mode = self.write_mode
        as_jpg = None

        if write_mode == 'smart':
            as_jpg = encode_jpeg(image, self.jpeg_quality)
            write_mode = 'raw'
            if self.smart_threshold is not None:
                if image.nbytes > self.smart_threshold:
                    write_mode = 'jpg'
        elif write_mode == 'proportion':
            if np.random.rand() < self.proportion:
                write_mode = 'jpg'
            else:
                write_mode = 'raw'

        destination['mode'] = IMAGE_MODES[write_mode]
        destination['height'], destination['width'] = image.shape[:2]

        if write_mode == 'jpg':
            if as_jpg is None:
                as_jpg = encode_jpeg(image, self.jpeg_quality)
            destination['data_ptr'], storage = malloc(as_jpg.nbytes)
            storage[:] = as_jpg
        elif write_mode == 'raw':
            image_bytes = np.ascontiguousarray(image).view('<u1').reshape(-1)
            destination['data_ptr'], storage = malloc(image.nbytes)
            storage[:] = image_bytes
        else:
            raise ValueError(f"Unsupported write mode {self.write_mode}")