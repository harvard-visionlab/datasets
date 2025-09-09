import numpy as np
import torch
from PIL import Image
from io import BytesIO
import io
from pdb import set_trace
from numba import jit

from abc import ABC, ABCMeta, abstractmethod
from dataclasses import replace
from typing import Optional, Callable, TYPE_CHECKING, Tuple, Type

from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, SimpleRGBImageDecoder, ResizedCropRGBImageDecoder
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.allocation_query import AllocationQuery
from dataclasses import replace

from ffcv.libffcv import imdecode, memcpy, resize_crop, pad

from numba.typed import Dict
IMAGE_MODES = Dict()
IMAGE_MODES['jpg'] = 0
IMAGE_MODES['raw'] = 1

from turbojpeg import TurboJPEG
# turbo = TurboJPEG()

def turbo_decode(img_data):
    return TurboJPEG().decode(img_data)
    
__all__ = ['PILCenterCropRGBImageDecoder']

class PILCenterCropRGBImageDecoder(SimpleRGBImageDecoder, ABC):
    """Abstract decoder for :class:`~ffcv.fields.RGBImageField` that performs a crop and and a resize operation.

    It supports both variable and constant resolution datasets.
    """
    def __init__(self, output_size, scale=(1.0, 1.0), ratio=224/256):
        super().__init__()
        self.scale = scale
        self.ratio = ratio
        self.output_size = output_size

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        widths = self.metadata['width']
        heights = self.metadata['height']
        # We convert to uint64 to avoid overflows
        self.max_width = np.uint64(widths.max())
        self.max_height = np.uint64(heights.max())
        output_shape = (self.output_size[0], self.output_size[1], 3)
        my_dtype = np.dtype('<u1')

        return (
            replace(previous_state, jit_mode=True,
                    shape=output_shape, dtype=my_dtype),
            (AllocationQuery(output_shape, my_dtype),
            AllocationQuery((self.max_height * self.max_width * np.uint64(3),), my_dtype),
            )
        )

    def generate_code(self) -> Callable:

        jpg = IMAGE_MODES['jpg']

        mem_read = self.memory_read
        my_range = Compiler.get_iterator()
        imdecode_c = Compiler.compile(imdecode)
        resize_crop_c = Compiler.compile(resize_crop)
        get_crop_c = Compiler.compile(get_center_crop)
        turbo = Compiler.compile(turbo_decode)
        
        scale = self.scale
        ratio = self.ratio
        if isinstance(scale, tuple):
            scale = np.array(scale)
        if isinstance(ratio, tuple):
            ratio = np.array(ratio)
        
        def decode(batch_indices, my_storage, metadata, storage_state):
            destination, temp_storage = my_storage
            for dst_ix in my_range(len(batch_indices)):
                source_ix = batch_indices[dst_ix]
                field = metadata[source_ix]
                image_data = mem_read(field['data_ptr'], storage_state)
                height = np.uint32(field['height'])
                width = np.uint32(field['width'])

                if field['mode'] == jpg:
                    temp_buffer = temp_storage[dst_ix]
                    # skip decoding, this should be fast right?
                    #imdecode_c(image_data, temp_buffer,
                    #           height, width, height, width, 0, 0, 1, 1, False, False)
                    selected_size = 3 * height * width
                    temp_buffer = temp_buffer.reshape(-1)[:selected_size]
                    temp_buffer = temp_buffer.reshape(height, width, 3)

                else:
                    temp_buffer = image_data.reshape(height, width, 3)

                # i, j, h, w = get_crop_c(height, width, scale, ratio)
                i, j, h, w = get_crop_c(height, width, scale, ratio)

                #resize_crop_c(temp_buffer, i, i + h, j, j + w,
                #              destination[dst_ix])

            return destination[:len(batch_indices)]
        decode.is_parallel = True
        return decode
    
def get_center_crop(height, width, _, ratio):
    s = min(height, width)
    c = int(ratio * s)
    delta_h = (height - c) // 2
    delta_w = (width - c) // 2

    return delta_h, delta_w, c, c    