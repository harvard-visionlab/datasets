from typing import Callable, Tuple
import numpy as np
from abc import ABCMeta
from ffcv.fields import rgb_image, ndarray
from numba.typed import Dict

from ffcv.pipeline.compiler import Compiler
from ffcv.libffcv import imdecode, resize_crop, pad

from dataclasses import replace

import numpy as np
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.allocation_query import AllocationQuery
import random
from ffcv.fields.rgb_image import (
    SimpleRGBImageDecoder,
)
from numba import njit
from ffcv.fields.basics import IntDecoder

from ffcv.transforms.resized_crop import IMAGE_MODES, get_random_crop, imdecode, resize_crop

class RandomResizedMultiCrop(rgb_image.RandomResizedCropRGBImageDecoder):
    """Decoder for :class:`~ffcv.fields.RGBImageField` that performs a Random crop and and a resize operation.
    It supports both variable and constant resolution datasets.
    Parameters
    ----------
    output_size : Tuple[int]
        The desired resized resolution of the images
    scale : Tuple[float]
        The range of possible ratios (in area) than can randomly sampled
    ratio : Tuple[float]
        The range of potential aspect ratios that can be randomly sampled
    """

    def __init__(self, output_size, n_crops=1, seed=None, second_seed=None, scale=(0.08, 1.0), ratio=(3/4, 4/3)):
        super().__init__(output_size, scale=scale, ratio=ratio)
        self.n_crops = n_crops
        self.seed = seed
        self.second_seed = second_seed
        self.counter = 0
        self.get_crop = get_random_crop

    def generate_code(self) -> Callable:

        jpg = 0

        mem_read = self.memory_read
        my_range = Compiler.get_iterator()
        imdecode_c = Compiler.compile(imdecode)
        resize_crop_c = Compiler.compile(resize_crop)
        get_crop = self.get_crop
        
        n_crops = self.n_crops
        scale = self.scale
        ratio = self.ratio
        if isinstance(scale, tuple):
            scale = np.array(scale)
        if isinstance(ratio, tuple):
            ratio = np.array(ratio)

        if self.seed is None and self.second_seed is None:

            def decode(indices, my_storage, metadata, storage_state):
                destination, temp_storage = my_storage
                for dst_ix in my_range(len(indices)):
                    source_ix = indices[dst_ix]
                    field = metadata[source_ix]
                    image_data = mem_read(field["data_ptr"], storage_state)
                    height = np.uint32(field["height"])
                    width = np.uint32(field["width"])

                    if field["mode"] == jpg:
                        temp_buffer = temp_storage[dst_ix]
                        imdecode_c(
                            image_data,
                            temp_buffer,
                            height,
                            width,
                            height,
                            width,
                            0,
                            0,
                            1,
                            1,
                            False,
                            False,
                        )
                        selected_size = 3 * height * width
                        temp_buffer = temp_buffer.reshape(-1)[:selected_size]
                        temp_buffer = temp_buffer.reshape(height, width, 3)

                    else:
                        temp_buffer = image_data.reshape(height, width, 3)
                    
                    for crop_idx in my_range(n_crops):
                        i, j, h, w = get_crop(
                            height,
                            width,
                            scale,
                            ratio,
                        )
                        resize_crop_c(temp_buffer, i, i + h, j, j + w, destination[dst_ix, crop_idx])
                
                return destination[:len(indices)]

            decode.is_parallel = True
            return decode

        if self.seed is not None and self.second_seed is None:
            seed = self.seed
            def decode(indices, my_storage, metadata, storage_state, counter):

                destination, temp_storage = my_storage
                for dst_ix in my_range(len(indices)):
                    source_ix = indices[dst_ix]
                    field = metadata[source_ix]
                    image_data = mem_read(field["data_ptr"], storage_state)
                    height = np.uint32(field["height"])
                    width = np.uint32(field["width"])

                    if field["mode"] == jpg:
                        temp_buffer = temp_storage[dst_ix]
                        imdecode_c(
                            image_data,
                            temp_buffer,
                            height,
                            width,
                            height,
                            width,
                            0,
                            0,
                            1,
                            1,
                            False,
                            False,
                        )
                        selected_size = 3 * height * width
                        temp_buffer = temp_buffer.reshape(-1)[:selected_size]
                        temp_buffer = temp_buffer.reshape(height, width, 3)

                    else:
                        temp_buffer = image_data.reshape(height, width, 3)
                    
                    for crop_idx in my_range(n_crops):
                        i, j, h, w = get_crop(
                            height,
                            width,
                            scale,
                            ratio,
                            seed+dst_ix+indices.shape[0]*counter
                        )
                        resize_crop_c(temp_buffer, i, i + h, j, j + w, destination[dst_ix,crop_idx])

                return destination[:len(indices)]

            decode.is_parallel = True
            decode.with_counter = True
            return decode


        if self.seed is not None and self.second_seed is not None:
            seed = self.seed
            second_seed = self.second_seed
            def decode(indices, my_storage, metadata, storage_state, counter):

                destination, temp_storage = my_storage
                for dst_ix in my_range(len(indices)):
                    source_ix = indices[dst_ix]
                    field = metadata[source_ix]
                    image_data = mem_read(field["data_ptr"], storage_state)
                    height = np.uint32(field["height"])
                    width = np.uint32(field["width"])

                    if field["mode"] == jpg:
                        temp_buffer = temp_storage[dst_ix]
                        imdecode_c(
                            image_data,
                            temp_buffer,
                            height,
                            width,
                            height,
                            width,
                            0,
                            0,
                            1,
                            1,
                            False,
                            False,
                        )
                        selected_size = 3 * height * width
                        temp_buffer = temp_buffer.reshape(-1)[:selected_size]
                        temp_buffer = temp_buffer.reshape(height, width, 3)

                    else:
                        temp_buffer = image_data.reshape(height, width, 3)
                    
                    for crop_idx in my_range(n_crops):
                        i, j, h, w = get_crop(
                            height,
                            width,
                            scale,
                            ratio,
                            seed+dst_ix+indices.shape[0]*counter,
                            second_seed+dst_ix+indices.shape[0]*counter
                        )
                        resize_crop_c(temp_buffer, i, i + h, j, j + w, destination[dst_ix,crop_idx])

                return destination[:len(indices)]

            decode.is_parallel = True
            decode.with_counter = True
            return decode

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:

        widths = self.metadata["width"]
        heights = self.metadata["height"]
        # We convert to uint64 to avoid overflows
        self.max_width = np.uint64(widths.max())
        self.max_height = np.uint64(heights.max())
        output_shape = (self.n_crops, self.output_size[0], self.output_size[1], 3)
        my_dtype = np.dtype("<u1")
        return (
            replace(previous_state, jit_mode=True, shape=output_shape, dtype=my_dtype),
            (
                AllocationQuery(output_shape, my_dtype),
                AllocationQuery(
                    (self.max_height * self.max_width * np.uint64(3),), my_dtype
                ),
            ),
        )