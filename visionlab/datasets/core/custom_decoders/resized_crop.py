"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

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

SimpleDecoder = SimpleRGBImageDecoder

IMAGE_MODES = Dict()
IMAGE_MODES['jpg'] = 0
IMAGE_MODES['raw'] = 1

# GAA
@njit(parallel=False, fastmath=True, inline='always')
def random_choice(values, seed=None):
    if seed is not None:
        np.random.seed(seed)
    index = np.random.randint(0, len(values))
    return values[index]

@njit(parallel=False, fastmath=True, inline='always')
def random_ratio(ratios, seed=None):
    if seed is not None:
        np.random.seed(seed)
    index = np.random.randint(0, len(ratios))
    ratio = ratios[index]
    
    return np.array([ratio,ratio])

# This function can take an optinal second seed to return a second crop
# that is not overlapping with the first one.
# GAA edited to actually allow non-square crops!
@njit(parallel=False, fastmath=True, inline='always')
def get_random_crop(height, width, scale, ratio, seed=None, second_seed=None):
    area = height * width
    log_ratio = np.log(ratio)

    if second_seed is not None:
        np.random.seed(second_seed)
        for _ in range(10):
            target_area = area * np.random.uniform(scale[0], scale[1])
            aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))

            w2 = int(round(np.sqrt(target_area * aspect_ratio)))
            h2 = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < w2 <= width and 0 < h2 <= height:
                i2 = int(np.random.uniform(0, height - h2 + 1))
                j2 = int(np.random.uniform(0, width - w2 + 1))
                return i2, j2, h2, w2

    if seed is not None:
        np.random.seed(seed)

    for _ in range(10):
        target_area = area * np.random.uniform(scale[0], scale[1])
        aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))

        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = int(np.random.uniform(0, height - h + 1))
            j = int(np.random.uniform(0, width - w + 1))
            return i, j, h, w

    # Fallback to central crop if no valid crop is found
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:
        w = width // 2
        h = height // 2

    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w

@njit(parallel=False, fastmath=True, inline="always")
def get_center_crop(height, width, _, ratio, seed=None, second_seed=None):
    s = min(height, width)
    c = int(ratio * s)
    delta_h = (height - c) // 2
    delta_w = (width - c) // 2

    return delta_h, delta_w, c, c

@njit(parallel=False, fastmath=True, inline="always")
def get_top_left_crop(height, width, _, ratio, seed=None, second_seed=None):
    s = min(height, width)
    c = int(ratio * s)
    delta_h = (height - c) // 2
    delta_w = (width - c) // 2
    return height - c, 0, c, c

@njit(parallel=False, fastmath=True, inline="always")
def get_top_right_crop(height, width, _, ratio, seed=None, second_seed=None):
    s = min(height, width)
    c = int(ratio * s)
    delta_h = (height - c) // 2
    delta_w = (width - c) // 2
    return height - c, width - c, c, c

@njit(parallel=False, fastmath=True, inline="always")
def get_bottom_left_crop(height, width, _, ratio, seed=None, second_seed=None):
    s = min(height, width)
    c = int(ratio * s)
    delta_h = (height - c) // 2
    delta_w = (width - c) // 2
    return 0, 0, c, c

@njit(parallel=False, fastmath=True, inline="always")
def get_bottom_right_crop(height, width, _, ratio, seed=None, second_seed=None):
    s = min(height, width)
    c = int(ratio * s)
    delta_h = (height - c) // 2
    delta_w = (width - c) // 2
    return 0, width - c, c, c

class RandomResizedCrop(rgb_image.RandomResizedCropRGBImageDecoder):
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

    def __init__(self, *args, seed=None, second_seed=None, **kwargs):
        super().__init__(*args, **kwargs)
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

                    i, j, h, w = get_crop(
                        height,
                        width,
                        scale,
                        ratio,
                    )
                    resize_crop_c(temp_buffer, i, i + h, j, j + w, destination[dst_ix])

                return destination[: len(indices)]

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

                    i, j, h, w = get_crop(
                        height,
                        width,
                        scale,
                        ratio,
                        seed+dst_ix+indices.shape[0]*counter
                    )
                    resize_crop_c(temp_buffer, i, i + h, j, j + w, destination[dst_ix])

                return destination[: len(indices)]

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

                    i, j, h, w = get_crop(
                        height,
                        width,
                        scale,
                        ratio,
                        seed+dst_ix+indices.shape[0]*counter,
                        second_seed+dst_ix+indices.shape[0]*counter
                    )
                    resize_crop_c(temp_buffer, i, i + h, j, j + w, destination[dst_ix])

                return destination[: len(indices)]

            decode.is_parallel = True
            decode.with_counter = True
            return decode

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, AllocationQuery]:

        widths = self.metadata["width"]
        heights = self.metadata["height"]
        # We convert to uint64 to avoid overflows
        self.max_width = np.uint64(widths.max())
        self.max_height = np.uint64(heights.max())
        output_shape = (self.output_size[0], self.output_size[1], 3)
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
    
class RandomResizedCropSampleRatios(rgb_image.RandomResizedCropRGBImageDecoder):
    """Decoder for :class:`~ffcv.fields.RGBImageField` that performs a Random crop and and a resize operation.
    It supports both variable and constant resolution datasets.
    Parameters
    ----------
    output_size : Tuple[int]
        The desired resized resolution of the images
    scale : Tuple[float]
        The range of possible ratios (in area) than can randomly sampled
    ratio : Tuple[float]
        The set of potential aspect ratios that will be randomly sampled
    """

    def __init__(self, *args, seed=None, second_seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.second_seed = second_seed
        self.counter = 0
        self.get_crop = get_random_crop
        self.choose_ratio = random_ratio
        
    def generate_code(self) -> Callable:

        jpg = 0

        mem_read = self.memory_read
        my_range = Compiler.get_iterator()
        imdecode_c = Compiler.compile(imdecode)
        resize_crop_c = Compiler.compile(resize_crop)
        get_crop = self.get_crop
        choose_ratio = self.choose_ratio
        
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
                    
                    curr_ratio = choose_ratio(ratio)
                    
                    i, j, h, w = get_crop(
                        height,
                        width,
                        scale,
                        curr_ratio,
                    )
                    resize_crop_c(temp_buffer, i, i + h, j, j + w, destination[dst_ix])

                return destination[: len(indices)]

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
                    
                    curr_ratio = choose_ratio(ratio, seed=seed+dst_ix+indices.shape[0]*counter)
                    
                    i, j, h, w = get_crop(
                        height,
                        width,
                        scale,
                        curr_ratio,
                        seed+dst_ix+indices.shape[0]*counter
                    )
                    resize_crop_c(temp_buffer, i, i + h, j, j + w, destination[dst_ix])

                return destination[: len(indices)]

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
                    
                    curr_ratio = choose_ratio(ratio, seed=seed+dst_ix+indices.shape[0]*counter)
                    
                    i, j, h, w = get_crop(
                        height,
                        width,
                        scale,
                        curr_ratio,
                        seed+dst_ix+indices.shape[0]*counter,
                        second_seed+dst_ix+indices.shape[0]*counter
                    )
                    resize_crop_c(temp_buffer, i, i + h, j, j + w, destination[dst_ix])

                return destination[: len(indices)]

            decode.is_parallel = True
            decode.with_counter = True
            return decode

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, AllocationQuery]:

        widths = self.metadata["width"]
        heights = self.metadata["height"]
        # We convert to uint64 to avoid overflows
        self.max_width = np.uint64(widths.max())
        self.max_height = np.uint64(heights.max())
        output_shape = (self.output_size[0], self.output_size[1], 3)
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


class LabelRandomResizedCrop(rgb_image.RandomResizedCropRGBImageDecoder):#IntDecoder, ndarray.NDArrayDecoder):
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

    def __init__(
        self,
        output_size,
        seed,
        scale,
        ratio,
        second_seed=None
    ):
        self.scale = scale
        self.ratio = ratio
        self.output_size = output_size
        self.seed = seed
        self.second_seed = second_seed

    def generate_code(self) -> Callable:

        my_range = Compiler.get_iterator()
        get_crop = get_random_crop

        scale = self.scale
        ratio = self.ratio
        if isinstance(scale, tuple):
            scale = np.array(scale)
        if isinstance(ratio, tuple):
            ratio = np.array(ratio)

        seed = self.seed
        second_seed = self.second_seed
        mem_read = self.memory_read

        if second_seed is None:

            def decode(indices, my_storage, metadata, storage_state, counter):
                old_alloc, destination = my_storage
                for ix in my_range(indices.shape[0]):
                    sample_id = indices[ix]
                    field = metadata[sample_id]
                    height = np.int32(field["height"])
                    width = np.int32(field["width"])

                    i, j, h, w = get_crop(
                        height,
                        width,
                        scale,
                        ratio,
                        seed+ix+indices.shape[0]*counter,
                    )
                    destination[ix, 0] = width
                    destination[ix, 1] = height
                    destination[ix, 2] = i
                    destination[ix, 3] = j
                    destination[ix, 4] = h
                    destination[ix, 5] = w
                return destination[:len(indices)]

            decode.is_parallel = True
            decode.with_counter = True
            return decode

        else:
            def decode(indices, my_storage, metadata, storage_state, counter):
                old_alloc, destination = my_storage
                for ix in my_range(indices.shape[0]):
                    sample_id = indices[ix]
                    field = metadata[sample_id]
                    height = np.int32(field["height"])
                    width = np.int32(field["width"])

                    i, j, h, w = get_crop(
                        height,
                        width,
                        scale,
                        ratio,
                        seed+ix+indices.shape[0]*counter,
                        second_seed+ix+indices.shape[0]*counter
                    )
                    destination[ix, 0] = width
                    destination[ix, 1] = height
                    destination[ix, 2] = i
                    destination[ix, 3] = j
                    destination[ix, 4] = h
                    destination[ix, 5] = w
                return destination[:len(indices)]

            decode.is_parallel = True
            decode.with_counter = True
            return decode

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, AllocationQuery]:

        widths = self.metadata["width"]
        heights = self.metadata["height"]
        # We convert to uint64 to avoid overflows
        self.max_width = np.uint64(widths.max())
        self.max_height = np.uint64(heights.max())
        output_shape = (self.output_size[0], self.output_size[1], 3)
        my_dtype = np.dtype("<u1")
        return (
            replace(previous_state, jit_mode=True, shape=(6,), dtype="int32"),
            (
                AllocationQuery(output_shape, my_dtype),
                AllocationQuery((6,), "int32"),
            ),
        )

class CenterCrop(RandomResizedCrop):
    """Decoder for :class:`~ffcv.fields.RGBImageField` that performs a center crop followed by a resize operation.
    It supports both variable and constant resolution datasets.
    Parameters
    ----------
    output_size : Tuple[int]
        The desired resized resolution of the images
    ratio: float
        ratio of (crop size) / (min side length)
    """

    # output size: resize crop size -> output size
    def __init__(self, output_size, ratio):
        super().__init__(output_size)
        self.scale = None
        self.ratio = ratio
        self.get_crop = get_center_crop

class CornerCrop(RandomResizedCrop):
    """Decoder for :class:`~ffcv.fields.RGBImageField` that performs a center crop followed by a resize operation.
    It supports both variable and constant resolution datasets.
    Parameters
    ----------
    output_size : Tuple[int]
        The desired resized resolution of the images
    ratio: float
        ratio of (crop size) / (min side length)
    """

    # output size: resize crop size -> output size
    def __init__(self, output_size, ratio, corner="top_left"):
        super().__init__(output_size)
        self.scale = None
        self.ratio = ratio
        if corner == "top_left":
            self.get_crop = get_top_left_crop
        elif corner == "top_right":
            self.get_crop = get_top_right_crop
        elif corner == "bottom_left":
            self.get_crop = get_bottom_left_crop
        elif corner == "bottom_right":
            self.get_crop = get_bottom_right_crop

class PadRGBImageDecoder(rgb_image.SimpleRGBImageDecoder, metaclass=ABCMeta):
    """Abstract decoder for :class:`~ffcv.fields.RGBImageField` that performs a padding.

    It supports both variable and constant resolution datasets. 
    If using variable resolution datasets, all images will be pad de the maximum resolution in the dataset.
    """
    def __init__(self):
        super().__init__()

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        widths = self.metadata['width']
        heights = self.metadata['height']
        # We convert to uint64 to avoid overflows
        self.max_width = np.uint64(widths.max())
        self.max_height = np.uint64(heights.max())
        output_shape = (self.max_height, self.max_width, 3)
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
        pad_c = Compiler.compile(pad)

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
                    imdecode_c(image_data, temp_buffer,
                               height, width, height, width, 0, 0, 1, 1, False, False)
                    selected_size = 3 * height * width
                    temp_buffer = temp_buffer.reshape(-1)[:selected_size]
                    temp_buffer = temp_buffer.reshape(height, width, 3)

                else:
                    temp_buffer = image_data.reshape(height, width, 3)

                i = 0
                j = 0
                h = height
                w = width

                pad_c(temp_buffer, i, i + h, j, j + w,
                              destination[dst_ix])

            return destination[:len(batch_indices)]
        decode.is_parallel = True
        return decode