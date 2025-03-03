import numpy as np
from abc import ABC

from abc import ABC, ABCMeta, abstractmethod
from dataclasses import replace
from typing import Optional, Callable, TYPE_CHECKING, Tuple, Type

from ffcv.fields.rgb_image import SimpleRGBImageDecoder
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.allocation_query import AllocationQuery
from dataclasses import replace

from ffcv.libffcv import imdecode, memcpy, resize_crop, pad
from ffcv.reader import Reader
from ffcv.fields import IntField, RGBImageField

__all__ = ['SimpleSampleReader', 'get_max_sample_size']
           
class SimpleSampleReader(SimpleRGBImageDecoder, ABC):
    """Abstract decoder for :class:`~ffcv.fields.RGBImageField` that does nothing but read your image bytes.
    
    We need to know how big the largest image is, so that we can allocate an array that fits any of them.
    
    You can use the get_max_sample_size function for that:
    
    from ffcv.reader import Reader
    from ffcv.fields import IntField, RGBImageField
    
    def get_max_sample_size(ffcv_file_path, custom_fields=dict(image=RGBImageField, label=IntField)):
        Reader(ffcv_file_path, custom_handlers=custom_fields)
        return max(reader.alloc_table['size'])
    
    Image Bytes can be decoded into an image like so:
        img = Image.open(io.BytesIO(image_bytes))
        
    or using any other jpeg decoder.
    
    """
    def __init__(self, max_size): # testing
        super().__init__()
        self.max_size = max_size

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
               
        my_dtype = np.dtype('<u1')

        return (
            replace(previous_state, jit_mode=True, shape=(self.max_size,), dtype=my_dtype),
            AllocationQuery((self.max_size,), my_dtype),
        )

    def generate_code(self) -> Callable:
        mem_read = self.memory_read
        my_range = Compiler.get_iterator()
        def decode(batch_indices, destination, metadata, storage_state):
            for dst_ix in my_range(len(batch_indices)):
                source_ix = batch_indices[dst_ix]
                field = metadata[source_ix]
                image_data = mem_read(field['data_ptr'], storage_state)  
                data_size = image_data.shape[0]
                destination[dst_ix, 0:data_size]=image_data
            return destination
        decode.is_parallel = True
        return decode
    
def get_max_sample_size(ffcv_file_path, custom_fields=dict(image=RGBImageField, label=IntField)):
    reader = Reader(ffcv_file_path, custom_handlers=custom_fields)
    return max(reader.alloc_table['size'])    