"""
Random horizontal flip

There's a bug in the original where they ALWAYS FLIP if the seed is set:
https://github.com/facebookresearch/FFCV-SSL/blob/main/ffcv/transforms/flip.py

GAA: fixed to flip only with prob flip_prob when seed is set
GAA: dropped the label thingy (not sure why it's needed if you can set the seed for all transforms)
"""
import random
import numpy as np
from dataclasses import replace
from numpy.random import rand
from numpy.random import default_rng
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler

__all__ = ['RandomHorizontalFlip']

class RandomHorizontalFlip(Operation):
    """Flip the image horizontally with probability flip_prob.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    flip_prob : float
        The probability with which to flip each image in the batch
        horizontally.
    """

    def __init__(self, flip_prob: float = 0.5, seed: int = None):
        super().__init__()
        self.flip_prob = flip_prob
        self.seed = seed

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        flip_prob = self.flip_prob
        seed = self.seed

        if seed is None:
            def flip(images, dst):
                should_flip = rand(images.shape[0]) < flip_prob
                for i in my_range(images.shape[0]):
                    if should_flip[i]:
                        dst[i] = images[i, :, ::-1]
                    else:
                        dst[i] = images[i]

                return dst

            flip.is_parallel = True
            return flip

        def flip(images, dst, counter):
            random.seed(seed + counter)
            should_flip = np.zeros(len(images))
            for i in range(len(images)):
                should_flip[i] = random.uniform(0, 1) < flip_prob # fixed
            for i in my_range(images.shape[0]):
                if should_flip[i]:
                    dst[i] = images[i, :, ::-1]
                else:
                    dst[i] = images[i]
            return dst
        flip.is_parallel = True
        flip.with_counter = True
        return flip


    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), 
                AllocationQuery(previous_state.shape, previous_state.dtype))
