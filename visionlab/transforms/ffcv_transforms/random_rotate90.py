"""
Random rotate 90 degree steps
"""
import random
import numpy as np
from dataclasses import replace
from numpy.random import rand
from numpy.random import default_rng
from typing import Callable, Optional, Tuple, List
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler

__all__ = ['RandomRotate90']

class RandomRotate90(Operation):
    """Randomly rotate with probability from set of angles.

    Parameters
    ----------
    rotate_prob : float
        The probability with which to rotate each image.
    """

    def __init__(self, rotate_prob: float = 0.5, steps: List = [0,1,2,3], seed: int = None):
        super().__init__()
        self.rotate_prob = rotate_prob
        self.steps = steps
        self.seed = seed

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        rotate_prob = self.rotate_prob
        steps = np.array(self.steps, dtype=np.int32)
        seed = self.seed

        if seed is None:
            def rotate(images, dst):
                should_rotate = rand(images.shape[0]) <= rotate_prob
                for i in my_range(images.shape[0]):
                    if should_rotate[i]:
                        rotate_steps = steps[random.randint(0, len(steps) - 1)]
                        dst[i] = np.rot90(images[i], k=rotate_steps)  # Rotate image
                    else:
                        dst[i] = images[i]
                return dst

            rotate.is_parallel = True
            return rotate

        def rotate(images, dst, counter):
            random.seed(seed + counter)
            should_rotate = np.zeros(len(images), dtype=np.int32)
            rotate_steps = np.zeros(len(images), dtype=np.int32)
            for i in my_range(images.shape[0]):
                should_rotate[i] = random.uniform(0, 1) < rotate_prob
                rotate_steps[i] = steps[random.randint(0, len(steps) - 1)]
            for i in my_range(images.shape[0]):
                if should_rotate[i]:
                    dst[i] = np.rot90(images[i], k=rotate_steps[i])  # Rotate image "rotate_steps[i]" times
                else:
                    dst[i] = images[i]
            return dst
        rotate.is_parallel = True
        rotate.with_counter = True
        return rotate


    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), 
                AllocationQuery(previous_state.shape, previous_state.dtype))
