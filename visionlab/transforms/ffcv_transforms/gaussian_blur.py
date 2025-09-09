import numpy as np
import random
from torch.nn.modules.utils import _pair
from typing import Callable, Optional, Tuple
from dataclasses import replace
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from scipy.signal import convolve2d

__all__ = ['GaussianBlur']

class GaussianBlur(Operation):
    """Blurs image with randomly chosen Gaussian blur.
    
    Only works for numpy arrays, not TensorImages ([..., C, H, W])            

    Args:
        blur_prob (float): probability to apply blurring to each input
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.
    """

    def __init__(self, blur_prob, kernel_size=5, sigma=(0.1, 2.0), seed=None):
        super().__init__()
        self.blur_prob = blur_prob
        self.kernel_size = kernel_size
        assert sigma[1] > sigma[0]
        self.sigmas = np.linspace(sigma[0], sigma[1], 10)
        from scipy import signal

        self.weights = np.stack(
            [
                signal.gaussian(kernel_size, s)
                for s in np.linspace(sigma[0], sigma[1], 10)
            ]
        )
        self.weights /= self.weights.sum(1, keepdims=True)
        self.seed = seed

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        blur_prob = self.blur_prob
        kernel_size = self.kernel_size
        weights = self.weights
        num_kernels = weights.shape[0]
        seed = self.seed
        
        if seed is None:
            # no seed, use np.random.rand; 
            def blur(images, _):

                for i in my_range(images.shape[0]):
                    if np.random.rand() < blur_prob:
                        k = np.random.randint(low=0, high=num_kernels)
                        for ch in range(images.shape[-1]):
                            images[i, ..., ch] = convolve2d(
                                images[i, ..., ch],
                                np.outer(weights[k], weights[k]),
                                mode="same",
                            )

                return images

            blur.is_parallel = True
            return blur
        else:
            # use the specified random seed with the `random` module
            def blur(images, _, counter):
                random.seed(seed + counter)
                
                should_blur = np.zeros(len(images))
                for i in range(len(images)):
                    should_blur[i] = random.uniform(0, 1) < blur_prob

                for i in my_range(images.shape[0]):
                    if should_blur[i]:
                        k = random.randint(0, num_kernels-1) # random.randint is inclusive of high value
                        for ch in range(images.shape[-1]):
                            images[i, ..., ch] = convolve2d(
                                images[i, ..., ch],
                                np.outer(weights[k], weights[k]),
                                mode="same",
                            )

                return images

            blur.is_parallel = True
            blur.with_counter = True
            return blur

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (
            replace(previous_state, jit_mode=False),
            None,
        )