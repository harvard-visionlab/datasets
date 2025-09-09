import torch
import torch.nn.functional as F
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

__all__ = ['GaussianBlurTorchImage']

class GaussianBlurTorchImage(Operation):
    """Blurs image with randomly chosen Gaussian blur.
    
    Only works for TensorImages ([..., C, H, W])            

    Args:
        blur_prob (float): probability to apply blurring to each input
        kernel_size (int or sequence, values must be odd): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.
    """

    def __init__(self, blur_prob, kernel_size=5, sigma=(0.1, 2.0), seed=None):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.blur_prob = blur_prob
        self.kernel_size = kernel_size
        assert sigma[1] > sigma[0]
        self.sigmas = np.linspace(sigma[0], sigma[1], 10)        
        self.weights = self.get_weights(kernel_size, sigma) # weights shape is: numKernels x H x W
        self.seed = seed
    
    def get_weights(self, kernel_size, sigma):
        # Create the 1D Gaussian kernel for each sigma
        weights_1d = torch.stack([
            self.get_gaussian_kernel(kernel_size, s)
            for s in np.linspace(sigma[0], sigma[1], 10)
        ])

        # Compute the 2D Gaussian kernel by taking the outer product
        weights_2d = torch.stack([torch.outer(w, w) for w in weights_1d])

        # Normalize the 2D kernels
        weights_2d /= weights_2d.view(weights_2d.size(0), -1).sum(1).unsqueeze(1).unsqueeze(2)

        return weights_2d
        
    def get_gaussian_kernel(self, kernel_size, sigma):
        # Create a vector with Gaussian values
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gaussian = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gaussian /= gaussian.sum()
        return gaussian
    
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
                self.weights = self.weights.to(device=images.device, dtype=images.dtype)
                
                for i in my_range(images.shape[0]):
                    if np.random.rand() < blur_prob:
                        k = np.random.randint(low=0, high=num_kernels)
                        for ch in range(images.shape[1]):
                            images[i, ch] = F.conv2d(
                                images[i, ch].unsqueeze(0).unsqueeze(0), 
                                self.weights[k].unsqueeze(0).unsqueeze(0), 
                                padding=kernel_size//2, # same padding for odd-sized kernels
                            ).squeeze(0).squeeze(0)

                return images

            blur.is_parallel = True
            return blur
        else:
            # use the specified random seed with the `random` module
            def blur(images, _, counter):
                self.weights = self.weights.to(device=images.device, dtype=images.dtype)
                
                random.seed(seed + counter)
                
                should_blur = np.zeros(len(images))
                for i in range(len(images)):
                    should_blur[i] = random.uniform(0, 1) < blur_prob

                for i in my_range(images.shape[0]):
                    if should_blur[i]:
                        k = random.randint(0, num_kernels-1) # random.randint is inclusive of high value
                        for ch in range(images.shape[1]):
                            images[i, ch] = F.conv2d(
                                images[i, ch].unsqueeze(0).unsqueeze(0), 
                                self.weights[k].unsqueeze(0).unsqueeze(0), 
                                padding=kernel_size//2, # same padding for odd-sized kernels
                            ).squeeze(0).squeeze(0)

                return images

            blur.is_parallel = True
            blur.with_counter = True
            return blur

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (
            replace(previous_state, jit_mode=False),
            None,
        )