"""
Random patch shuffle.

Divide the image into an NxN grid of patches, and randomly shuffle the patches.

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

__all__ = ['RandomPatchShuffleRatio']

class RandomPatchShuffleRatio(Operation):
    """Randomly divide an image into patches, and shuffle their locations.

    Parameters
    ----------
    shuffle_prob : float
        The probability with which to shuffle the image.
    patch_sizes: List
        List of patch sizes to choose from (in pct of resolution). 
    seed: int
        Random seed for reproducible transforms.
    """

    def __init__(self, 
                 shuffle_prob: float = 0.5, 
                 patch_sizes: List = [112/224, 56/224, 28/224, 14/224], # patch size in pct of resolution
                 seed: int = None):
        super().__init__()
        self.shuffle_prob = shuffle_prob
        self.patch_sizes = patch_sizes
        self.seed = seed

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        shuffle_prob = self.shuffle_prob
        patch_sizes = np.array(self.patch_sizes, dtype=np.float32)
        seed = self.seed

        def shuffle(images, dst, counter):
            random.seed(seed + counter if seed is not None else seed)
            
            should_shuffle = np.zeros(len(images), dtype=np.int32)
            patch_size = np.zeros(len(images), dtype=np.float32)
            for i in range(images.shape[0]):
                should_shuffle[i] = random.uniform(0, 1) <= shuffle_prob
                patch_size[i] = patch_sizes[random.randint(0, len(patch_sizes) - 1)]
                
            for i in my_range(images.shape[0]):
                if should_shuffle[i]:
                    # we're using dst[i] as a temporary buffer to hold a copy of images[i]
                    # we can modify slices of images[i,a:b,c:d] inplace (but not slices of dst[i,a:b,c:d])
                    # so we modify images inplace and return the inplace-modified `images`
                    img_array = dst[i] # using memory allocated to dst
                    img_array[:] = images[i] # placing a copy of the images[i] in there
                    
                    # Calculate the number of patches in each dimension
                    height, width, _ = img_array.shape
                    M, N = int(patch_size[i]*height), int(patch_size[i]*width)
                    num_patches_vertical = height // M
                    num_patches_horizontal = width // N
                    if height % M != 0 or width % N != 0:
                        raise ValueError("Image dimensions must be divisible by block size")
                    
                    # Generate a permutation of indices
                    num_patches = num_patches_vertical * num_patches_horizontal
                    indices = list(range(num_patches))
                    for j in range(len(indices) - 1, 0, -1):
                        k = random.randint(0, j) # inclusive of j, so possible for j==k
                        indices[j], indices[k] = indices[k], indices[j]  # Swap
                    
                    # modify images[i] inplace with the shuffled patches from img_array (copy of original images[i])
                    for row in range(num_patches_vertical):
                        for col in range(num_patches_horizontal):
                            # get src_idx for this patch
                            src_idx = row * num_patches_horizontal + col
                                
                            # get dst_idx and coordinates for this patch
                            dst_idx = indices[src_idx]
                            dst_row = (dst_idx // num_patches_horizontal)
                            dst_col = (dst_idx % num_patches_horizontal)

                            images[i,dst_row*M:(dst_row+1)*M, dst_col*N:(dst_col+1)*N] = img_array[row*M:(row+1)*M, col*N:(col+1)*N]

            return images
        shuffle.is_parallel = True
        shuffle.with_counter = True
        return shuffle


    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), 
                AllocationQuery(previous_state.shape, previous_state.dtype))
