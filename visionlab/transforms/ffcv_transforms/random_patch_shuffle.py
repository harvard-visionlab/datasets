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

__all__ = ['RandomPatchShuffle']

class RandomPatchShuffle(Operation):
    """Randomly divide an image into NxN patches, and shuffle their locations.

    Parameters
    ----------
    resizedshuffle_prob : float
        The probability with which to shuffle the image.
    patch_sizes: List
        List of patch sizes to choose from (in pixels). 
    seed: int
        Random seed for reproducible transforms.
    """

    def __init__(self, 
                 shuffle_prob: float = 0.5, 
                 patch_sizes: List = [112, 56, 28, 14, 7], 
                 seed: int = None):
        super().__init__()
        self.shuffle_prob = shuffle_prob
        self.patch_sizes = patch_sizes
        self.seed = seed

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        shuffle_prob = self.shuffle_prob
        patch_sizes = np.array(self.patch_sizes, dtype=np.int32)
        seed = self.seed

        def shuffle(images, dst, counter):
            random.seed(seed + counter if seed is not None else seed)
            
            should_shuffle = np.zeros(len(images), dtype=np.int32)
            patch_size = np.zeros(len(images), dtype=np.int32)
            for i in range(images.shape[0]):
                should_shuffle[i] = random.uniform(0, 1) <= shuffle_prob
                patch_size[i] = patch_sizes[random.randint(0, len(patch_sizes) - 1)]
            
            for i in my_range(images.shape[0]):
                if should_shuffle[i]:
                    img_array = images[i]
                    
                    # Calculate the number of patches in each dimension
                    M, N = patch_size[i], patch_size[i]
                    height, width, _ = img_array.shape
                    num_patches_vertical = height // M
                    num_patches_horizontal = width // N
                    if height % M != 0 or width % N != 0:
                        raise ValueError("Image dimensions must be divisible by block size")

                    # Create a list of patches
                    patches = []
                    for row in range(num_patches_vertical):
                        for col in range(num_patches_horizontal):
                            # Extract the block
                            patch = img_array[row*M:(row+1)*M, col*N:(col+1)*N, :]
                            patches.append(patch)
                    
                    # Generate a permutation of indices and then reorder the patches manually
                    indices = list(range(len(patches)))
                    for j in range(len(indices) - 1, 0, -1):
                        k = random.randint(0, j) # inclusive of j, so possible for j==k
                        indices[j], indices[k] = indices[k], indices[j]  # Swap

                    # Reorder the patches based on shuffled indices
                    shuffled_patches = [patches[idx] for idx in indices]
                    
                    # Reassemble the image from shuffled patches
                    for idx, patch in enumerate(shuffled_patches):
                        row = (idx // num_patches_horizontal) * M
                        col = (idx % num_patches_horizontal) * N
                        dst[i][row:row+M, col:col+N, :] = patch

                else:
                    dst[i] = images[i]
            return dst
        shuffle.is_parallel = True
        shuffle.with_counter = True
        return shuffle


    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), 
                AllocationQuery(previous_state.shape, previous_state.dtype))
