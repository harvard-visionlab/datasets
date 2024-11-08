from typing import List
from collections import defaultdict
from copy import deepcopy
import numpy as np
import random
from torch.utils.data import Dataset
from tqdm import tqdm

def BalancedSplits(
    dataset: Dataset,
    splits: List[float],
    seed: int = 42,
    label_field: str = 'label',
    label_index: int = 1, 
) -> List[Dataset]:
    """
    Create balanced splits from a dataset, ensuring each split contains an equal number of images per class.

    Args:
        dataset (Dataset): The dataset to split.
        splits (List[float]): List of split proportions (e.g., [0.8, 0.2]).
        seed (int): Random seed for reproducibility.

    Returns:
        List[Dataset]: List of Dataset instances, each with its own samples for the split.
    """
    if sum(splits) > 1:
        raise ValueError("Splits' sum must be less than or equal to 1.")

    # Set the random seed
    random.seed(seed)
    np.random.seed(seed)

    # Group samples by class
    class_indices = defaultdict(list)
    for idx,item in enumerate(dataset):
        if isinstance(item, dict):
            assert label_field in item, f"Expected {label_field} to be in item. Change label field to match an item key: {list(item.keys())}"
            label = item[label_field]
        else:
            assert len(item) >= label_index, f"Item has {len(item)} values, make sure label_index is set to the correct position"
            label = item[label_index]
        class_indices[label].append(idx)

    # Find the minimum class count to balance the dataset
    min_class_count = min(len(indices) for indices in class_indices.values())

    # Limit each class to min_class_count samples and shuffle them
    for label in class_indices:
        np.random.shuffle(class_indices[label])
        class_indices[label] = class_indices[label][:min_class_count]

    # Calculate the number of samples per class for each split
    split_sizes = [int(min_class_count * split) for split in splits]

    # Generate balanced indices for each split
    split_indices = [[] for _ in splits]
    for label, indices in class_indices.items():
        start = 0
        for i, split_size in enumerate(split_sizes):
            split_indices[i].extend(indices[start:start + split_size])
            start += split_size

    # Create deep copies of the dataset for each split and set their samples
    split_datasets = []
    for indices in split_indices:
        split_dataset = deepcopy(dataset)
        split_dataset.samples = [dataset.samples[idx] for idx in indices]
        split_dataset.targets = [label for _, label in split_dataset.samples]  # Update targets for consistency
        split_datasets.append(split_dataset)

    return split_datasets
