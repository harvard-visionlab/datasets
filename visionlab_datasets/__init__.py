from .cog import *
from .datasets import *
from .ml_probe import *
from .ml_training import *
from .neuro import *
from .utils import *
from .version import *
from .registry import dataset_registry

dataset_types = ['ml_training', 'ml_probe', 'neuro', 'cog']

def load_dataset(source, split=None, **kwargs):
    dataset_type, dataset_name = source.split("/")
    assert dataset_type in dataset_types, f"Dataset type should be in {dataset_types}, got {dataset_type}"
    
    splits = list(dataset_registry.datasets[dataset_type][dataset_name].keys())
    assert split in splits, f"Expected split to be in {splits}, got {split}"
    dataset = dataset_registry.datasets[dataset_type][dataset_name][split](**kwargs)
    
    return dataset

def list_datasets(pattern="*"):
    """List available datasets in the registry, filtered by pattern."""
    datasets = dataset_registry.list_datasets(pattern)
    if not datasets:
        print("No datasets found matching the specified pattern.")
    else:
        print("Available datasets:")
        for full_name, description in datasets:
            print(f"- {full_name}: {description}")
