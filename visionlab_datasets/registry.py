import fnmatch
from pdb import set_trace

class DatasetRegistry:
    def __init__(self):
        self.datasets = {}
        self.metadata = {}
        
    def register(self, category, name, split, fmt=None, metadata=""):
        """Register a dataset with a given name and category as a decorator or callable."""
        def decorator(cls):
            # Initialize nested dictionaries in both datasets and metadata
            if category not in self.datasets:
                self.datasets[category] = {}
                self.metadata[category] = {}
            if name not in self.datasets[category]:
                self.datasets[category][name] = {}
            if name not in self.metadata[category]:
                self.metadata[category][name] = {}
            
            # Store the class reference and description
            self.datasets[category][name][split] = {}
            self.metadata[category][name][split] = {}
            self.datasets[category][name][split][fmt] = cls
            self.metadata[category][name][split][fmt] = metadata
                
            return cls  # Return the class unmodified
        return decorator
    
    def list_datasets(self, pattern="*"):
        """List datasets matching a specified pattern."""
        matched_datasets = []
        
        # Flatten the datasets for pattern matching
        for category, datasets in self.datasets.items():
            for name, splits in datasets.items():
                for split, formats in splits.items():
                    for fmt, metadata in formats.items():
                        # Create a fully qualified name like 'ml/imagenet/train' or 'neuro/konkle_72_objects/GRADIENT'
                        full_name = f"{category}/{name}/{split}/{fmt}"
                        if fnmatch.fnmatch(full_name, pattern):
                            matched_datasets.append((f"{category}/{name}", split, fmt, metadata))
        
        return matched_datasets
    
    def _generate_dataset_overview(self, pattern="*"):
        """Generate a hierarchical string representation of the datasets matching a pattern."""
        lines = ["Available datasets:"]
        
        # Iterate over datasets and filter by pattern
        for category, datasets in self.datasets.items():
            category_added = False
            for name, splits in datasets.items():
                name_added = False
                for split, description in splits.items():
                    # Create the full name for pattern matching
                    full_name = f"{category}/{name}/{split}"
                    if fnmatch.fnmatch(full_name, pattern):
                        if not category_added:
                            lines.append(f"- {category}:")
                            category_added = True
                        if not name_added:
                            lines.append(f"    - {name}:")
                            name_added = True
                        lines.append(f"        - {split}: {description}")
        
        if len(lines) == 1:
            lines.append(f"  No datasets found matching the specified pattern {pattern}.")
            
        return "\n".join(lines)
    
    def pretty_print(self, pattern="*"):
        """Print the dataset registry in a readable, hierarchical format, filtered by pattern."""
        print(self._generate_dataset_overview(pattern))
    
    def __repr__(self):
        return self._generate_dataset_overview()
    
dataset_registry = DatasetRegistry()    