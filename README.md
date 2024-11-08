# datasets
datasets for cog-neuro-ai research

Includes interfaces to machine-learning "datasets" (stimuli like ImageNet, Ecoset, Places365, VGGFAce2, etc.), neural datasets (like Konkle72Objects, etc.), and behavioral datasets (like asdf...).

### setup

For visionlab users, make sure you have setup your "cluster/laptop/devbox/colab" environments with the necessary aws and notion tokens, and environment variables [see here]()

### installation

Requirements/dependencies are not automatically installed (e.g., torch, torchvision, numpy, pandas) so you can install the visionlab_datasets package without fear of mucking up your environment. Activate an environment with the following dependencies installed: ..., then install the visionlab_datasets package:
```
pip install git+https://github.com/harvard-visionlab/datasets.git
```

# Usage Examples

```
from visionlab_datasets import load_dataset, list_datasets
```

## list datasets
You can look here for ml datasets, or here for neuroai datasets, or list them with list_datasets

**list all datasets**:
```
list_datasets()
```

**list all ml datasets**:
```
list_datasets('ml/*')
```

**list all neuro datasets**:
```
list_datasets('neuro/*')
```

**list all cognitive (behavioral) datasets**:
```
list_datasets('cog/*')
```

**list all imagenet datasets**:
```
list_datasets('*imagenet*')
```

## load imagenet
```
# ml datasets come in multiple formats; for training "ffcv" is recommended, 
# for analysis streaming datasets are recommended
dataset = load_dataset('ml/imagenet1k', "val", fmt='streaming', res=256) # short edge resized to 256
```

## load konkle72objects stimuli
```
dataset = load_dataset('neuroai/konkle72objects', "stimuli")
```

## load konkle72objects neural data
```
dataset = load_dataset('neuroai/konkle72objects', "sectors")
```



