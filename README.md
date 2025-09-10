# datasets
visionlab datasets

### setup

Make sure you have your aws credentials setup

### environment (example)
```
conda create -n dataprep python=3.12
conda activate dataprep
conda install -c conda-forge mamba
python -m pip install uv
python -m uv pip install git+https://github.com/rwightman/pytorch-image-models.git
python -m uv pip install git+https://github.com/Lightning-AI/litdata.git@main
python -m uv pip install --upgrade git+https://github.com/lilohuang/PyTurboJPEG.git
mamba install s5cmd
mamba install -c conda-forge opencv -y
mamba install ipykernel -y
python -m ipykernel install --user --name=dataprep
```

### installation

Requirements/dependencies are not automatically installed (e.g., torch, torchvision, numpy, pandas) so you can install the visionlab_datasets package without fear of mucking up your environment. Activate an environment with the following dependencies installed: ..., then install the visionlab_datasets package:
```
pip install git+https://github.com/harvard-visionlab/datasets.git
```

# Usage Examples

```
from visionlab.datasets import load_dataset, list_datasets
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



