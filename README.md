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

Requirements/dependencies are not automatically installed, so you can install visionlab.datasets without fear of blasting / destroying your current python environment. See previous section for the minimal environment needed to use visionlab.datasets
```
python -m uv pip install git+https://github.com/harvard-visionlab/datasets.git
```

# Usage Examples

```
from visionlab.datasets import StreamingDataset, load_dataset, list_streaming_datasets
```

## list_streaming_datasets
