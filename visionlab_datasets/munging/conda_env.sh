# create a "datasets" environment for all of the required data munging
# the environment should work for both litdata and ffcv datasets
module load cuda/12.2.0-fasrc01
module load gcc/9.5.0-fasrc01

# base conda installation
# ffcv has an opencv 4.6 requirement, so install a python version that works with that (3.10.13):
conda create -n datasets python=3.10 opencv=4.6

source activate datasets

conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -y -c conda-forge libjpeg-turbo

# install litdata, timm, some misc
python -m pip install 'litdata[extras]'
python -m pip install git+https://github.com/rwightman/pytorch-image-models.git
python -m pip install fastprogress seaborn fire
python -m pip install --upgrade git+https://github.com/lilohuang/PyTurboJPEG.git#egg=PyTurboJPEG

# ffcv-ssl
# mamba install -y pkg-config compilers
# mamba install -y opencv=4.6
# python -m pip install --upgrade --force-reinstall opencv-python
# python -m pip install --no-deps git+https://github.com/facebookresearch/FFCV-SSL.git#egg=FFCV-SSL

# install ffcv (with dependencies, which will break things, which we'll fix, then reinstall...trust me)
mamba install -y pkg-config compilers
# conda install -y opencv=4.6
# pkg-config --modversion opencv4
python -m pip install --force-reinstall git+https://github.com/facebookresearch/FFCV-SSL.git#egg=FFCV-SSL

# fix things ffcv install broke, reinstall without deps
python -m pip install --force-reinstall 'numpy<2' 'numpy>=1.21'
python -m pip uninstall -y opencv-python-headless numba
conda install -y opencv=4.6
python -m pip install opencv-python-headless numba
pip install --force-reinstall --no-deps git+https://github.com/facebookresearch/FFCV-SSL.git#egg=FFCV-SSL

# image handling
conda install -y conda-forge::libpng conda-forge::av -y
python -m pip install imgstore

module load cuda/12.2.0-fasrc01

# nvImageCodec Python for CUDA 12.x
python -m pip install nvidia-nvimgcodec-cu12

# Install nvJPEG for CUDA 12.x
python -m pip install nvidia-nvjpeg-cu12

# optional
python -m pip install nvidia-nvjpeg2k-cu12

# for jxl
mamba install -y conda-forge::libbrotlicommon libjpeg-turbo anaconda::libpng conda-forge::libavif conda-forge::libjxl-tools
python -m pip install jxlpy

# install kernel
python -m pip install ipykernel
python -m ipykernel install --user --name=datasets
