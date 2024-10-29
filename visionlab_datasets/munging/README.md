# Prepare datasets

# ecoset

```
# download to a scratch drive
tmux new -s dataprep
source activate litdata
download_rawdata visionlab-datasets/ecoset/ /n/holyscratch01/alvarez_lab/Lab/datasets/ --profile default

# decompress archive
cd /n/holyscratch01/alvarez_lab/Lab/datasets/
chmod u+x unzip_ecoset.sh
./unzip_ecoset.sh

# create streaming datasets
generate_lightning_dataset --root_dir /n/holyscratch01/alvarez_lab/Lab/datasets/ecoset --split test --output_root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/ecoset-litdata -short_resize 256 --long_crop 512 --quality 100 --image_format "jpgbytes" --chunk_bytes "64MB" --num_expected 28250

generate_lightning_dataset --root_dir /n/holyscratch01/alvarez_lab/Lab/datasets/ecoset --split train --output_root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/ecoset-litdata -short_resize 256 --long_crop 512 --quality 100 --image_format "jpgbytes" --chunk_bytes "100MB" --num_expected 28250

# sync streaming datasets 

# activate env with ffcv installed
conda deactivate
source activate pytorch2 

```

# vggface2

```
# download to a scratch drive
tmux new -s dataprep
source activate litdata
download_rawdata visionlab-datasets/vggface2/ /n/holyscratch01/alvarez_lab/Lab/datasets/ --profile default

# check integrity

# decompress archive
cd /n/holyscratch01/alvarez_lab/Lab/datasets/
unzip ecoset.zip -P asdf

```