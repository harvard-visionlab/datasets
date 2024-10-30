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

generate_lightning_dataset --root_dir /n/holyscratch01/alvarez_lab/Lab/datasets/ecoset --split train --output_root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/ecoset-litdata -short_resize 256 --long_crop 512 --quality 100 --image_format "jpgbytes" --chunk_bytes "100MB" --num_expected 1444911

# verify streaming datasets
check_dataset --dataset_format lightning --local_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/ecoset-litdata/streaming-s256-l512-jpgbytes-q100/test --num_expected 28250

check_dataset --dataset_format lightning --local_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/ecoset-litdata/streaming-s256-l512-jpgbytes-q100/train --num_expected 1444911

# check file sizes
du -h --max-depth=1 /n/alvarez_lab_tier1/Users/alvarez/datasets/ecoset-litdata/streaming-s256-l512-jpgbytes-q100

# sync streaming datasets
sync_dataset /n/alvarez_lab_tier1/Users/alvarez/datasets/ecoset-litdata/streaming-s256-l512-jpgbytes-q100/test/ s3://visionlab-litdata/vision-datasets/ecoset-litdata/streaming-s256-l512-jpgbytes-q100/test/ --profile lit-write

# activate env with ffcv installed
conda deactivate
source activate pytorch2 

# generate ffcv datasets

# append sha256sum to files

# verify datasets

# sync datasets

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

# 20BN-Something-Something-V2

```
# download to a scratch drive
tmux new -s dataprep
source activate litdata
download_rawdata visionlab-datasets/20BN-Something-Something-V2/ /n/holyscratch01/alvarez_lab/Lab/datasets/ --profile default

# decompress archive
cd /n/holyscratch01/alvarez_lab/Lab/datasets/20BN-Something-Something-V2
tar -xvzf 20BN-Something-Something-v2-videos.tar.gz

```