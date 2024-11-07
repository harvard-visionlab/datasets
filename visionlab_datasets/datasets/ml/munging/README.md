# Prepare datasets

# imagenet
```
tmux new -s dataprep

# download rawdata to a scratch drive
source activate datasets
download_rawdata visionlab-datasets/imagenet1k-raw/ /n/holyscratch01/alvarez_lab/Lab/datasets/ --profile awsread

# create streaming datasets
generate_lightning_dataset --root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/rawdata/imagenet1k --split val --output_root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata -short_resize 256 --long_crop 512 --quality 95 --image_format "jpgbytes" --chunk_bytes "64MB" --num_expected 50000

generate_lightning_dataset --root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/rawdata/imagenet1k --split train --output_root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata -short_resize 256 --long_crop 512 --quality 95 --image_format "jpgbytes" --chunk_bytes "128MB" --num_expected 1281167

generate_lightning_dataset --root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/rawdata/imagenet1k --split val --output_root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata -short_resize 256 --long_crop 512 --quality 75 --image_format "jpgbytes" --chunk_bytes "64MB" --num_expected 50000

generate_lightning_dataset --root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/rawdata/imagenet1k --split train --output_root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata -short_resize 256 --long_crop 512 --quality 75 --image_format "jpgbytes" --chunk_bytes "128MB" --num_expected 1281167

# verify streaming datasets
check_dataset --dataset_format litdata --local_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata/streaming-s256-l512-jpgbytes-q100/val --num_expected 50000

check_dataset --dataset_format litdata --local_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata/streaming-s256-l512-jpgbytes-q100/train --num_expected 1281167

check_dataset --dataset_format litdata --local_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata/streaming-s256-l512-jpgbytes-q95/val --num_expected 50000

check_dataset --dataset_format litdata --local_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata/streaming-s256-l512-jpgbytes-q95/train --num_expected 1281167

check_dataset --dataset_format litdata --local_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata/streaming-s256-l512-jpgbytes-q75/val --num_expected 50000

check_dataset --dataset_format litdata --local_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata/streaming-s256-l512-jpgbytes-q75/train --num_expected 1281167

# sync dataset
sync_dataset /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata/streaming-s256-l512-jpgbytes-q100/ s3://visionlab-litdata/imagenet1k/streaming-s256-l512-jpgbytes-q100/ --profile lit-write

sync_dataset /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata/streaming-s256-l512-jpgbytes-q95/ s3://visionlab-litdata/imagenet1k/streaming-s256-l512-jpgbytes-q95/ --profile lit-write

sync_dataset /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/imagenet1k-litdata/streaming-s256-l512-jpgbytes-q75/ s3://visionlab-litdata/imagenet1k/streaming-s256-l512-jpgbytes-q75/ --profile lit-write

# generate ffcv datasets
generate_ffcv_dataset --root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/rawdata/imagenet1k --split val --write_path /n/alvarez_lab_tier1/Users/alvarez/datasets/ffcv/imagenet1k/imagenet1k-s256-l512-jpg-q100-val.ffcv -short_resize 256 --long_crop 512 --quality 100 --write_mode "jpg" --chunk_size 100 --num_expected 50000

generate_ffcv_dataset --root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/rawdata/imagenet1k --split train --write_path /n/alvarez_lab_tier1/Users/alvarez/datasets/ffcv/imagenet1k/imagenet1k-s256-l512-jpg-q100-train.ffcv -short_resize 256 --long_crop 512 --quality 100 --write_mode "jpg" --chunk_size 100 --num_expected 1281167

```

# ecoset

```
conda activate datasets

# download to a scratch drive
tmux new -s dataprep
source activate litdata
download_rawdata visionlab-datasets/ecoset/ /n/holyscratch01/alvarez_lab/Lab/datasets/ --profile default

# decompress archive
cd /n/holyscratch01/alvarez_lab/Lab/datasets/
chmod u+x unzip_ecoset.sh
./unzip_ecoset.sh

# create streaming datasets
generate_lightning_dataset --root_dir /n/holyscratch01/alvarez_lab/Lab/datasets/ecoset --split test --output_root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/ecoset-litdata -short_resize 256 --long_crop 512 --quality 100 --image_format "jpgbytes" --chunk_bytes "64MB" --num_expected 28250

generate_lightning_dataset --root_dir /n/holyscratch01/alvarez_lab/Lab/datasets/ecoset --split train --output_root_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/ecoset-litdata -short_resize 256 --long_crop 512 --quality 100 --image_format "jpgbytes" --chunk_bytes "100MB" --num_expected 1444911

# verify streaming datasets
check_dataset --dataset_format lightning --local_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/ecoset-litdata/streaming-s256-l512-jpgbytes-q100/test --num_expected 28250

check_dataset --dataset_format lightning --local_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/ecoset-litdata/streaming-s256-l512-jpgbytes-q100/train --num_expected 1444911

# check file sizes
du -h --max-depth=1 /n/alvarez_lab_tier1/Users/alvarez/datasets/ecoset-litdata/streaming-s256-l512-jpgbytes-q100

# sync streaming datasets
sync_dataset /n/alvarez_lab_tier1/Users/alvarez/datasets/litdata/ecoset-litdata/streaming-s256-l512-jpgbytes-q100/ s3://visionlab-litdata/ecoset/streaming-s256-l512-jpgbytes-q100/ --profile lit-write

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