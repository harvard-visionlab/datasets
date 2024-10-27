import os
from litdata.streaming.cache import Dir
from ..streaming_dataset import StreamingDatasetVisionlab
from ..find_datasets import 

bucket_locations_by_resolution = {
    (256, 100): "vision-datasets/magenet1k-litdata/streaming-s256-l512-jpgbytes-q100",
    (256, 95): "vision-datasets/magenet1k-litdata/streaming-s256-l512-jpgbytes-q95"
}

def imagenet1k(split, res=256, quality=100, cache_dir=None, transform=None, profile='wasabi'):
    if res is None: res = 'full'
    assert split in splits, f"Expected split to be in {splits}, got {split}"
    assert res in resolutions, f"Expected res to be one of {resolutions}, got {res}"
    
    bucket_prefix = os.path.join(bucket_locations_by_resolution[(res,quality)], split)
    dirs = get_streaming_dirs(bucket_prefix)
    
    dataset = StreamingDataset(Dir(path=local_dir, url=remote_dir),
                               storage_options=storage_options)
    
    num_images = len(dataset)
    assert num_images==num_expected[split], f"Oops, expected {num_expected[split]} images, found {num_images}. Check the files at the dataset location: {extracted_folder}"
        
    return dataset