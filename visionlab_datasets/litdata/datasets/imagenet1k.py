import os
from pdb import set_trace
from litdata.streaming.cache import Dir
from ...auth import get_credentials_by_profile
from ..streaming_dataset import StreamingDatasetVisionlab
from ..find_datasets import get_streaming_dirs

__all__ = ['imagenet1k']

bucket_locations_by_resolution = {
    (256, 100): "vision-datasets/imagenet1k-litdata/streaming-s256-l512-jpgbytes-q100",
    (256, 95): "vision-datasets/imagenet1k-litdata/streaming-s256-l512-jpgbytes-q95"
}

num_expected = {
    "val": 50_000,
    "train": 1_281_167,
}

def imagenet1k(split, res=256, quality=100, transform=None, profile='wasabi', **kwargs):
    '''
        split: train or val
        res: (int) resolution of shortest edge for preprocessed dataset
        quality: (int) compression quality
        transform: image transforms
        profile: aws profile to use if connecting via s3
        **kwargs: extra kwargs to pass to StreamingDatasetVisionlab
    '''
    # validate input arguments
    if res is None: res = 256
    if quality is None: quality = 100
    splits = list(num_expected.keys())
    resolutions = list(set([res for res,_ in  bucket_locations_by_resolution.keys()]))
    qualities = list(set([quality for _,quality in  bucket_locations_by_resolution.keys()]))
    if res is None: res = 'full'
    assert split in splits, f"Expected split to be in {splits}, got {split}"
    assert res in resolutions, f"Expected res to be one of {resolutions}, got {res}"
    assert quality in qualities, f"Expected quality to be one of {qualities}, got {quality}"
    
    # get the local and remote directories for this dataset:
    bucket_prefix = os.path.join(bucket_locations_by_resolution[(res,quality)], split)
    dirs = get_streaming_dirs(bucket_prefix)

    if dirs['remote_dir'] is not None and dirs['remote_dir'].startswith("s3:"):
        aws_credentials = get_credentials_by_profile(profile)
    else:
        aws_credentials = None

    dataset = StreamingDatasetVisionlab(Dir(path=dirs['local_dir'], url=dirs['remote_dir']),
                                        storage_options=aws_credentials, **kwargs)
    
    num_images = len(dataset)
    assert num_images==num_expected[split], f"Oops, expected {num_expected[split]} images, found {num_images}. Check the files at the dataset location: {extracted_folder}"
        
    return dataset