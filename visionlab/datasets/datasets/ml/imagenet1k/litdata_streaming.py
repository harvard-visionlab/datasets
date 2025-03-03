import os
import traceback
from pdb import set_trace
from litdata.streaming.cache import Dir
from visionlab.datasets.utils.s3_auth import get_credentials_by_profile
from visionlab.datasets.utils.find_streaming_datasets import get_streaming_dirs
from visionlab.datasets.core.streaming_dataset import StreamingDatasetVisionlab
from visionlab.datasets.registry import dataset_registry

__all__ = ['imagenet1k_val_streaming']

# exclude the bucketname, use get_streaming_dirs to find the streaming dataset root
# (which could be a mounted volume on lightning, or a remote bucket otherwise)
bucket_locations_by_resolution_and_quality = {
    (256, 100): dict(prefix="imagenet1k/streaming-s256-l512-jpgbytes-q100",
                     version=dict(val='1730021323.9057539', train="1741030773.04552"))
    # (256, 95): "imagenet1k-litdata/streaming-s256-l512-jpgbytes-q95"
}

num_expected = {
    "val": 50_000,
    "train": 1_281_167,
}

def imagenet1k(split, res=256, quality=100, transforms=None, profile_name=None, **kwargs):
    '''
        split: train or val
        res: (int) resolution of shortest edge for preprocessed dataset
        quality: (int) compression quality
        transform: image transforms
        profile_name: aws profile to use if connecting via s3
        **kwargs: extra kwargs to pass to StreamingDatasetVisionlab
    '''
    if profile_name is None:
        profile_name = os.environ.get('AWS_DEFAULT_PROFILE', 'wasabi')
        
    # validate input arguments
    if res is None: res = 256
    if quality is None: quality = 100
    splits = list(num_expected.keys())
    resolutions = list(set([res for res,_ in  bucket_locations_by_resolution_and_quality.keys()]))
    qualities = list(set([quality for _,quality in  bucket_locations_by_resolution_and_quality.keys()]))
    if res is None: res = 'full'
    assert split in splits, f"Expected split to be in {splits}, got {split}"
    assert res in resolutions, f"Expected res to be one of {resolutions}, got {res}"
    assert quality in qualities, f"Expected quality to be one of {qualities}, got {quality}"
    
    # get the local and remote directories for this dataset:
    info = bucket_locations_by_resolution_and_quality[(res,quality)]
    bucket_prefix = os.path.join(info['prefix'], split)
    dirs = get_streaming_dirs(bucket_prefix)
    expected_version = info['version'][split]
    
    if dirs['remote_dir'] is not None and dirs['remote_dir'].startswith("s3:"):
        aws_credentials = get_credentials_by_profile(profile_name)
        del aws_credentials['region']
    else:
        aws_credentials = None
                
    # create the dataset 
    try:
        dataset = StreamingDatasetVisionlab(Dir(path=dirs['local_dir'], url=dirs['remote_dir']),
                                            storage_options=aws_credentials, 
                                            transforms=transforms,
                                            expected_version=expected_version,
                                            **kwargs)
    except Exception as e:
        print(dirs)
        traceback.print_exc()  # This prints the full traceback immediately
        raise ValueError("Something went wrong") from e
    
    # assert expected number of samples
    num_images = len(dataset)
    assert num_images==num_expected[split], f"Oops, expected {num_expected[split]} images, found {num_images}. Check the files at the dataset location: {extracted_folder}"
        
    return dataset

@dataset_registry.register("ml", "imagenet1k", split="val", fmt="streaming", metadata="ImageNet1K Litdata-Streaming")
def imagenet1k_val_streaming(**kwargs):
    return imagenet1k("val", **kwargs)