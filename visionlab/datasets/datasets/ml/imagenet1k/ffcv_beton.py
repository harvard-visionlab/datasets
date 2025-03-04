import os
import traceback
import contextlib
import scipy.io as sio
from botocore import UNSIGNED
from botocore.client import Config

from visionlab.datasets.registry import dataset_registry
from visionlab.datasets.core import RemoteDataset

from pdb import set_trace

s3_urls = {
    (256, 100, 'val'): "s3+wasabi://visionlab-datasets/ffcv/imagenet1k/imagenet1k-s256-l512-jpgbytes-q100-val-a3b9af78.ffcv",
    (256, 100, 'train'): "s3+wasabi://visionlab-datasets/ffcv/imagenet1k/imagenet1k-s256-l512-jpgbytes-q100-train-122ef3d6.ffcv",
}

num_expected = dict(train=1_281_167, val=50_000)

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
    
    resolutions = list(set([res for res,_,_ in  s3_urls.keys()]))
    qualities = list(set([quality for _,quality,_ in  s3_urls.keys()]))
    splits = list(set([split for _,_,split in  s3_urls.keys()]))
    
    assert split in splits, f"Expected split to be in {splits}, got {split}"
    assert res in resolutions, f"Expected res to be one of {resolutions}, got {res}"
    assert quality in qualities, f"Expected quality to be one of {qualities}, got {quality}"
    
    source = s3_urls[(res, quality, split)]
    
    # create the dataset 
    try:
        dataset = RemoteDataset(source, profile_name=profile_name)
    except Exception as e:
        print(source)
        traceback.print_exc()  # This prints the full traceback immediately
        raise ValueError("Something went wrong") from e
    
    # assert expected number of samples
    num_images = len(dataset)
    assert num_images==num_expected[split], f"Oops, expected {num_expected[split]} images, found {num_images}. Check the files at the dataset location: {extracted_folder}"
        
    return dataset

@dataset_registry.register("ml", "imagenet1k", split="val", fmt="ffcv", metadata="ImageNet1K FFCV-Beton Dataset")
def imagenet1k_val_ffcv(**kwargs):
    return imagenet1k("val", **kwargs)