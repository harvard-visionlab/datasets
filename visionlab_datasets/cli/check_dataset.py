import os, sys
import boto3
import threading
import fire
import tempfile
import shutil

from pdb import set_trace

def check_dataset(dataset_format, local_dir, num_expected, expected_version=None):
    """
    Checks that a dataset has the expected number of items.

    Args:
        dataset_format (str): type of dataset (lightning, ffcv, images)
        local_dir (str): path to dataset
        num_expected (int): number of samples expected
        
    Example:
        check_dataset --dataset_format lightning --local_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/ecoset-litdata/streaming-s256-l512-jpgbytes-q100/test --num_expected 28250
    
        check_dataset --dataset_format lightning --local_dir /n/alvarez_lab_tier1/Users/alvarez/datasets/ecoset-litdata/streaming-s256-l512-jpgbytes-q100/train --num_expected 1444911

    """
    formats = ['lightning', 'litdata']
    
    if dataset_format == "lightning" or dataset_format == "litdata":
        from ..litdata.streaming_dataset import StreamingDatasetVisionlab
        dataset = StreamingDatasetVisionlab(local_dir, expected_version=expected_version)
        print(dataset)
        assert len(dataset)==num_expected, f"Expected {num_expected} samples, got {len(dataset)}"
        print("\n==> Dataset OK\n")
    else:
        raise ValueError(f"Unrecognized dataset_format {dataset_format}, expected one of {formats}")
        
def main():
    fire.Fire(check_dataset)

if __name__ == '__main__':
    fire.Fire(check_dataset)