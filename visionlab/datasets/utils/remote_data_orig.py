import os
import sys
import re
import errno
import warnings
import hashlib
import boto3

from torch.hub import tqdm
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from s3_filestore.auth import get_client_with_userdata
from s3_filestore.utils import parse_s3_uri

import torch
import tarfile
from torch.hub import download_url_to_file, get_dir
from pdb import set_trace

default_data_dir = os.environ.get('VISIONLAB_DATADIR', 
                                  torch.hub.get_dir().replace("/hub", "/data"))
                                  
__all__ = ['download_data_from_url', 'get_remote_data_file', 'calculate_sha256', 'get_signed_url', 'get_signed_s3_url']

# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')

def get_signed_s3_url(url):
    assert url.startswith("s3://"), f"Expected s3_url starting with s3://, got {url}"
    bucket_name, bucket_key = parse_s3_uri(url)
    url = get_signed_url(bucket_name, bucket_key)
    
    return url

def get_signed_url(bucket_name, bucket_key, expires_in_seconds=3600, profile=os.environ.get('S3_PROFILE', None)):
    s3_client = get_client_with_userdata(profile)
    signed_url = s3_client.generate_presigned_url('get_object', 
                                                  Params={'Bucket': bucket_name, 'Key': bucket_key},
                                                  ExpiresIn=expires_in_seconds,
                                                  HttpMethod='GET')
    return signed_url
    
def download_data_from_url(
    url: str,
    data_dir: Optional[str] = None,
    progress: bool = True,
    check_hash: bool = False,
    hash_prefix: Optional[str] = None,
    file_name: Optional[str] = None
) -> Dict[str, Any]:
    r"""Downloads the object at the given URL.

    If downloaded file is a .tar file or .tar.gz file, it will be automatically
    decompressed.

    If the object is already present in `data_dir`, it's deserialized and
    returned.

    The default value of ``data_dir`` is ``<hub_dir>/../data`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (str): URL of the object to download
        data_dir (str, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (str, optional): name for the downloaded file. Filename from ``url`` will be used if not set.

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')    
    
    if data_dir is None:
        data_dir = default_data_dir

    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
        

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
        
    if check_hash and hash_prefix is None:
        matches = HASH_REGEX.findall(filename) # matches is Optional[Match[str]]
        hash_prefix = matches[-1] if matches else None
        assert hash_prefix is not None, "check_hash is True, but the filename does not contain a hash_prefix. Expected <filename>-<hashid>.<ext>"
    
    if hash_prefix is not None:
        hash_dir = os.path.join(data_dir, hash_prefix)
        os.makedirs(hash_dir, exist_ok=True)
        cached_file = os.path.join(hash_dir, filename)
    else:
        cached_file = os.path.join(data_dir, filename)

    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return cached_file

def get_remote_data_file(url, cache_dir=None, progress=True, 
                         check_hash=False, hash_prefix=None, file_name=None,
                         expires_in_seconds=3600, profile=os.environ.get('S3_PROFILE', None)) -> Mapping[str, Any]:
    
    if url.startswith("s3://"):
        bucket_name, bucket_key = parse_s3_uri(url)
        print(bucket_name, bucket_key)
        url = get_signed_url(bucket_name, bucket_key)
        print(url)

    if cache_dir is None: 
        cache_dir = default_data_dir

    cached_filename = download_data_from_url(
        url = url,
        data_dir = cache_dir,
        progress = progress,
        check_hash = check_hash,
        hash_prefix = hash_prefix,
        file_name = file_name,
    )

    print(f"cached_filename: {cached_filename}")    
    extracted_folder = decompress_tarfile_if_needed(cached_filename)

    return cached_filename, extracted_folder

def get_top_level_directory(file_path):
    with tarfile.open(file_path, 'r:*') as tar:
        top_folder_name = os.path.commonprefix(tar.getnames())
    return top_folder_name

def get_top_level_directory_fast(file_path):
    with tarfile.open(file_path, 'r:*') as tar:
        for member in tar:
            if '/' in member.name:
                return member.name.split('/')[0]
    return None

def decompress_tarfile_if_needed(file_path, output_dir=None):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")

    # Determine the directory of the tar file
    dir_name = os.path.dirname(file_path) or '.'

    # Set the output directory. If none is provided, default to the tar file's location
    if output_dir is None:
        output_dir = dir_name

    # If the output directory doesn't exist, create it
    os.makedirs(output_dir, exist_ok=True)

    # Assuming the first folder inside the tar file is the root folder of its contents
    # This will be used to check if the contents have already been extracted
    top_folder_name = get_top_level_directory_fast(file_path)
    expected_extracted_folder = os.path.join(output_dir, top_folder_name)
    
    # Check if the contents have already been extracted
    if os.path.exists(expected_extracted_folder):
        print(f"Contents have already been extracted to {expected_extracted_folder}.")
    else:
        # Contents have not been extracted; proceed with extraction
        print(f"Extracting {file_path} to {output_dir}")
        with tarfile.open(file_path, 'r:*') as tar:
            tar.extractall(path=output_dir)
            print(f"File {file_path} has been decompressed to {output_dir}.")

    return expected_extracted_folder

def calculate_sha256(file_path, block_size=8192):
    """
    Calculate the SHA256 hash of a file.

    :param file_path: path to the file being read.
    :param block_size: size of each read block. A value used for memory efficiency.
                       Default is 8192 bytes.
    :return: the SHA256 hash.
    """
    sha256 = hashlib.sha256()

    # Open the file in binary mode and read it in chunks.
    with open(file_path, 'rb') as f:
        while True:
            # Read a block from the file
            data = f.read(block_size)
            if not data:
                break  # Reached end of file

            # Update the hash
            sha256.update(data)

    # Return the hexadecimal representation of the digest
    return sha256.hexdigest()

def get_remote_hash(url, progress=True):
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    try:
        sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                sha256.update(buffer)
                pbar.update(len(buffer))

        digest = sha256.hexdigest()
    finally:
        pass
    
    return digest
