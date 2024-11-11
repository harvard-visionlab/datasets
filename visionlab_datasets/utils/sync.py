import os
import requests
import subprocess
import shutil
from pathlib import Path
from urllib.parse import urlparse
from tqdm.notebook import tqdm
from pdb import set_trace

from .archive_helpers import decompress_if_needed
from .remote_data import get_file_metadata
from .cache_dir import get_cache_root, get_cache_dir
from .s3_info import s3_file_exists, is_bucket_folder, is_streaming_dataset
from .s3_downloads import sync_bucket_to_local, download_s3_file

def sync_directories(source, destination, delete=False, verbose=True):
    """
    Sync files from the source directory to the destination directory using rsync.

    Parameters:
    source (str): The source directory path.
    destination (str): The destination directory path.
    delete (bool): If True, delete files in the destination that are not in the source (like --delete option in rsync).
    verbose (bool): If True, show rsync output in detail.

    Returns:
    None
    """
    Path(destination).mkdir(parents=True, exist_ok=True)

    # Construct the rsync command with progress
    rsync_command = ["rsync", "-av", "--progress"]

    # Add the --delete flag if specified
    if delete:
        rsync_command.append("--delete")

    # Add source and destination to the command
    rsync_command.extend([source, destination])

    # Run the rsync command in a subprocess
    try:
        if verbose:
            print(f"Syncing from {source} to {destination} with rsync...")
        result = subprocess.run(rsync_command, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while syncing directories: {e}")
        print(e.stderr)

        
def download_file(url, destination_path):
    """
    Download a file from an HTTP or HTTPS URL and save it to a specified file path,
    with a progress bar to show download progress.

    Parameters:
    url (str): The URL of the file to download.
    destination_path (str): The full path, including file name, where the file will be saved.

    Returns:
    None
    """
    # Send a GET request to the URL with streaming enabled
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)

    # Get the total file size from headers (for progress display)
    file_size = int(response.headers.get('Content-Length', 0))
    chunk_size = 8192  # 8 KB chunks

    # Set up progress bar
    with open(destination_path, 'wb') as file, tqdm(
        total=file_size, unit='B', unit_scale=True, desc=destination_path, ncols=80
    ) as progress_bar:
        # Download the file in chunks and write to the destination path
        for chunk in response.iter_content(chunk_size=chunk_size):
            file.write(chunk)
            progress_bar.update(len(chunk))

            
def sync_to_local_cache(source, cache_root=None, decompress=True, progress=True, profile_name=None, region='us-east-1'):
    '''
        Sync data to the local cache directory:
            (individual file from any source): cache_root/hashid/filename
            (s3 bucket directory): cache_root/s3/<endpoint_url>/<bucket-name>/<bucket-key>
            (mnt directory): cache_root/mnt/<mountpoint>/<path>/
            
        cache_root is expected to be fast/large storage (e.g., netscratch), and will 
        default to a reasonable location (see utils.cache_dir).
        
        Data source can be s3, http(s), or a mounted directory (e.g., slow external drive).
        
        Data source can be any format. If it's a litdata Streaming Dataset,
        we skip syncing because StreamingDataset handles this automatically.
        
        Recognized archives (.zip and .tar) are automatically decompressed by default (decompress=True).
        
        Individual files are stored in os.path.join(cache_root, metadata['hash'], filename),
        where metadata is the first 32 chracters of the sha256 sum of the first 8192bytes read from a file.
        Probability of a hash collision is extremely low.
        
    '''    
    
    # litdata streaming dataset automatically handles syncing local copies
    if is_streaming_dataset(source, profile_name=profile_name):
        print("==> Source is a streaming dataset, skipping sync:")
        return source
    
    # get the cache root directory
    cache_root = get_cache_root() if cache_root is None else cache_root
    
    # handle sync for differnt locations/formats of data:
    if is_bucket_folder(source):
        # sync s3 directory
        local_path = get_cache_dir(source, cache_root=cache_root, profile_name=profile_name)
        sync_bucket_to_local(source, local_path)
        return local_path
    
    elif os.path.isdir(source):
        # sync mnt directory
        local_path = get_cache_dir(source, cache_root=cache_root, profile_name=profile_name)
        sync_directories(source, local_path)
        return local_path
    
    else:
        # `source` appears a single file:
        metadata = get_file_metadata(source) 
        assert metadata is not None and metadata['size'] > 0, f"Could not access source data at {source}"

        # get the local_path (destination cache location)
        filename = Path(metadata['path']).name
        local_path = os.path.join(cache_root, metadata['hash'], filename)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        # cache miss, sync the data:
        if not os.path.exists(local_path):   
            # sync the data
            if metadata['scheme'] == 's3':
                download_s3_file(source, local_path, profile_name=profile_name, region=region)
            elif metadata['scheme'] in ['http','https']:
                download_file(source, local_path)
            elif os.path.isfile(source):
                print(f"==> copying from mnt location: {source} to cache_path {local_path}")
                shutil.copy(source, local_path)
        else:
            print(f"==> cache hit: {local_path}")
            
        # decompress archives
        if local_path.endswith(".zip"):
            local_path = decompress_if_needed(local_path)
            
        if local_path.endswith(".tar") or local_path.endswith(".tar.gz") or local_path.endswith(".tgz"):
            local_path = decompress_if_needed(local_path)
            
        return local_path
    
    raise ValueError(f"Failed to copy data from {source} to local cache")