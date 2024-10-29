import os, sys
import boto3
import threading
import fire
import tempfile
import shutil

from pdb import set_trace

def format_size(size_in_bytes):
    """Format bytes into a human-readable string (KB, MB, GB, etc.)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} EB"

class ProgressPercentage(object):
    def __init__(self, bucket, key):
        self._bucket = bucket
        self._key = key
        self._size = self._bucket.Object(self._key).content_length
        self._seen_so_far = 0
        self._lock = threading.Lock()
        print(f"Total size to download: {format_size(self._size)}")
        
    def __call__(self, bytes_amount):
        # Thread-safe progress update
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            seen_so_far_formatted = format_size(self._seen_so_far)
            total_size_formatted = format_size(self._size)
            sys.stdout.write(
                f"\rDownloading {self._key}: {seen_so_far_formatted} / {total_size_formatted} ({percentage:.2f}%)",
            )
            sys.stdout.flush()
            
def download_s3_folder(bucket_location, local_directory, profile='wasabi',
                       ignore_nesting_warning=False):
    """
    Downloads the contents of an S3 bucket folder to a local directory.

    Args:
        bucket_location (str): The S3 bucket location, e.g., 'visionlab-datasets/ecoset'.
        local_directory (str): The local directory where the contents will be downloaded.
        profile (str): The S3 profile to use. Default is 'wasabi'.
        ignore_nesting_warning (bool): Flag to ignore nesting warning if destination path seems incorrect.
        
    Example:
        download_rawdata visionlab-datasets/testing1234/ /n/holyscratch01/alvarez_lab/Lab/datasets/ --profile wasabi
        download_rawdata visionlab-datasets/ecoset/ /n/holyscratch01/alvarez_lab/Lab/datasets/ --profile default

    """
    # Set up the session
    print(f"Starting session with profile_name={profile}")
    # Set up the session
    session = boto3.Session(profile_name=profile)
    # Use endpoint_url if specified in the profile
    config = session._session.full_config.get('profiles', {}).get(profile, {})
    endpoint_url = config.get('s3', {}).get('endpoint_url', None)
    print(f"endpoint_url: {endpoint_url}")
    s3 = session.resource('s3', endpoint_url=endpoint_url) if endpoint_url else session.resource('s3')

    # Split bucket name and prefix
    bucket_name, prefix = bucket_location.split('/', 1)
    print(f"bucket_name: {bucket_name}")
    print(f"bucket_prefix: {prefix}")
    bucket = s3.Bucket(bucket_name)
    
    # Check for potential nesting issue
    if not ignore_nesting_warning:
        nested_path = os.path.join(local_directory, prefix)
        if local_directory.rstrip('/').endswith(prefix.rstrip('/')):
            suggested_path = local_directory.rstrip('/').replace(prefix.rstrip('/'), "")
            print(f"\n==> Warning: The specified local directory appears to cause nesting: {nested_path} "
                  f"\nTo avoid this, set the local directory to a root-directory like: {suggested_path} "
                  f"\nUse --ignore_nesting_warning to proceed anyway.\n")
            return
        
    # Download all objects with the specified prefix
    for obj in bucket.objects.filter(Prefix=prefix):
        local_path = os.path.join(local_directory, obj.key)

        if obj.key.endswith('/'):
            # If the object is a directory, create the directory locally
            if not os.path.exists(local_path):
                os.makedirs(local_path)
        else:
            # If the object is a file, download it only if the size has changed
            if not os.path.exists(local_path) or os.path.getsize(local_path) != obj.size:
                if not os.path.exists(os.path.dirname(local_path)):
                    os.makedirs(os.path.dirname(local_path))

                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    bucket.download_file(obj.key, temp_file.name, Callback=ProgressPercentage(bucket, obj.key))
                    shutil.move(temp_file.name, local_path)

                print(f"\nDownloaded: {obj.key} to {local_path}")
            else:
                print(f"Skipped (no size change): {os.path.join(local_path)}")

def main():
    fire.Fire(download_s3_folder)

if __name__ == '__main__':
    fire.Fire(download_s3_folder)