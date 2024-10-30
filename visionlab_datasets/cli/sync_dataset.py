import os, sys
import boto3
import threading
import fire
import tempfile
import shutil
from tqdm import tqdm
from urllib.parse import urlparse, urljoin

from pdb import set_trace
            
def sync_dataset(local_directory, bucket_location, profile='wasabi'):
    """
    Downloads the contents of an S3 bucket folder to a local directory.

    Args:
        bucket_location (str): The S3 bucket location, e.g., 'visionlab-datasets/ecoset'.
        local_directory (str): The local directory where the contents will be downloaded.
        profile (str): The S3 profile to use. Default is 'wasabi'.
        ignore_nesting_warning (bool): Flag to ignore nesting warning if destination path seems incorrect.
        
    Example:
        sync_dataset /n/alvarez_lab_tier1/Users/alvarez/datasets/ecoset-litdata/streaming-s256-l512-jpgbytes-q100/ s3://visionlab-litdata/vision-datasets/ecoset-litdata/streaming-s256-l512-jpgbytes-q100/ --profile lit-write
        
        sync_dataset /n/alvarez_lab_tier1/Users/alvarez/datasets/ecoset-litdata/streaming-s256-l512-jpgbytes-q100/ s3://visionlab-litdata/vision-datasets/ecoset-litdata/streaming-s256-l512-jpgbytes-q100/ --profile wasabi-admin

    """
    # Set up the session
    print(f"Starting s3 session with profile_name={profile}")
    # Set up the session
    session = boto3.Session(profile_name=profile)
    # Use endpoint_url if specified in the profile
    config = session._session.full_config.get('profiles', {}).get(profile, {})
    endpoint_url = config.get('s3', {}).get('endpoint_url', None)
    print(f"endpoint_url: {endpoint_url}")
    s3 = session.resource('s3', endpoint_url=endpoint_url) if endpoint_url else session.resource('s3')

    # Parse the URI
    parsed_uri = urlparse(bucket_location)

    # Extract bucket_name and prefix
    bucket_name = parsed_uri.netloc
    prefix = parsed_uri.path.lstrip('/')
    print(f"bucket_name: {bucket_name}")
    print(f"bucket_prefix: {prefix}")
    bucket = s3.Bucket(bucket_name)
    
    # Function to check if the file exists with the same size in S3
    def file_exists_with_same_size(s3_bucket, s3_key, local_file):
        try:
            s3_object = s3_bucket.Object(s3_key)
            s3_object.load()
            return s3_object.content_length == os.path.getsize(local_file)
        except s3.meta.client.exceptions.ClientError:
            # If the object does not exist, we get a 404, so we return False
            return False
        
    # Walk through the local directory and upload each file
    for root, _, files in os.walk(local_directory):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_key = urljoin(prefix, relative_path)

            # Check for existing file with the same size in S3 to skip if needed
            if file_exists_with_same_size(bucket, s3_key, local_path):
                print(f"Skipping {local_path} (same size in S3)")
                continue
            
            # Get the file size
            file_size = os.path.getsize(local_path)

            # Set up tqdm progress bar
            progress_bar = tqdm(total=file_size, unit='MB', unit_scale=True, desc="Uploading")

            # Define a callback to update tqdm progress
            def upload_progress(chunk):
                progress_bar.update(chunk)
        
            # Upload with progress tracking
            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
            bucket.upload_file(
                local_path,
                s3_key,
                Callback=upload_progress
            )
            
            # Close the progress bar once done
            progress_bar.close()
    
def main():
    fire.Fire(sync_dataset)

if __name__ == '__main__':
    fire.Fire(sync_dataset)