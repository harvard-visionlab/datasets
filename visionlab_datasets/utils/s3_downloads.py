import os
import sys
import boto3
from tqdm.notebook import tqdm
from ..auth import get_aws_credentials
from .cache_dir import get_cache_dir
from .s3_helpers import is_object_public

def download_folder(folder_key, target_directory, bucket_name, region='us-east-1', dryrun=False):
    if not folder_key.endswith('/'):
        folder_key += '/'
    
    # Initialize S3 resource with the specified region
    s3 = boto3.resource('s3', region_name=region)
    bucket = s3.Bucket(bucket_name)
    
    # Get all files in the bucket that start with the folder key
    files = [obj.key for obj in bucket.objects.filter(Prefix=folder_key) if not obj.key.endswith('/')]
    print(f"Found {len(files)} files to download from: s3://{bucket_name}/{folder_key}")
    print(f"to: {target_directory}")
    
    # Download each file
    for file_key in tqdm(files, desc="Syncing files", file=sys.stdout, unit="file"):
        # Extract the part of the file path after the folder key
        relative_path = file_key[len(folder_key):]
        
        # Skip empty paths (e.g., in case of trailing '/')
        if not relative_path:
            continue
        
        target_path = os.path.join(target_directory, relative_path)
        temp_path = target_path + '.filepart'
        
        # Check if the file already exists and matches the size in S3
        s3_object = s3.Object(bucket_name, file_key)
        if os.path.exists(target_path) and os.path.getsize(target_path) == s3_object.content_length:
            print(f"Skipping {file_key}: already exists and matches size.")
            continue

        # Dry run: only print what would happen
        if dryrun:
            print(f"Would download {file_key} to {target_path}")
            continue

        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Download with a progress bar
        with tqdm(total=s3_object.content_length, unit='B', unit_scale=True, desc=f"Downloading {relative_path}", file=sys.stdout) as progress_bar:
            def progress_hook(bytes_amount):
                progress_bar.update(bytes_amount)

            # Download the file to the temporary path
            s3_object.download_file(temp_path, Callback=progress_hook)

        # Move the temporary file to the final path after successful download
        os.rename(temp_path, target_path)

def sync_bucket_to_local(bucket_url, cache_dir=None, profile_name=None, dryrun=False):
    # Parse bucket name and folder key
    if bucket_url.startswith("s3://"):
        bucket_url = bucket_url[5:]
    bucket_name, _, folder_key = bucket_url.partition('/')

    # Set cache directory if not provided
    if cache_dir is None:
        cache_dir = get_cache_dir()

    # Perform the folder download
    download_folder(folder_key, os.path.join(cache_dir, bucket_name, folder_key), bucket_name, dryrun=dryrun)      
    
def download_s3_file(s3_url, cache_dir=None, region='us-east-1', dryrun=False, profile_name=None):
    # Parse the bucket name and file key from the S3 URL
    bucket_name, _, file_key = s3_url.strip("s3://").partition('/')

    # Set cache directory if not provided and construct target path
    if cache_dir is None:
        cache_dir = get_cache_dir()
    target_path = os.path.join(cache_dir, bucket_name, file_key)

    # First, check if the object is public
    is_public = is_object_public(s3_url, region)

    if not is_public:
        # If private, retrieve AWS credentials using the helper
        creds = get_aws_credentials(profile_name)
        if not creds["aws_access_key_id"] or not creds["aws_secret_access_key"]:
            raise ValueError("AWS credentials are missing or incomplete.")

        # Initialize the S3 resource with credentials for private access
        s3 = boto3.resource(
            's3',
            region_name=region,
            aws_access_key_id=creds["aws_access_key_id"],
            aws_secret_access_key=creds["aws_secret_access_key"],
            aws_session_token=creds.get("aws_session_token"),
            endpoint_url=creds["endpoint_url"]
        )
    else:
        # Public access: Initialize the S3 resource without credentials
        s3 = boto3.resource('s3', region_name=region)
    
    s3_object = s3.Object(bucket_name, file_key)

    # Check if the file already exists locally and matches the size on S3
    try:
        content_length = s3_object.content_length  # Get the S3 object size
    except ClientError as e:
        print(f"Could not access object {file_key} in bucket {bucket_name}: {e}")
        return

    if os.path.exists(target_path) and os.path.getsize(target_path) == content_length:
        print(f"Skipping {s3_url}\n(already exists and matches size): {target_path}")
        return target_path

    # Dry run: only print what would happen
    if dryrun:
        print(f"Would download {s3_url}\nto {target_path}")
        return target_path

    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Define a temporary path for download to avoid incomplete files
    temp_path = target_path + '.filepart'

    # Download with a progress bar
    with tqdm(total=content_length, unit='B', unit_scale=True, desc=f"Downloading {file_key}", file=sys.stdout) as progress_bar:
        def progress_hook(bytes_amount):
            progress_bar.update(bytes_amount)

        # Download the file to the temporary path
        s3_object.download_file(temp_path, Callback=progress_hook)

    # Move the temporary file to the final path after successful download
    os.rename(temp_path, target_path)
    print(f"Downloaded {s3_url}\nto {target_path}")
    return target_path