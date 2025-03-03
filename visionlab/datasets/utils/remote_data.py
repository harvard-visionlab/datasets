import os
import re
import hashlib
import boto3
import requests
from urllib.parse import urlparse
from botocore.exceptions import ClientError
from botocore import UNSIGNED
from botocore.client import Config
from pdb import set_trace

from .s3_auth import get_aws_credentials, is_object_public
from .s3_downloads import download_s3_file

def fetch(source, cache_dir=None, endpoint_url=None, region='us-east-1', dryrun=False, profile_name=None):        
    parsed = urlparse(source)
    hasher = hashlib.sha256()

    if parsed.scheme == "s3":
        filepath = download_s3_file(source, 
                                    cache_dir=cache_dir, 
                                    endpoint_url=endpoint_url,
                                    region=region, 
                                    dryrun=dryrun, 
                                    profile_name=profile_name)
    elif parsed.scheme in ["http", "https"] and parsed.netloc.startswith("s3"):
        endpoint_url, region = parse_s3_url(source)
        filepath = download_s3_file(f"s3://{parsed.path[1:]}", 
                                    cache_dir=cache_dir, 
                                    endpoint_url=endpoint_url,
                                    region=region, 
                                    dryrun=dryrun, 
                                    profile_name=profile_name)
    elif parsed.scheme in ["http", "https"]:
        set_trace()

    return filepath

def parse_s3_url(url):
    parsed = urlparse(url)
    parts = parsed.netloc.split(".")
    if len(parts) == 3:
        endpoint_url = f"https://{parsed.netloc}"
        region = 'us-east-1'
    elif len(parts) == 4:
        region = parts.pop(1)
        endpoint_url = "https://" + ".".join(parts)
    else:
        raise ValueError("Unexpected URL format: {}".format(url))
        
    return endpoint_url, region
    
def get_file_metadata(source, read_limit=8192, hash_length=32, profile_name=None,
                      endpoint_url=None, region=None):
    """
    Retrieve file metadata, including size and unique identifier (content hash) based on the source.
    
    Parameters:
    source (str): The source URI, which can be an S3 URI, HTTP/HTTPS URL, or local file path.
    read_limit (int): The number of bytes to read for generating the hash.
    profile_name (str): AWS profile name to use for private S3 access if needed.
    region (str): AWS region for the S3 bucket.
    
    Returns:
    dict: A dictionary containing 'size' (in bytes) and 'hash' (SHA-256 hash of content sample).
    """
    parsed = urlparse(source)
    hasher = hashlib.sha256()
    size = None

    if parsed.scheme == "s3":
        # For S3 URIs
        bucket_name = parsed.netloc
        key = parsed.path.lstrip('/')

        # Check if the S3 object is public
        creds = get_aws_credentials(profile_name)
        if endpoint_url is None:
            endpoint_url = creds["endpoint_url"]
        if region is None:
            region = creds["region"]
        is_public = is_object_public(source, endpoint_url=endpoint_url, region=region)

        if not is_public:
            # Initialize the S3 client with credentials for private access
            s3 = boto3.client(
                's3',
                region_name=region,
                aws_access_key_id=creds["aws_access_key_id"],
                aws_secret_access_key=creds["aws_secret_access_key"],
                aws_session_token=creds.get("aws_session_token"),
                endpoint_url=creds.get("endpoint_url")
            )
        else:
            # Public access: Initialize the S3 client without credentials
            s3 = boto3.client('s3', 
                              region_name=region,
                              endpoint_url=endpoint_url,
                              config=Config(signature_version=UNSIGNED))

        # Get object metadata and partial content
        try:
            response = s3.head_object(Bucket=bucket_name, Key=key)
            size = response['ContentLength']
            # Fetch a limited range of bytes to compute a consistent hash
            range_response = s3.get_object(Bucket=bucket_name, Key=key, Range=f'bytes=0-{read_limit-1}')
            hasher.update(range_response['Body'].read())
        except ClientError as e:
            print(f"Could not access object {key} in bucket {bucket_name}: {e}")
            return

    elif parsed.scheme in ["http", "https"]:
        # For HTTP/HTTPS URLs
        head_response = requests.head(source)
        size = head_response.headers.get('Content-Length')

        # If Content-Length is not provided, perform a full GET request to get the file size
        if size is None:
            # Fetch a limited range of bytes to compute the hash
            range_response = requests.get(source, headers={'Range': f'bytes=0-{read_limit-1}'})
            hasher.update(range_response.content)
            
            # Make a second request without range to get the full size
            full_response = requests.get(source, stream=True)
            size = int(full_response.headers.get('Content-Length', 0))
            full_response.close()
        else:
            # If Content-Length is available, convert it to an integer
            size = int(size)
            
            # Fetch limited data for the hash
            range_response = requests.get(source, headers={'Range': f'bytes=0-{read_limit-1}'})
            hasher.update(range_response.content)

    elif os.path.isfile(source):
        # For local files
        size = os.path.getsize(source)
        
        # Open the file and read a limited range of bytes to compute the hash
        with open(source, 'rb') as f:
            hasher.update(f.read(read_limit))
    
    else:
        raise ValueError("Unsupported source type. Must be S3 URI, URL, or local file path.")

    unique_id = hasher.hexdigest()
    final_hash = unique_id[:hash_length] if hash_length else unique_id
    
    return {'scheme': parsed.scheme, 'netloc': parsed.netloc, 'path': parsed.path, 'size': size, 'hash': final_hash if size > 0 else None, 'read_limit': read_limit}