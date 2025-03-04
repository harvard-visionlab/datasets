import os
import boto3
import re
import requests
from pathlib import Path
from botocore.exceptions import ClientError
from botocore import UNSIGNED
from botocore.client import Config
from urllib.parse import urlparse

from pdb import set_trace

from .s3_auth import get_aws_credentials

def is_streaming_dataset(source, profile_name=None):        
    parsed = urlparse(source)
    stem, ext = split_name(Path(parsed.path))
    return (
        'litdfata' in source
        or (parsed.scheme == "" and os.path.isfile(os.path.join(source, 'index.json')))
        or (parsed.scheme.startswith("s3") and not ext and s3_file_exists(os.path.join(source, 'index.json'), profile_name=profile_name))
    )

def extract_s3_provider(s3_path):
    # Handle extended S3 URI format (s3+provider://)
    if s3_path.startswith("s3+"):
        provider, remainder = s3_path[3:].split("://", 1)
        if provider == "wasabi":
            provider = "wasabisys"
        # Convert to standard s3:// format for parsing
        s3_path = "s3://" + remainder
    else:
        provider = "amazonaws"
    
    return s3_path, provider

def parse_extended_s3_path(s3_path_extended, region=None, endpoint_url=None):
    
    s3_path, provider = extract_s3_provider(s3_path_extended)
    
    # Set the endpoint based on provider
    if endpoint_url is None:
        endpoint_url = f"https://s3.{provider}.com" if not region else f"https://s3.{region}.{provider}.com"
    
    # Parse the S3 path to extract bucket and object key
    bucket_name, object_key = parse_s3_path(s3_path)
    
    return s3_path, provider, endpoint_url, bucket_name, object_key

def s3_file_exists(s3_path, endpoint_url=None, region=None, profile_name=None):
    
    # parse potentially extended path (s3+wasabi://) to get bucket_name and object_key
    _, _, endpoint_url, bucket_name, object_key = parse_extended_s3_path(s3_path, endpoint_url=endpoint_url, region=region)
    
    # First, check if the object is public
    is_public = is_object_public(s3_path, region)
        
    if not is_public:
        # If private, retrieve AWS credentials using the helper
        creds = get_aws_credentials(profile_name)
        s3 = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=creds["aws_access_key_id"],
            aws_secret_access_key=creds["aws_secret_access_key"],
            aws_session_token=creds.get("aws_session_token"),
            endpoint_url=creds.get("endpoint_url") if not endpoint_url else endpoint_url
        )
    else:
        # Public access: Initialize the S3 slient without credentials
        s3 = boto3.client('s3', 
                          region_name=region,
                          endpoint_url=endpoint_url,
                          config=Config(signature_version=UNSIGNED))

    try:
        # Attempt to retrieve the file metadata to check its existence
        # This will raise an exception if the object doesn't exist or access is denied
        response = s3.head_object(Bucket=bucket_name, Key=object_key) 
        return True
    except ClientError as e:
        # Check if the error is due to a missing object or lack of permissions
        if e.response['Error']['Code'] in ["404", "403"]:
            return False
        else:
            raise e  # Re-raise any other exceptions to handle separately
            
def _count_s3_objects(bucket_name, prefix, region='us-east-1', profile_name=None):
    """
    Count the number of objects in an S3 bucket with a specified prefix.

    Parameters:
    bucket_name (str): The name of the S3 bucket.
    prefix (str): The prefix (path) to search for objects.
    region (str): AWS region where the bucket is located.
    profile_name (str): Optional. AWS profile name for private S3 access if needed.

    Returns:
    int: The number of objects with the specified prefix.
    """
    # Check if the object is public
    is_public = is_object_public(f"s3://{bucket_name}/{prefix}", region)

    if not is_public:
        # If private, retrieve AWS credentials using the helper
        creds = get_aws_credentials(profile_name)
        # Initialize the S3 resource with credentials for private access
        s3 = boto3.resource(
            's3',
            region_name=region,
            aws_access_key_id=creds["aws_access_key_id"],
            aws_secret_access_key=creds["aws_secret_access_key"],
            aws_session_token=creds.get("aws_session_token"),
            endpoint_url=creds.get("endpoint_url")
        )
    else:
        # Public access: Initialize the S3 resource without credentials
        s3 = boto3.resource('s3', region_name=region)

    bucket = s3.Bucket(bucket_name)
    
    # List objects with the given prefix and count them
    return sum(1 for _ in bucket.objects.filter(Prefix=prefix))


def is_bucket_folder(s3_url, region='us-east-1', profile_name=None):
    """
    Check if an S3 URI points to a folder (i.e., contains multiple files).

    Parameters:
    s3_url (str): The S3 URI (e.g., 's3://bucket-name/path/to/object').
    region (str): AWS region where the bucket is located.
    profile_name (str): Optional. AWS profile name for private S3 access if needed.

    Returns:
    bool: True if the S3 URI points to a folder with multiple files, False otherwise.
    """
    if not s3_url.startswith("s3://"): 
        return False
    
    # Parse the bucket name and prefix from the S3 URL
    bucket_name, _, prefix = s3_url.strip("s3://").partition('/')
    
    # Count objects with the specified prefix
    object_count = _count_s3_objects(bucket_name, prefix, region, profile_name)
    
    # Check if the prefix points to multiple objects
    return object_count > 1


def is_bucket_file(s3_url, region='us-east-1', profile_name=None):
    """
    Check if an S3 URI points to a single file.

    Parameters:
    s3_url (str): The S3 URI (e.g., 's3://bucket-name/path/to/object').
    region (str): AWS region where the bucket is located.
    profile_name (str): Optional. AWS profile name for private S3 access if needed.

    Returns:
    bool: True if the S3 URI points to a single file, False if it's a folder or doesn't exist.
    """
    if not s3_url.startswith("s3://"): 
        return False
    
    # Parse the bucket name and prefix from the S3 URL
    bucket_name, _, prefix = s3_url.strip("s3://").partition('/')
    
    # Count objects with the specified prefix
    object_count = _count_s3_objects(bucket_name, prefix, region, profile_name)
    
    # Check if there is exactly one object and it matches the exact prefix
    if object_count == 1:
        # Verify if the only object matches the exact key
        s3 = boto3.resource('s3', region_name=region)
        bucket = s3.Bucket(bucket_name)
        objects = list(bucket.objects.filter(Prefix=prefix))
        return objects[0].key == prefix

    return False

def is_object_public(s3_path, endpoint_url=None, region=None, timeout=5):
    """
    Check if an S3 object is public and exists.
    
    Args:
        s3_path (str): S3 URI or URL in various formats:
            - s3://bucket-name/object-key (standard S3)
            - s3+wasabi://bucket-name/object-key (Wasabi)
            - s3+[provider]://bucket-name/object-key (custom provider)
            - https://s3.amazonaws.com/bucket-name/object-key
            - https://bucket-name.s3.amazonaws.com/object-key
        endpoint_url (str, optional): The S3 compatible endpoint URL.
            Default is None, which uses the provider specified in the URI or AWS S3.
        region (str, optional): The AWS region. Default is "us-east-1".
            
    Returns:
        bool: True if the object exists and is publicly accessible, False otherwise.
        
    Raises:
        ValueError: If the s3_path format is not recognized.
    """
    s3_path, provider, endpoint_url, bucket_name, object_key = parse_extended_s3_path(s3_path, 
                                                                                      endpoint_url=endpoint_url,
                                                                                      region=region)
    
    if not bucket_name or not object_key:
        raise ValueError(f"Invalid S3 path format: {s3_path}")
        
    # Create a URL for the HEAD request
    public_url = construct_public_url(bucket_name, object_key, endpoint_url)
    
    try:
        # Perform a HEAD request to check if the object is publicly accessible
        response = requests.head(public_url, timeout=timeout)
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"Error checking object at URL {public_url}: {e}")
        return False

def parse_s3_path(s3_path):
    """
    Parse an S3 path in various formats to extract bucket name and object key.
    
    Args:
        s3_path (str): S3 path in one of the supported formats.
        
    Returns:
        tuple: (bucket_name, object_key)
    """
    # Format: s3://bucket-name/object-key
    if s3_path.startswith("s3://"):
        parts = s3_path[5:].split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], ""
    
    # Format: https://s3.amazonaws.com/bucket-name/object-key
    # or https://s3.wasabisys.com/bucket-name/object-key
    elif s3_path.startswith(("https://s3.", "http://s3.")):
        parsed_url = urlparse(s3_path)
        path_parts = parsed_url.path.lstrip('/').split('/', 1)
        if len(path_parts) == 2:
            return path_parts[0], path_parts[1]
        return path_parts[0], ""
    
    # Format: https://bucket-name.s3.amazonaws.com/object-key
    # or https://bucket-name.s3.wasabisys.com/object-key
    elif re.match(r"https?://[^/]+\.s3\.[^/]+", s3_path):
        parsed_url = urlparse(s3_path)
        bucket_name = parsed_url.netloc.split('.s3.')[0]
        object_key = parsed_url.path.lstrip('/')
        return bucket_name, object_key
        
    return None, None

def construct_public_url(bucket_name, object_key, endpoint_url):
    """
    Construct a public URL for the S3 object.
    
    Args:
        bucket_name (str): The S3 bucket name.
        object_key (str): The S3 object key.
        endpoint_url (str): The S3 endpoint URL.
        
    Returns:
        str: The public URL for the S3 object.
    """
    # Clean endpoint_url to ensure it doesn't have trailing slash
    endpoint_url = endpoint_url.rstrip('/')
    
    # Construct the URL
    return f"{endpoint_url}/{bucket_name}/{object_key}"

def is_object_private(s3_url, region='us-east-1'):
    return not is_object_public(s3_url, region) 

def split_name(path: Path):
    """Split a path into the stem and the complete extension (all suffixes)."""
    suffixes = path.suffixes
    if suffixes:
        ext = "".join(suffixes)
        stem = path.name[:-len(ext)]
    else:
        ext = ""
        stem = path.name
    return stem, ext