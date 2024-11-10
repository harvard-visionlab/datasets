import os
import boto3
from botocore.exceptions import ClientError
from urllib.parse import urlparse

from .s3_auth import get_aws_credentials, is_object_public

def is_streaming_dataset(source, profile_name=None):      
    parsed = urlparse(source)
    return (
        'litdfata' in source
        or (parsed.scheme == "" and os.path.isfile(os.path.join(source, 'index.json')))
        or (parsed.scheme == "s3"  and s3_file_exists(os.path.join(source, 'index.json'), profile_name=profile_name))
    )

def s3_file_exists(s3_url, region='us-east-1', profile_name=None):
    # Parse the bucket name and file key from the S3 URL
    bucket_name, _, file_key = s3_url.strip("s3://").partition('/')

    # First, check if the object is public
    is_public = is_object_public(s3_url, region)

    if not is_public:
        # If private, retrieve AWS credentials using the helper
        creds = get_aws_credentials(profile_name)
        #if not creds["aws_access_key_id"] or not creds["aws_secret_access_key"]:
        #    raise ValueError("AWS credentials are missing or incomplete.")

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

    # Attempt to retrieve the file metadata to check its existence
    try:
        s3_object.load()  # This will raise an exception if the object doesn't exist or access is denied
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