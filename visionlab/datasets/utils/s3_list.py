import os
import sys
import boto3
from tqdm.notebook import tqdm
from .s3_auth import get_aws_credentials
from .cache_dir import get_cache_dir

def list_bucket_folders(bucket_url, profile_name=None):
    return list_bucket(bucket_url, profile_name=profile_name, Delimiter='/')

def list_bucket_files(bucket_url, profile_name=None):
    return list_bucket(bucket_url, profile_name=profile_name, Delimiter='')
    
def list_bucket(bucket_url, profile_name=None, Delimiter=None):
    bucket_name, _, prefix = bucket_url.strip("s3://").partition('/')
    
    # Retrieve AWS credentials
    creds = get_aws_credentials(profile_name)
    
    # Check if credentials were retrieved successfully
    if not creds or not creds["aws_access_key_id"] or not creds["aws_secret_access_key"]:
        raise ValueError("AWS credentials are missing or incomplete.")
    
    # Create an S3 client using the retrieved credentials
    s3_client = boto3.client(
        's3',
        aws_access_key_id=creds["aws_access_key_id"],
        aws_secret_access_key=creds["aws_secret_access_key"],
        endpoint_url=creds["endpoint_url"]
    )
    
    # List contents of the specified S3 bucket and subfolder (if provided)
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter=Delimiter)
        files = []
        print(f"{bucket_name}/")
        if Delimiter == '/':
            # List subfolders within the specified subfolder
            if "CommonPrefixes" in response:
                for common_prefix in response["CommonPrefixes"]:
                    print(f"    {common_prefix['Prefix']}")
                    files.append(common_prefix['Prefix'])
        else:
            # List files in the specified subfolder
            if "Contents" in response:
                for obj in response["Contents"]:
                    # Skip the folder itself if it appears in the list
                    if obj["Key"] != prefix:
                        print(f"    {obj['Key']}, Size: {obj['Size']} bytes")
                        files.append(obj['Key'])
        
        return files
    except s3_client.exceptions.NoSuchBucket:
        print(f"Bucket '{bucket_name}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
