import os
import re
import boto3
import requests
import warnings
from botocore.exceptions import ProfileNotFound
from botocore.configloader import load_config
import requests
from urllib.parse import urlparse

from pdb import set_trace

def get_storage_options(profile=None):
    creds = get_aws_credentials(profile_name=profile)
    return {
        "aws_access_key_id": creds['aws_access_key_id'],
        "aws_secret_access_key": creds['aws_secret_access_key'],
        "endpoint_url": creds['endpoint_url'],
    }
    
def get_aws_credentials(profile_name=None):
    if profile_name is not None:
        creds = get_credentials_by_profile(profile_name)
        if creds is not None:
            return creds
            
    return {
        "aws_access_key_id": os.getenv('AWS_ACCESS_KEY_ID'),
        "aws_secret_access_key": os.getenv('AWS_SECRET_ACCESS_KEY'),
        "endpoint_url": os.getenv('AWS_ENDPOINT_URL', 'https://s3.amazonaws.com'),
        "region": os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
    }

def get_credentials_by_profile(profile_name):
    try:
        # Create a boto3 session with the specified profile
        session = boto3.Session(profile_name=profile_name)
        credentials = session.get_credentials()
        
        # Load AWS config to access profile-specific settings
        aws_config = load_config('~/.aws/config')
        
        # Retrieve the profile section from the config
        profile_config = aws_config.get("profiles", {}).get(profile_name, {})
        
        # Check for endpoint_url in 's3' or 's3api' service configuration
        endpoint_url = profile_config.get("s3", {}).get("endpoint_url") or profile_config.get("s3api", {}).get("endpoint_url")

        # Check for region in 's3' or 's3api' service configuration
        region = profile_config.get("s3", {}).get("region") or profile_config.get("s3api", {}).get("region")
        
        # Return credentials and endpoint URL
        return {
            "aws_access_key_id": credentials.access_key,
            "aws_secret_access_key": credentials.secret_key,
            "endpoint_url": endpoint_url or 'https://s3.amazonaws.com',
            "region": region or "us-east-1"
        }

    except ProfileNotFound:
        print(f"Profile '{profile_name}' not found.")
        return None



# def is_object_public(s3_path, endpoint_url="https://s3.amazonaws.com", region="us-east-1"):
#     """
#     Check if an S3 object is public and exists.
    
#     Args:
#         s3_path (str): S3 URI (e.g., s3://bucket-name/object-key) or public S3 URL (e.g., https://s3.amazonaws.com/bucket-name/object-key).
        
#     Returns:
#         bool: True if the object exists and is publicly accessible, False otherwise.
#     """
#     # Convert S3 URI to public S3 URL
#     if s3_path.startswith("s3://"):
#         bucket_name, _, object_key = s3_path.replace("s3://", "").partition('/')
#         domain = endpoint_url.replace("s3.", f"s3.{region}.")
#         s3_url = f"{domain}/{bucket_name}/{object_key}"
#     else:
#         s3_url = s3_path  # Assume it's already a public URL

#     try:
#         # Perform a HEAD request to check if the object is public
#         response = requests.head(s3_url)
#         return response.status_code == 200
#     except requests.RequestException as e:        
#         print(f"Error checking object at url {s3_url}: {e}")
#         return False
        
# def is_object_public(s3_url, region='us-east-1'):
#     bucket_name, _, object_key = s3_url.strip("s3://").partition('/')
    
#     try:
#         # Create an S3 client
#         s3_client = boto3.client('s3', region_name=region)
        
#         # Get the ACL of the object
#         acl = s3_client.get_object_acl(Bucket=bucket_name, Key=object_key)
        
#         # Check if the ACL grants public read access
#         for grant in acl['Grants']:
#             grantee = grant.get('Grantee', {})
#             permission = grant.get('Permission')
#             if grantee.get('URI') == 'http://acs.amazonaws.com/groups/global/AllUsers' and permission == 'READ':
#                 return True
        
#         return False
    
#     except Exception as e:
#         if e.response['Error']['Code'] == 'NoSuchKey':
#             # Suppress warning for expected "key not found" error
#             return False
#         else:
#             warnings.warn(f"Error getting ACL for {object_key} in {bucket_name}: {e}")
#             return False

   
