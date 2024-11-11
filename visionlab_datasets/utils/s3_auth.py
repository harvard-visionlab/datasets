import os
import boto3
import warnings
from botocore.exceptions import ProfileNotFound
from botocore.configloader import load_config
import requests
# from s3_filestore import auth
    
# def get_signed_url(bucket_name, bucket_key, expires_in_seconds=3600, profile=os.environ.get('S3_PROFILE', None)):
#     s3_client = auth.get_client_with_userdata(profile)
#     signed_url = s3_client.generate_presigned_url('get_object', 
#                                                   Params={'Bucket': bucket_name, 'Key': bucket_key},
#                                                   ExpiresIn=expires_in_seconds,
#                                                   HttpMethod='GET')
#     return signed_url

# def sign_url_if_needed(url):
#     return auth.sign_url_if_needed(url)

def get_aws_credentials(profile_name=None):
    if profile_name is not None:
        return get_credentials_by_profile(profile_name)
    
    return {
        "aws_access_key_id": os.getenv('AWS_ACCESS_KEY_ID'),
        "aws_secret_access_key": os.getenv('AWS_SECRET_ACCESS_KEY'),
        "endpoint_url": os.getenv('AWS_ENDPOINT_URL', 'https://s3.amazonaws.com')
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
        
        # Return credentials and endpoint URL
        return {
            "aws_access_key_id": credentials.access_key,
            "aws_secret_access_key": credentials.secret_key,
            "endpoint_url": endpoint_url or 'https://s3.amazonaws.com'
        }

    except ProfileNotFound:
        print(f"Profile '{profile_name}' not found.")
        return None

def is_object_public(s3_url, region='us-east-1'):
    bucket_name, _, object_key = s3_url.strip("s3://").partition('/')
    
    try:
        # Create an S3 client
        s3_client = boto3.client('s3', region_name=region)
        
        # Get the ACL of the object
        acl = s3_client.get_object_acl(Bucket=bucket_name, Key=object_key)
        
        # Check if the ACL grants public read access
        for grant in acl['Grants']:
            grantee = grant.get('Grantee', {})
            permission = grant.get('Permission')
            if grantee.get('URI') == 'http://acs.amazonaws.com/groups/global/AllUsers' and permission == 'READ':
                return True
        
        return False
    
    except Exception as e:
        warnings.warn(f"Error getting ACL for {object_key} in {bucket_name}: {e}")
        return False

def is_object_private(s3_url, region='us-east-1'):
    return not is_object_public(s3_url, region)    