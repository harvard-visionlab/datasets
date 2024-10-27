import os
import boto3
from botocore.exceptions import ProfileNotFound
from botocore.configloader import load_config
import requests
from s3_filestore import auth
    
def get_signed_url(bucket_name, bucket_key, expires_in_seconds=3600, profile=os.environ.get('S3_PROFILE', None)):
    s3_client = auth.get_client_with_userdata(profile)
    signed_url = s3_client.generate_presigned_url('get_object', 
                                                  Params={'Bucket': bucket_name, 'Key': bucket_key},
                                                  ExpiresIn=expires_in_seconds,
                                                  HttpMethod='GET')
    return signed_url

def sign_url_if_needed(url):
    return auth.sign_url_if_needed(url)

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
            "endpoint_url": endpoint_url
        }

    except ProfileNotFound:
        print(f"Profile '{profile_name}' not found.")
        return None
