import boto3
import requests
from s3_filestore import auth
    
def get_signed_url(bucket_name, bucket_key, expires_in_seconds=3600, profile=None):
    s3_client = auth.get_client_with_userdata(profile)
    signed_url = s3_client.generate_presigned_url('get_object', 
                                                  Params={'Bucket': bucket_name, 'Key': bucket_key},
                                                  ExpiresIn=expires_in_seconds,
                                                  HttpMethod='GET')
    return signed_url

def sign_url_if_needed(url):
    return auth.sign_url_if_needed(url)
