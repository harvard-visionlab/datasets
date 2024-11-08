import boto3

from ..auth import get_aws_credentials
from .cache_dir import get_cache_dir

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
        print(f"Error getting ACL for {object_key} in {bucket_name}: {e}")
        return False

def is_object_private(s3_url, region='us-east-1'):
    return not is_object_public(s3_url, region)
