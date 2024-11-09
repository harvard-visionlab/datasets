import boto3
from botocore.exceptions import ClientError

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

def s3_file_exists(s3_url, region='us-east-1', profile_name=None):
    # Parse the bucket name and file key from the S3 URL
    bucket_name, _, file_key = s3_url.strip("s3://").partition('/')

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