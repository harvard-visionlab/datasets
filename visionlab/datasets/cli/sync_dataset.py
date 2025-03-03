import os
import signal
import subprocess
import fire

from visionlab.datasets.utils import get_aws_credentials

from pdb import set_trace

def sync_dataset(local_directory, bucket_location, profile=None, size_only=True):
    """
    Syncs the contents of a local directory to an S3 bucket using AWS CLI.

    Args:
        local_directory (str): The local directory where the contents are located.
        bucket_location (str): The S3 bucket location to sync to, e.g., 's3://visionlab-datasets/ecoset'.
        profile (str): The S3 profile to use. If None, default credentials are used.
        size_only (bool): If True, compare only file sizes when syncing.
        
    Example:
        sync_dataset \
        /n/alvarez_lab_tier1/Users/alvarez/datasets/imagenet-r-litdata/streaming-PILImage-q100/ \
        s3://visionlab-litdata/imagenet-r/streaming-PILImage-q100/ \
        --profile default
    
    """
    # Retrieve AWS credentials
    aws_credentials = get_aws_credentials(profile)
    aws_access_key_id = aws_credentials.get("aws_access_key_id")
    aws_secret_access_key = aws_credentials.get("aws_secret_access_key")
    endpoint_url = aws_credentials.get("endpoint_url")
    
    # Construct the AWS CLI sync command
    sync_command = [
        "aws", "s3", "sync", local_directory, bucket_location
    ]
    
    if size_only:
        sync_command.append("--size-only")
    if endpoint_url:
        sync_command.extend(["--endpoint-url", endpoint_url])
    
    # Set up environment variables
    env = os.environ.copy()
    if aws_access_key_id and aws_secret_access_key:
        env["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        env["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    
    # Run the AWS CLI command
    try:
        print(f"==> Starting sync from {local_directory} to {bucket_location}\n")
        print("==> Running command:", " ".join(sync_command))

        # Start the process in a new process group
        with subprocess.Popen(
            sync_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line-buffered
            env=env,
            preexec_fn=os.setsid  # Create a new process group for signal handling
        ) as process:
            try:
                # Stream output to the console
                for line in process.stdout:
                    print(f"\r{line.strip()}", end="")

                # Wait for process to complete
                returncode = process.wait()
            except KeyboardInterrupt:
                print("\nInterrupt received, stopping sync...")
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait()  # Ensure process terminates before continuing
                raise  # Re-raise the interrupt

        # Handle completion status
        if returncode == 0:
            print("\nSync completed successfully.")
        else:
            print(f"\nError: Sync process exited with return code {returncode}.")
    
    except KeyboardInterrupt:
        # Top-level interrupt handling
        print("\nSync interrupted by user. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def main():
    fire.Fire(sync_dataset)

if __name__ == "__main__":
    fire.Fire(sync_dataset)
