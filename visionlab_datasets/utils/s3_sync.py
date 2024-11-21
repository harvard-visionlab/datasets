import os
import subprocess
import signal

def s3_sync_data(from_dir, to_dir, storage_options=None, size_only=True, include=None, exclude=None):
    """
    Syncs the contents between a local directory and an S3 bucket in either direction.

    Args:
        from_dir (str): The source directory or S3 bucket, e.g., 's3://bucket-name/path' or '/local/path'.
        to_dir (str): The destination directory or S3 bucket, e.g., '/local/path' or 's3://bucket-name/path'.
        storage_options (dict): AWS credentials and configuration, including "aws_access_key_id", 
                                "aws_secret_access_key", and "endpoint_url".
        size_only (bool): If True, compare only file sizes when syncing.
        include (str or list): Patterns to include in the sync. Can be a string or a list of strings.
        exclude (str or list): Patterns to exclude in the sync. Can be a string or a list of strings.

    Example:
        Sync from local to S3:
            sync_data("/local/path", "s3://bucket-name/path", include="*.jpg", exclude="*.tmp")

        Sync from S3 to local:
            sync_data("s3://bucket-name/path", "/local/path", exclude="*.log")
    """
    # Ensure the local directory exists if it's the destination
    if not to_dir.startswith("s3://") and not os.path.exists(to_dir):
        os.makedirs(to_dir, exist_ok=True)
        
    # Normalize include and exclude to be lists
    if isinstance(include, str):
        include = [include]
    if isinstance(exclude, str):
        exclude = [exclude]

    # Construct the AWS CLI sync command
    sync_command = ["aws", "s3", "sync", from_dir, to_dir]

    if size_only:
        sync_command.append("--size-only")

    if include:
        for pattern in include:
            sync_command.extend(["--include", pattern])

    if exclude:
        for pattern in exclude:
            sync_command.extend(["--exclude", pattern])

    # Add custom endpoint URL if provided
    endpoint_url = storage_options.get("endpoint_url") if storage_options else None
    if endpoint_url:
        sync_command.extend(["--endpoint-url", endpoint_url])

    # Set up environment variables for AWS credentials
    env = os.environ.copy()
    if storage_options:
        aws_access_key_id = storage_options.get("aws_access_key_id")
        aws_secret_access_key = storage_options.get("aws_secret_access_key")
        if aws_access_key_id and aws_secret_access_key:
            env["AWS_ACCESS_KEY_ID"] = aws_access_key_id
            env["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

    # Execute the sync command
    try:
        print(f"==> Starting sync from {from_dir} to {to_dir}\n")
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
