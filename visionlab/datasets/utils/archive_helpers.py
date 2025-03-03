import os
import zipfile
import tarfile

from pdb import set_trace

def get_top_level_directory_fast(file_path):
    with tarfile.open(file_path, 'r:*') as tar:
        for member in tar:
            if '/' in member.name:
                return member.name.split('/')[0]
    return None

def decompress_tarfile_if_needed(file_path, output_dir=None):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")

    # Determine the directory of the tar file
    dir_name = os.path.dirname(file_path) or '.'

    # Set the output directory. If none is provided, default to the tar file's location
    if output_dir is None:
        output_dir = dir_name

    # If the output directory doesn't exist, create it
    os.makedirs(output_dir, exist_ok=True)

    # Assuming the first folder inside the tar file is the root folder of its contents
    # This will be used to check if the contents have already been extracted
    top_folder_name = get_top_level_directory_fast(file_path)
    expected_extracted_folder = os.path.join(output_dir, top_folder_name)
    
    # Check if the contents have already been extracted
    if os.path.exists(expected_extracted_folder):
        print(f"Contents have already been extracted to {expected_extracted_folder}.")
    else:
        # Contents have not been extracted; proceed with extraction
        print(f"Extracting {file_path} to {output_dir}")
        with tarfile.open(file_path, 'r:*') as tar:
            tar.extractall(path=output_dir)
            print(f"File {file_path} has been decompressed to {output_dir}.")

    return expected_extracted_folder

def decompress_zipfile_if_needed(file_path, output_dir=None):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")

    # Determine the directory of the zip file
    dir_name = os.path.dirname(file_path) or '.'

    # Set the output directory. If none is provided, default to the zip file's location
    if output_dir is None:
        output_dir = dir_name

    # If the output directory doesn't exist, create it
    os.makedirs(output_dir, exist_ok=True)

    # Assuming the first folder inside the zip file is the root folder of its contents
    # This will be used to check if the contents have already been extracted
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        top_folder_name = zip_ref.namelist()[0].split('/')[0]
        expected_extracted_folder = os.path.join(output_dir, top_folder_name)

        # Check if the contents have already been extracted
        if os.path.exists(expected_extracted_folder):
            print(f"Contents have already been extracted to {expected_extracted_folder}.")
        else:
            # Contents have not been extracted; proceed with extraction
            print(f"Extracting {file_path} to {output_dir}")
            zip_ref.extractall(path=output_dir)
            print(f"File {file_path} has been decompressed to {output_dir}.")

    return expected_extracted_folder

def decompress_if_needed(file_path, output_dir=None, ignore_non_archives=True):
    # Determine the file extension and call the appropriate decompression function
    if (not file_path.endswith('.pth.tar')) and (file_path.endswith('.tar') or file_path.endswith('.tar.gz') or file_path.endswith('.tgz')):
        return decompress_tarfile_if_needed(file_path, output_dir)
    elif file_path.endswith('.zip'):
        return decompress_zipfile_if_needed(file_path, output_dir)
    elif ignore_non_archives:
        return file_path
    else:
        raise ValueError("Unsupported file type. Only .tar, .tar.gz, .tgz, and .zip files are supported.")
