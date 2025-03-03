#!/usr/bin/env python3
"""
Compute the sha256sum of a file and rename it to include a substring of the hash.

The new filename will have the format:
    {stem}-hash{sha256sum[0:N]}{ext}
where:
    - stem: the original filename (without its extension)
    - ext: the original file extension (including the leading dot)
    - N: number of characters to keep from the sha256 hash (default = 8)

Example usage (when installed with visionlab.datasets):
    rename_with_hash /path/to/example.txt --hash_length=8

Example usage (python)
    python rename_with_hash.py /path/to/example.txt --hash_length=8
"""

import hashlib
import fire
from pathlib import Path
from pdb import set_trace

def compute_sha256(file_path):
    """
    Compute the SHA-256 hash of the file at file_path.

    Args:
        file_path (Path): The path to the file.

    Returns:
        str: The full hexadecimal SHA-256 hash.
    """
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        # Read the file in chunks to support large files.
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def split_name(path: Path):
    """Split a path into the stem and the complete extension (all suffixes)."""
    suffixes = path.suffixes
    if suffixes:
        ext = "".join(suffixes)
        stem = path.name[:-len(ext)]
    else:
        ext = ""
        stem = path.name
    return stem, ext

def rename_file_with_hash(file_path: str, hash_length: int = 8, dry_run: bool = False) -> str:
    """
    Compute the sha256sum of a file and rename the file to include the hash.

    The new file name is formatted as:
        {stem}-hash{sha256sum[0:hash_length]}{ext}
    where stem and ext are derived from the original file name.

    Args:
        file_path (str): The path to the file.
        hash_length (int): The number of characters to use from the sha256 hash (default: 8).

    Returns:
        str: The new file path as a string.
    """
    path = Path(file_path)
    if not path.is_file():
        raise ValueError(f"'{file_path}' is not a valid file.")
    
    # Compute the full sha256 hash.
    print(f"==> computing sha256 hash for file: {file_path}")
    full_hash = compute_sha256(path)
    
    # Get the full stem and extension (handling multiple extensions)
    stem, ext = split_name(path)
    
    # Construct the new file name.
    new_filename = f"{stem}-{full_hash[:hash_length]}{ext}"
    new_file_path = path.with_name(new_filename)
    
    # Rename the file.
    if dry_run:
        print(f"[Dry Run] File would be renamed:\n  From: {path}\n  To:   {new_file_path}")
    else:
        path.rename(new_file_path)
        print(f"Renamed file:\n  From: {path}\n  To:   {new_file_path}")
    
    return str(new_file_path)

def main():
    fire.Fire(rename_file_with_hash)

if __name__ == "__main__":
    fire.Fire(rename_file_with_hash)    