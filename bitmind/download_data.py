import datasets
from datasets import load_dataset
import argparse
import time
import sys
import os
import subprocess
import glob

from bitmind.constants import DATASET_META, HUGGINGFACE_CACHE_DIR

datasets.logging.set_verbosity_warning()
datasets.disable_progress_bar()


def clear_cache(cache_dir):
    """Clears lock files and incomplete downloads from the cache directory."""
    # Find lock and incomplete files
    lock_files = glob.glob(cache_dir + "/*lock")
    incomplete_files = glob.glob(cache_dir + "/downloads/**/*.incomplete", recursive=True)
    try:
        if lock_files:
            subprocess.run(["rm", *lock_files], check=True)
        if incomplete_files:
            for file in incomplete_files:
                os.remove(file)
        print("Hugging Face cache lock files cleared successfully.")
    except Exception as e:
        print(f"Failed to clear Hugging Face cache lock files: {e}")


def fix_permissions(path):
    """Attempts to fix permission issues on a given path."""
    try:
        subprocess.run(["chmod", "-R", "775", path], check=True)
        print(f"Fixed permissions for {path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to fix permissions for {path}: {e}")


def download_dataset(dataset_path, download_mode: str, cache_dir: str, max_wait: int = 300):
    """ Downloads the datasets present in datasets.json with exponential backoff
        download_mode: either 'force_redownload' or 'use_cache_if_exists'
        cache_dir: huggingface cache directory. ~/.cache/huggingface by default 
    """
    retry_wait = 10   # initial wait time in seconds
    attempts = 0     # initialize attempts counter
    print(f"Downloading {dataset_path} dataset...")
    while True:
        try:
            dataset = load_dataset(
                dataset_path,
                cache_dir=cache_dir,
                download_mode=download_mode,
                trust_remote_code=True)
            break
        except Exception as e:
            print(e)
            if '429' in str(e) or 'ReadTimeoutError' in str(e):
                print(f"Rate limit hit or timeout, retrying in {retry_wait}s...")
            elif isinstance(e, PermissionError):
                file_path = str(e).split(": '")[1].rstrip("'")
                print(f"Permission error at {file_path}, attempting to fix...")
                fix_permissions(file_path)  # Attempt to fix permissions directly
                clear_cache(cache_dir)      # Clear cache to remove any incomplete or locked files
            else:
                print(f"Unexpected error, stopping retries for {dataset_path}")
                raise e

            if retry_wait > max_wait:
                print(f"Download failed for {dataset_path} after {attempts} attempts. Try again later")
                sys.exit(1)

            time.sleep(retry_wait)
            retry_wait *= 2  # exponential backoff
            attempts += 1

    print(f"Downloaded {dataset_path} dataset to {cache_dir}")
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Hugging Face datasets for validator challenge generation and miner training.')
    parser.add_argument('--force_redownload', action='store_true', help='force redownload of datasets')
    parser.add_argument('--cache_dir', type=str, default=HUGGINGFACE_CACHE_DIR, help='huggingface cache directory')
    args = parser.parse_args()

    download_mode = "reuse_cache_if_exists"
    if args.force_redownload:
        download_mode = "force_redownload"

    os.makedirs(args.cache_dir, exist_ok=True)
    clear_cache(args.cache_dir)  # Clear the cache of lock and incomplete files.

    for dataset_type in DATASET_META:
        for dataset in DATASET_META[dataset_type]:
            download_dataset(dataset['path'], download_mode, args.cache_dir)
