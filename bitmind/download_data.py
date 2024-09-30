from typing import Optional
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import datasets
import argparse
import time
import sys
import os
import subprocess
import glob
import requests

from bitmind.constants import DATASET_META, HUGGINGFACE_CACHE_DIR

datasets.logging.set_verbosity_warning()
datasets.disable_progress_bar()


def load_huggingface_dataset(
    path: str,
    split: str = 'train',
    name: Optional[str] = None,
    download_mode: str = 'reuse_cache_if_exists',
) -> datasets.Dataset:
    """
    Load a dataset from Hugging Face or a local directory.

    Args:
        path (str): Path to the dataset or 'imagefolder:<directory>' for image folder. Can either be to a publicly
            hosted huggingface datset with the format <organizatoin>/<datset-name> or a local directory with the format
            imagefolder:<path/to/directory>
        split (str, optional): Name of the dataset split to load (default: None).
            Make sure to check what splits are available for the datasets you're working with.
        name (str, optional): Name of the dataset (if loading from Hugging Face, default: None).
            Some huggingface datasets provide various subets of different sizes, which can be accessed via thi
            parameter.
        download_mode (str, optional): Download mode for the dataset (if loading from Hugging Face, default: None).
            can be None or "force_redownload"
    Returns:
        Union[dict, load_dataset.Dataset]: The loaded dataset or a specific split of the dataset as requested.
    """
    if 'imagefolder' in path:
        _, directory = path.split(':')
        if name:
            dataset = load_dataset(path='imagefolder', name=name, data_dir=directory)
        else:
            dataset = load_dataset(path='imagefolder', data_dir=directory)
    else:
        dataset = download_dataset(
            dataset_path=path,
            dataset_name=name,
            download_mode=download_mode,
            cache_dir=HUGGINGFACE_CACHE_DIR)

    if split is None:
        return dataset

    return dataset[split]


def download_image(url: str) -> Image.Image:
    """
    Download an image from a URL.

    Args:
        url (str): The URL of the image to download.

    Returns:
        Image.Image or None: The downloaded image as a PIL Image object if
            successful, otherwise None.
    """
    response = requests.get(url)
    if response.status_code == 200:
        image_data = BytesIO(response.content)
        return Image.open(image_data)

    else:
        #print(f"Failed to download image: {response.status_code}")
        return None


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


def download_dataset(
    dataset_path: str,
    dataset_name: str,
    download_mode: str,
    cache_dir: str,
    max_wait: int = 300
):
    """ Downloads the datasets present in datasets.json with exponential backoff
        download_mode: either 'force_redownload' or 'use_cache_if_exists'
        cache_dir: huggingface cache directory. ~/.cache/huggingface by default 
    """
    retry_wait = 10   # initial wait time in seconds
    attempts = 0     # initialize attempts counter
    print(f"Downloading {dataset_path} (subset={dataset_name}) dataset...")
    while True:
        try:
            if dataset_name:
                dataset = load_dataset(dataset_path,
                                       name=dataset_name, #config/subset name
                                       cache_dir=cache_dir,
                                       download_mode=download_mode,
                                       trust_remote_code=True)
            else:
                dataset = load_dataset(dataset_path,
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
            download_dataset(dataset['path'], dataset.get('name', None), download_mode, args.cache_dir)
