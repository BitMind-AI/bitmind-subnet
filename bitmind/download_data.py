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

from bitmind.constants import IMAGE_DATASET_META, VIDEO_DATASET_META, HUGGINGFACE_CACHE_DIR

datasets.logging.set_verbosity_warning()
datasets.disable_progress_bar()


def load_huggingface_dataset(
    path: str,
    split: str = 'train',
    name: Optional[str] = None,
    download_mode: str = 'reuse_cache_if_exists',
    local_data_path: Optional[str] = None
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
        local_data_path (str, optional): Some datasets on Hugging Face only contain metadata and require a manual
            download step to acquire the actual media. This path specifies where said media will be stored. 
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
            cache_dir=HUGGINGFACE_CACHE_DIR,
            local_data_path=local_data_path)

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


def download_dataset(
    dataset_path: str,
    dataset_name: str,
    download_mode: str,
    cache_dir: str,
    local_data_path: str = None,
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
            if dataset_name is not None:
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

            if local_data_path is not None:
                print(f"Downloading media for {dataset_path} to {local_data_path}")
                download_media(dataset_path, local_data_path)

            break
        except Exception as e:
            print(e)
            if '429' in str(e) or 'ReadTimeoutError' in str(e):
                print(f"Rate limit hit or timeout, retrying in {retry_wait}s...")
            elif isinstance(e, PermissionError):
                file_path = str(e).split(": '")[1].rstrip("'")
                print(f"Permission error at {file_path}, attempting to fix...")
                fix_permissions(file_path)  # Attempt to fix permissions directly
                clean_cache(cache_dir)      # Clear cache to remove any incomplete or locked files
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


def clean_cache(cache_dir):
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


def download_media(dataset_path, local_data_path):
    if dataset_path == 'nkp37/OpenVid-1M':
        download_openvid1m_files(local_data_path)
    else:
        raise NotImplementedError


def download_openvid1m_files(output_directory):
    """ Downloads the actual video data associated with the metadata in the OpenVid-1M huggingface dataset """
    zip_folder = os.path.join(output_directory, "download")
    video_folder = os.path.join(output_directory, "video")
    os.makedirs(zip_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    error_log_path = os.path.join(zip_folder, "download_log.txt")

    for i in range(0, 186):
        url = f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}.zip"
        file_path = os.path.join(zip_folder, f"OpenVid_part{i}.zip")
        if os.path.exists(file_path):
            print(f"file {file_path} exits.")
            continue

        command = ["wget", "-O", file_path, url]
        unzip_command = ["unzip", "-j", file_path, "-d", video_folder]
        try:
            subprocess.run(command, check=True)
            print(f"file {url} saved to {file_path}")
            print(unzip_command)
            subprocess.run(unzip_command, check=True)
        except subprocess.CalledProcessError as e:
            error_message = f"file {url} download failed: {e}\n"
            print(error_message)
            with open(error_log_path, "a") as error_log_file:
                error_log_file.write(error_message)
            
            part_urls = [
                f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}_partaa",
                f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}_partab"
            ]

            for part_url in part_urls:
                part_file_path = os.path.join(zip_folder, os.path.basename(part_url))
                if os.path.exists(part_file_path):
                    print(f"file {part_file_path} exits.")
                    continue

                part_command = ["wget", "-O", part_file_path, part_url]
                try:
                    subprocess.run(part_command, check=True)
                    print(f"file {part_url} saved to {part_file_path}")
                except subprocess.CalledProcessError as part_e:
                    part_error_message = f"file {part_url} download failed: {part_e}\n"
                    print(part_error_message)
                    with open(error_log_path, "a") as error_log_file:
                        error_log_file.write(part_error_message)
            file_path = os.path.join(zip_folder, f"OpenVid_part{i}.zip")
            cat_command = "cat " + os.path.join(zip_folder, f"OpenVid_part{i}_part*") + " > " + file_path
            unzip_command = ["unzip", "-j", file_path, "-d", video_folder]
            os.system(cat_command)
            subprocess.run(unzip_command, check=True)
    
    data_folder = os.path.join(output_directory, "data", "train")
    os.makedirs(data_folder, exist_ok=True)
    data_urls = [
        "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv",
        "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVidHD.csv"
    ]
    for data_url in data_urls:
        data_path = os.path.join(data_folder, os.path.basename(data_url))
        command = ["wget", "-O", data_path, data_url]
        subprocess.run(command, check=True)

    # delete zip files
    # delete_command = "rm -rf " + zip_folder
    # os.system(delete_command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Hugging Face datasets for validator challenge generation and miner training.')
    parser.add_argument('--force_redownload', action='store_true', help='force redownload of datasets')
    parser.add_argument('--modality', default='image', choices=['video', 'image'], help='download image or video datasets')
    parser.add_argument('--cache_dir', type=str, default=HUGGINGFACE_CACHE_DIR, help='huggingface cache directory')
    args = parser.parse_args()

    download_mode = "reuse_cache_if_exists"
    if args.force_redownload:
        download_mode = "force_redownload"

    os.makedirs(args.cache_dir, exist_ok=True)
    clean_cache(args.cache_dir)  # Clear the cache of lock and incomplete files.

    if args.modality == 'image':
        dataset_meta = IMAGE_DATASET_META
    elif args.modality == 'video':
        dataset_meta = VIDEO_DATASET_META
    
    for dataset_type in dataset_meta:
        for dataset in dataset_meta[dataset_type]:
            download_dataset(
                dataset_path=dataset['path'], 
                dataset_name=dataset.get('name', None), 
                download_mode=download_mode, 
                local_data_path=dataset.get('local_data_path', None), 
                cache_dir=args.cache_dir)
