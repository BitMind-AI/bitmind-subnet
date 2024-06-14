from datasets import load_dataset
import argparse
import time
import sys
import os

from bitmind.constants import DATASET_META


def download_datasets(download_mode: str, cache_dir: str, max_wait: int = 300):
    """ Downloads the datasets present in datasets.json with exponential backoff

    Args:
        download_mode: either 'force_redownload' or 'use_cache_if_exists'
        cache_dir: huggingface cache directory. ~/.cache/huggingface by default 
    """ 
    os.makedirs(args.cache_dir, exist_ok=True)
    for dataset_type in DATASET_META:
        for dataset in DATASET_META[dataset_type]:
            retry_wait = 3   # initial wait time in seconds
            print(f"Downloading {dataset['path']} dataset...")
            while True:
                try:
                    load_dataset(dataset['path'], cache_dir=args.cache_dir, )
                    break
                except Exception as e:
                    print(e)
                    print(f"Error downloading {dataset['path']}. Retrying in {retry_wait}s...")
                    time.sleep(retry_wait)
                    if retry_wait > max_wait:
                        print(f"Download failed for {dataset['path']}. Try again later")
                        sys.exit(1)
                    retry_wait = retry_wait * 2

            print(f"Downloaded {dataset['path']} dataset to {args.cache_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Hugging Face datasets for validator challenge generation and miner training.')
    parser.add_argument('--force_redownload', action='store_true', help='force redownload of datasets')
    parser.add_argument('--cache_dir', type=str, default=os.path.expanduser('~/.cache/huggingface'), help='huggingface cache directory')
    args = parser.parse_args()

    download_mode = "reuse_cache_if_exists"
    if args.force_redownload:
        download_mode = "force_redownload" 
        
    download_datasets(download_mode, args.cache_dir)
