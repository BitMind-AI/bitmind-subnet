from datasets import load_dataset
import argparse
import os

from bitmind.constants import DATASET_META


def download_datasets(download_mode: str, cache_dir: str):
    """ Downloads the datasets present in datasets.json

    Args:
        download_mode: either 'force_redownload' or 'use_cache_if_exists'
        cache_dir: huggingface cache directory. ~/.cache/huggingface by default 
    """ 
    os.makedirs(args.cache_dir, exist_ok=True)
    for dataset_type in DATASET_META:
        for dataset in DATASET_META[dataset_type]:
            print(f"Downloading {dataset['path']} dataset...")
            load_dataset(dataset['path'], cache_dir=args.cache_dir, )
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
