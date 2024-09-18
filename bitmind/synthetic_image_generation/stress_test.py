import argparse
import logging
import json
import os
import torch
import time
from pathlib import Path
import pandas as pd
from math import ceil
from PIL import Image
import copy
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

from datasets import load_dataset

from synthetic_image_generator import SyntheticImageGenerator
from bitmind.image_dataset import ImageDataset
from bitmind.constants import TARGET_IMAGE_SIZE, DATASET_META
from utils.hugging_face_utils import (
    dataset_exists_on_hf, load_and_sort_dataset, upload_to_huggingface, 
    slice_dataset, save_as_json
)
from utils.image_utils import resize_image, resize_images_in_directory
from bitmind.image_dataset import ImageDataset
from bitmind.utils.data import sample_dataset_index_name


def slice_dataset(dataset, start_index, end_index=None):
    """
    Slice the dataset according to provided start and end indices.

    Parameters:
    dataset (Dataset): The dataset to be sliced.
    start_index (int): The index of the first element to include in the slice.
    end_index (int, optional): The index of the last element to include in the slice. If None, slices to the end of the dataset.

    Returns:
    Dataset: The sliced dataset.
    """
    if end_index is not None and end_index < len(dataset):
        return dataset.select(range(start_index, end_index))
    else:
        return dataset.select(range(start_index, len(dataset)))
        

def main():
    synthetic_image_generator = SyntheticImageGenerator(prompt_type='annotation',
                                        use_random_diffuser=False,
                                        diffuser_name='black-forest-labs/FLUX.1-dev')
    
    # celeb = ImageDataset('bitmind/celeb-a-hq', 'train')
    # ffhq = ImageDataset('bitmind/ffhq-256', 'train')
    # mscoco = ImageDataset('bitmind/MS-COCO-unique-256', 'train')

    real_image_datasets = [
        ImageDataset(ds['path'], 'train', ds.get('name', None), ds['create_splits'])
        for ds in DATASET_META['real']
    ]
    
    # sample image(s) from real dataset for captioning
    real_dataset_index, source_dataset = sample_dataset_index_name(real_image_datasets)
    real_dataset = real_image_datasets[real_dataset_index]
    images_to_caption, image_indexes = real_dataset.sample(k=1)  # [{'image': PIL Image ,'id': int}, ...]

    start = time.time()
    # generate captions for the real images, then synthetic images from these captions
    sample = synthetic_image_generator.generate(
        k=1, real_images=images_to_caption)[0]  # {'prompt': str, 'image': PIL Image ,'id': int}
    end = time.time()
    time_elapsed = end-start
    print(f"Time elapsed: {time_elapsed}")
    print(sample)

if __name__ == "__main__":
    main()
