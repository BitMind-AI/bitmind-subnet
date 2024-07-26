from typing import Optional, Union
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import datasets
import numpy as np

from bitmind.download_data import download_dataset
from bitmind.constants import HUGGINGFACE_CACHE_DIR


def download_image(url: str) -> Image.Image:
    """
    Download an image from a URL.

    Args:
        url (str): The URL of the image to download.

    Returns:
        Image.Image or None: The downloaded image as a PIL Image object if successful,
                             otherwise None.
    """
    response = requests.get(url)
    if response.status_code == 200:
        image_data = BytesIO(response.content)
        return Image.open(image_data)

    else:
        #print(f"Failed to download image: {response.status_code}")
        return None


def load_huggingface_dataset(
    path: str,
    split: Optional[str] = None,
    name: Optional[str] = None,
    create_splits: bool = False,
    download_mode: Optional[str] = None
) -> Union[dict, datasets.Dataset]:
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
        create_splits (bool, optional): Whether to create train/validation/test splits (default: False).
            If the huggingface dataset hasn't been pre-split (i.e., it only contains "Train"), we split it here
            randomly.
        download_mode (str, optional): Download mode for the dataset (if loading from Hugging Face, default: None).
            can be None or "force_redownload"
    Returns:
        Union[dict, load_dataset.Dataset]: The loaded dataset or a specific split of the dataset as requested.
    """
    if 'imagefolder' in path:
        _, directory = path.split(':')
        dataset = load_dataset(path='imagefolder', data_dir=directory, split='train')
    else:
        download_dataset(path, "reuse_cache_if_exists", cache_dir=HUGGINGFACE_CACHE_DIR)

    if not create_splits:
        if split is not None:
            return dataset[split]
        return dataset

    dataset = dataset.shuffle(seed=42)

    split_dataset = {}
    train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
    split_dataset['train'] = train_test_split['train']
    temp_dataset = train_test_split['test']

    # Split the temporary dataset into validation and test
    val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=42)
    split_dataset['validation'] = val_test_split['train']
    split_dataset['test'] = val_test_split['test']
    return split_dataset[split]


def sample_dataset_index_name(image_datasets: list) -> tuple[int, str]:
    """
    Randomly selects a dataset index from the provided dataset list and returns the index and source name.

    Parameters
    ----------
    image_datasets : list
        A list of dataset objects to select from.

    Returns
    -------
    tuple[int, str]
        A tuple containing the index of the randomly selected dataset and the source name.
    """
    dataset_index = np.random.randint(0, len(image_datasets))
    source_name = image_datasets[dataset_index].huggingface_dataset_path
    return dataset_index, source_name