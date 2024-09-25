from typing import Optional, Union, List, Tuple, Dict
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import numpy as np
import datasets
import requests
import datasets

from bitmind.real_fake_dataset import RealFakeDataset
from bitmind.image_dataset import ImageDataset
from bitmind.download_data import download_dataset
from bitmind.constants import HUGGINGFACE_CACHE_DIR

datasets.logging.set_verbosity_error()
datasets.disable_progress_bar()


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


def load_and_split_datasets(dataset_meta: list) -> Dict[str, List[ImageDataset]]:
    """
    Helper function to load and split dataset into train, validation, and test sets.

    Args:
        dataset_meta: List containing metadata about the dataset to load.

    Returns:
        A dictionary with keys == "train", "validation", or "test" strings,
        and values == List[ImageDataset].

        Dict[str, List[ImageDataset]]

        e.g. given two dataset paths in dataset_meta,
        {'train': [<ImageDataset object>, <ImageDataset object>],
        'validation': [<ImageDataset object>, <ImageDataset object>],
        'test': [<ImageDataset object>, <ImageDataset object>]}
    """
    splits = ['train', 'validation', 'test']
    datasets = {split: [] for split in splits}

    for meta in dataset_meta:
        print(f"Loading {meta['path']} (subset={meta.get('name', None)}) for all splits... ")
        dataset = ImageDataset(
            huggingface_dataset_path=meta['path'],
            huggingface_dataset_split='train',
            huggingface_dataset_name=meta.get('name', None),
            create_splits=True, # dataset.dataset == (train, val, test) splits from load_huggingface_dataset(...)
            download_mode=meta.get('download_mode', None)
        )

        train_ds, val_ds, test_ds = dataset.dataset

        for split, data in zip(splits, [train_ds, val_ds, test_ds]):
            # Create a new ImageDataset instance without calling __init__
            # This avoids calling load_huggingface_dataset(...) and redownloading
            split_dataset = ImageDataset.__new__(ImageDataset)

            # Assign the appropriate split data
            split_dataset.dataset = data

            # Copy other attributes from the initial dataset
            split_dataset.huggingface_dataset_path = dataset.huggingface_dataset_path
            split_dataset.huggingface_dataset_name = dataset.huggingface_dataset_name
            split_dataset.sampled_images_idx = dataset.sampled_images_idx

            # Append to the corresponding split list
            datasets[split].append(split_dataset)

        split_lengths = ', '.join([f"{split} len={len(datasets[split][0])}" for split in splits])
        print(f'done, {split_lengths}')

    return datasets


def create_source_label_mapping(
    real_datasets: Dict[str, List[ImageDataset]],
    fake_datasets: Dict[str, List[ImageDataset]],
    group_by_name: bool = False
    ) -> Dict:

    source_label_mapping = {}
    grouped_source_labels = {}
    # Iterate through real datasets and set their source label to 0.0
    for split, dataset_list in real_datasets.items():
        for dataset in dataset_list:
            source = dataset.huggingface_dataset_path
            if source not in source_label_mapping.keys():
                source_label_mapping[source] = 0.0

    # Assign incremental labels to fake datasets
    for split, dataset_list in fake_datasets.items():
        for dataset in dataset_list:
            source = dataset.huggingface_dataset_path
            if group_by_name and '__' in source:
                model_name = source.split('__')[1]
                if model_name in grouped_source_labels:
                    fake_source_label = grouped_source_labels[model_name]
                else:
                    fake_source_label = max(source_label_mapping.values()) + 1
                    grouped_source_labels[model_name] = fake_source_label

                if source not in source_label_mapping:
                    source_label_mapping[source] = fake_source_label
            else:
                if source not in source_label_mapping:
                    source_label_mapping[source] = max(source_label_mapping.values()) + 1

    return source_label_mapping


def create_real_fake_datasets(
    real_datasets: Dict[str, List[ImageDataset]],
    fake_datasets: Dict[str, List[ImageDataset]],
    train_transforms: transforms.Compose = None,
    val_transforms: transforms.Compose = None,
    test_transforms: transforms.Compose = None,
    source_labels: bool = False,
    group_sources_by_name: bool = False) -> Tuple[RealFakeDataset, ...]:
    """
    Args:
        real_datasets: Dict containing train, val, and test keys. Each key maps to a list of ImageDatasets
        fake_datasets: Dict containing train, val, and test keys. Each key maps to a list of ImageDatasets
        train_transforms: transforms to apply to training dataset
        val_transforms: transforms to apply to val dataset
        test_transforms: transforms to apply to test dataset
    Returns:
        Train, val, and test RealFakeDatasets

    """
    source_label_mapping = None
    if source_labels:
        source_label_mapping = create_source_label_mapping(
            real_datasets, fake_datasets, group_sources_by_name)

    print(f"Source label mapping: {source_label_mapping}")

    train_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['train'],
        fake_image_datasets=fake_datasets['train'],
        transforms=train_transforms,
        source_label_mapping=source_label_mapping)

    val_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['validation'],
        fake_image_datasets=fake_datasets['validation'],
        transforms=val_transforms,
        source_label_mapping=source_label_mapping)

    test_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['test'],
        fake_image_datasets=fake_datasets['test'],
        transforms=test_transforms,
        source_label_mapping=source_label_mapping)

    if source_labels:
        return train_dataset, val_dataset, test_dataset, source_label_mapping
    return train_dataset, val_dataset, test_dataset


def load_huggingface_dataset(
    path: str,
    split: str = 'train',
    name: Optional[str] = None,
    download_mode: str = 'reuse_cache_if_exists',
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
        dataset = download_dataset(path, name=name, download_mode=download_mode, cache_dir=HUGGINGFACE_CACHE_DIR)

    if split is None:
        return dataset

    return dataset[split]


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
