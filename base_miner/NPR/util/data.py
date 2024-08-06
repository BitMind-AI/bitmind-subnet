from typing import List, Tuple, Dict
import torchvision.transforms as transforms

from bitmind.real_fake_dataset import RealFakeDataset
from bitmind.image_dataset import ImageDataset
from bitmind.constants import DATASET_META

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
        print(f"Loading {meta['path']} for all splits... ", end='')
        dataset = ImageDataset(
            meta['path'],
            meta.get('name', None),
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
            split_dataset.huggingface_datset_name = dataset.huggingface_datset_name
            split_dataset.sampled_images_idx = dataset.sampled_images_idx

            # Append to the corresponding split list
            datasets[split].append(split_dataset)

        split_lengths = ', '.join([f"{split} len={len(datasets[split][0])}" for split in splits])
        print(f'done, {split_lengths}')

    return datasets

def load_datasets(dataset_meta: dict = DATASET_META) -> Tuple[
        Dict[str, List[ImageDataset]],
        Dict[str, List[ImageDataset]]
    ]:
    """
    Loads several ImageDatasets, each of which is an abstraction of a huggingface dataset.

    Args:
        dataset_meta: dictionary containing metadata about the real and fake image datasets
             to load. See datasets.json.

    Returns:
        (real_datasets: Dict[str, List[ImageDataset]], fake_datasets: Dict[str, List[ImageDataset]])

    """
    fake_datasets = load_and_split_datasets(dataset_meta['fake'])
    real_datasets = load_and_split_datasets(dataset_meta['real'])

    return real_datasets, fake_datasets


def create_real_fake_datasets(
    real_datasets: Dict[str, List[ImageDataset]],
    fake_datasets: Dict[str, List[ImageDataset]],
    train_transforms: transforms.Compose,
    val_transforms: transforms.Compose,
    test_transforms: transforms.Compose,
) -> Tuple[RealFakeDataset, ...]:
    """

    Args:
        real_datasets: Dict containing train, val, and test keys. Each key maps to a list of ImageDatasets
        fake_datasets: Dict containing train, val, and test keys. Each key maps to a list of ImageDatasets
        base_transforms: transforms to apply to all images
        train_transforms: transforms to apply to training dataset
        val_transforms: transforms to apply to val dataset
        test_transforms: transforms to apply to val dataset
    Returns:
        Train, val, and test RealFakeDatasets

    """

    train_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['train'],
        fake_image_datasets=fake_datasets['train'],
        transforms=train_transforms)

    val_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['validation'],
        fake_image_datasets=fake_datasets['validation'],
        transforms=val_transforms)

    test_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['test'],
        fake_image_datasets=fake_datasets['test'],
        transforms=test_transforms)

    return train_dataset, val_dataset, test_dataset
