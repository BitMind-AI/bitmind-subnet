from typing import Optional, Union, List, Tuple, Dict
import torchvision.transforms as transforms
import numpy as np
import datasets
import requests
import datasets

from bitmind.download_data import load_huggingface_dataset
from base_miner.datasets import ImageDataset, VideoDataset, RealFakeDataset

datasets.logging.set_verbosity_error()
datasets.disable_progress_bar()


def split_dataset(dataset):
    # Split data into train, validation, test and return the three splits
    dataset = dataset.shuffle(seed=42)

    if 'train' in dataset:
        dataset = dataset['train']

    split_dataset = {}
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    split_dataset['train'] = train_test_split['train']
    temp_dataset = train_test_split['test']

    # Split the temporary dataset into validation and test
    val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=42)
    split_dataset['validation'] = val_test_split['train']
    split_dataset['test'] = val_test_split['test']

    return split_dataset['train'], split_dataset['validation'], split_dataset['test']


def load_and_split_datasets(
    dataset_meta: list,
    modality: str,
    split_transforms: Dict[str, transforms.Compose] = {},
) -> Dict[str, List[ImageDataset]]:
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
        dataset = load_huggingface_dataset(meta['path'], None, meta.get('name'))
        train_ds, val_ds, test_ds = split_dataset(dataset)

        for split, data in zip(splits, [train_ds, val_ds, test_ds]):
            if modality == 'image':
                image_dataset = ImageDataset(huggingface_dataset=data, transforms=split_transforms.get(split, None))
            elif modality == 'video':
                image_dataset = VideoDataset(huggingface_dataset=data, transforms=split_transforms.get(split, None))
            else:
                raise NotImplementedError(f'Unsupported modality: {modality}')
            datasets[split].append(image_dataset)

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
        source_label_mapping=source_label_mapping)

    val_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['validation'],
        fake_image_datasets=fake_datasets['validation'],
        source_label_mapping=source_label_mapping)

    test_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['test'],
        fake_image_datasets=fake_datasets['test'],
        source_label_mapping=source_label_mapping)

    if source_labels:
        return train_dataset, val_dataset, test_dataset, source_label_mapping
    return train_dataset, val_dataset, test_dataset


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
