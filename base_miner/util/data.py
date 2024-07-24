from typing import List, Tuple, Dict
import torchvision.transforms as transforms

from bitmind.real_fake_dataset import RealFakeDataset
from bitmind.image_dataset import ImageDataset
from bitmind.constants import DATASET_META


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
        (real_datasets: Dict[str, ImageDataset], fake_datasets: Dict[str, ImageDataset])

    """
    splits = ['train', 'validation', 'test']

    fake_datasets = {split: [] for split in splits}
    for split in splits:
        for meta in dataset_meta['fake']:
            print(f"Loading {meta['path']}[{split}] ... ", end='')
            dataset = ImageDataset(
                meta['path'],
                split,
                meta.get('name', None),
                meta['create_splits'],
                meta.get('download_mode', None)
            )
            fake_datasets[split].append(dataset)
            print(f'done, len={len(dataset)}')

    real_datasets = {split: [] for split in splits}
    for split in splits:
        for meta in dataset_meta['real']:
            print(f"Loading {meta['path']}[{split}] ... ", end='')
            dataset = ImageDataset(
                meta['path'],
                split,
                meta.get('name', None),
                meta['create_splits'],
                meta.get('download_mode', None)
            )
            real_datasets[split].append(dataset)
            print(f'done, len={len(dataset)}')

    return real_datasets, fake_datasets


def create_real_fake_datasets(
    real_datasets: Dict[str, List[ImageDataset]],
    fake_datasets: Dict[str, List[ImageDataset]],
    base_transforms: transforms.Compose,
    data_aug_transforms: transforms.Compose = None,
    augment_test: bool = False
) -> Tuple[RealFakeDataset, ...]:
    """

    Args:
        real_datasets: Dict containing train, val, and test keys. Each key maps to a list of ImageDatasets
        fake_datasets: Dict containing train, val, and test keys. Each key maps to a list of ImageDatasets

    Returns:
        Train, val, and test RealFakeDatasets

    """
    test_transforms = base_transforms
    if data_aug_transforms is not None:
        train_transforms = transforms.Compose(base_transforms.transforms + data_aug_transforms.transforms)
        if augment_test:
            test_transforms = train_transforms
    else:
        train_transforms = base_transforms

    train_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['train'],
        fake_image_datasets=fake_datasets['train'],
        transforms=train_transforms)

    val_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['validation'],
        fake_image_datasets=fake_datasets['validation'],
        transforms=base_transforms)

    test_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['test'],
        fake_image_datasets=fake_datasets['test'],
        transforms=test_transforms,
        )

    return train_dataset, val_dataset, test_dataset
