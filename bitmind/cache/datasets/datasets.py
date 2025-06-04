"""
Dataset definitions for the validator cache system
"""

from typing import List

from bitmind.types import Modality, MediaType, DatasetConfig


def get_image_datasets() -> List[DatasetConfig]:
    """
    Get the list of image datasets used by the validator.

    Returns:
        List of image dataset configurations
    """
    return [
        # Real image datasets
        DatasetConfig(
            path="bitmind/bm-eidon-image",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["frontier"],
        ),
        DatasetConfig(
            path="bitmind/bm-real",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
        ),
        DatasetConfig(
            path="bitmind/open-image-v7-256",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["diverse"],
        ),
        DatasetConfig(
            path="bitmind/celeb-a-hq",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["faces", "high-quality"],
        ),
        DatasetConfig(
            path="bitmind/ffhq-256",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["faces", "high-quality"],
        ),
        DatasetConfig(
            path="bitmind/MS-COCO-unique-256",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["diverse"],
        ),
        DatasetConfig(
            path="bitmind/AFHQ",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["animals", "high-quality"],
        ),
        DatasetConfig(
            path="bitmind/lfw",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["faces"],
        ),
        DatasetConfig(
            path="bitmind/caltech-256",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["objects", "categorized"],
        ),
        DatasetConfig(
            path="bitmind/caltech-101",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["objects", "categorized"],
        ),
        DatasetConfig(
            path="bitmind/dtd",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["textures"],
        ),
        DatasetConfig(
            path="bitmind/idoc-mugshots-images",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["faces"],
        ),
        # Synthetic image datasets
        DatasetConfig(
            path="bitmind/JourneyDB",
            type=Modality.IMAGE,
            media_type=MediaType.SYNTHETIC,
            tags=["midjourney"],
        ),
        DatasetConfig(
            path="bitmind/GenImage_MidJourney",
            type=Modality.IMAGE,
            media_type=MediaType.SYNTHETIC,
            tags=["midjourney"],
        ),
        DatasetConfig(
            path="bitmind/bm-aura-imagegen",
            type=Modality.IMAGE,
            media_type=MediaType.SYNTHETIC,
            tags=["sora"],
        ),
        # Semisynthetic image datasets
        DatasetConfig(
            path="bitmind/face-swap",
            type=Modality.IMAGE,
            media_type=MediaType.SEMISYNTHETIC,
            tags=["faces", "manipulated"],
        ),
    ]


def get_video_datasets() -> List[DatasetConfig]:
    """
    Get the list of video datasets used by the validator.
    """
    return [
        # Real video datasets
        DatasetConfig(
            path="bitmind/bm-eidon-video",
            type=Modality.VIDEO,
            media_type=MediaType.REAL,
            tags=["frontier"],
            compressed_format="zip",
        ),
        DatasetConfig(
            path="shangxd/imagenet-vidvrd",
            type=Modality.VIDEO,
            media_type=MediaType.REAL,
            tags=["diverse"],
            compressed_format="zip",
        ),
        DatasetConfig(
            path="nkp37/OpenVid-1M",
            type=Modality.VIDEO,
            media_type=MediaType.REAL,
            tags=["diverse", "large-zips"],
            compressed_format="zip",
        ),
        # Semisynthetic video datasets
        DatasetConfig(
            path="bitmind/semisynthetic-video",
            type=Modality.VIDEO,
            media_type=MediaType.SEMISYNTHETIC,
            tags=["faces"],
            compressed_format="zip",
        ),
    ]


def initialize_dataset_registry():
    """
    Initialize and populate the dataset registry.

    Returns:
        Fully populated DatasetRegistry instance
    """
    from bitmind.cache.datasets.dataset_registry import DatasetRegistry

    registry = DatasetRegistry()

    registry.register_all(get_image_datasets())
    registry.register_all(get_video_datasets())

    return registry
