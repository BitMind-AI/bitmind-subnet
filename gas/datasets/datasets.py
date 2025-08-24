"""
Dataset definitions for the validator cache system
"""

import os
import yaml
from typing import List
from pathlib import Path

from gas.types import Modality, MediaType, DatasetConfig


def _load_datasets_config() -> dict:
    """
    Load datasets configuration from YAML file.
    
    Returns:
        Dictionary containing dataset configurations
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    # Get the path to the config file in the same directory as this module
    config_path = Path(__file__).parent / "datasets.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Datasets config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in datasets config: {e}")


def _parse_dataset_configs(config_list: List[dict]) -> List[DatasetConfig]:
    """
    Parse a list of dataset configuration dictionaries into DatasetConfig objects.
    
    Args:
        config_list: List of dataset configuration dictionaries
        
    Returns:
        List of DatasetConfig objects
    """
    configs = []
    for config_dict in config_list:
        try:
            # Create DatasetConfig object - it will handle validation in __post_init__
            config = DatasetConfig(**config_dict)
            configs.append(config)
        except Exception as e:
            print(f"Warning: Failed to parse dataset config {config_dict.get('path', 'unknown')}: {e}")
            continue
    
    return configs


def get_image_datasets() -> List[DatasetConfig]:
    """
    Get the list of image datasets used by the validator.

    Returns:
        List of image dataset configurations loaded from config file
    """
    try:
        config = _load_datasets_config()
        image_configs = config.get('image_datasets', [])
        return _parse_dataset_configs(image_configs)
    except Exception as e:
        print(f"Error loading image datasets config: {e}")
        print("Falling back to empty dataset list.")
        return []


def get_video_datasets() -> List[DatasetConfig]:
    """
    Get the list of video datasets used by the validator.
    
    Returns:
        List of video dataset configurations loaded from config file
    """
    try:
        config = _load_datasets_config()
        video_configs = config.get('video_datasets', [])
        return _parse_dataset_configs(video_configs)
    except Exception as e:
        print(f"Error loading video datasets config: {e}")
        print("Falling back to empty dataset list.")
        return []


def get_mixed_datasets() -> List[DatasetConfig]:
    """
    Get the list of mixed datasets (containing both real and AI-generated content) used by the validator.
    These datasets contain labels that allow splitting content by actual media type at load time.
    
    Returns:
        List of mixed dataset configurations loaded from config file
    """
    try:
        config = _load_datasets_config()
        mixed_configs = config.get('mixed_datasets', [])
        return _parse_dataset_configs(mixed_configs)
    except Exception as e:
        print(f"Error loading mixed datasets config: {e}")
        print("Falling back to empty dataset list.")
        return []


def initialize_dataset_registry():
    """
    Initialize and populate the dataset registry.

    Returns:
        Fully populated DatasetRegistry instance
    """
    from gas.datasets.dataset_registry import DatasetRegistry

    registry = DatasetRegistry()

    registry.register_all(get_image_datasets())
    registry.register_all(get_video_datasets())

    return registry