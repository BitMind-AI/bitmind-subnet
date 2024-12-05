"""
Author: Zhiyuan Yan
Email: zhiyuanyan@link.cuhk.edu.cn
Date: 2023-03-30
Description: Abstract Base Class for all types of deepfake datasets.
"""

import os
import cv2
from PIL import Image
import sys
import yaml
import numpy as np
from copy import deepcopy
import random
import torch
from torch import nn
from torch.utils import data
from torchvision.utils import save_image
from torchvision.transforms import Compose
from einops import rearrange
from typing import List, Tuple, Optional
from datasets import Dataset

from .base_dataset import BaseDataset


class VideoDataset(BaseDataset):
    def __init__(
        self,
        huggingface_dataset_path: Optional[str] = None,
        huggingface_dataset_split: str = 'train',
        huggingface_dataset_name: Optional[str] = None,
        huggingface_dataset: Optional[Dataset] = None,
        download_mode: Optional[str] = None,
        max_frames_per_video: Optional[int] = 4,
        transforms: Optional[Compose] = None
    ):
        """Initialize the ImageDataset.
        
        Args:
            huggingface_dataset_path (str, optional): Path to the Hugging Face dataset.
                Can be a publicly hosted dataset (<organization>/<dataset-name>) or 
                local directory (imagefolder:<path/to/directory>)
            huggingface_dataset_split (str): Dataset split to load. Defaults to 'train'.
            huggingface_dataset_name (str, optional): Name of the specific Hugging Face dataset subset.
            huggingface_dataset (Dataset, optional): Pre-loaded Hugging Face dataset instance.
            download_mode (str, optional): Download mode for the dataset.
                Can be None or "force_redownload"
        """
        super().__init__(
            huggingface_dataset_path=huggingface_dataset_path,
            huggingface_dataset_split=huggingface_dataset_split,
            huggingface_dataset_name=huggingface_dataset_name,
            huggingface_dataset=huggingface_dataset,
            download_mode=download_mode,
            transforms=transforms,
        )
        self.max_frames = max_frames_per_video

    def __getitem__(self, index):
        """Return the data point at the given index.

        Args:
            index (int): The index of the data point.
            no_norm (bool): Whether to skip normalization.

        Returns:
            tuple: Contains image tensor, label tensor, landmark tensor,
                  and mask tensor.
        """
        image_paths = self.dataset[index]['frames']

        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        images = []
        for image_path in image_paths[:self.max_frames]:
            try:
                img = Image.open(image_path)
                images.append(img)
            except Exception as e:
                print(f"Error loading image at index {index}: {e}")
                return self.__getitem__(0)

        if self.transforms is not None:
            images = self.transforms(images)

        # Stack images along the time dimension (frame_dim)
        image_tensors = torch.stack(images, dim=0)  # Shape: [frame_dim, C, H, W]
        
        frames, channels, height, width = image_tensors.shape
        x = torch.randint(0, width, (1,)).item()
        y = torch.randint(0, height, (1,)).item()
        mask_grid_size = 16
        x1 = max(x - mask_grid_size // 2, 0)
        x2 = min(x + mask_grid_size // 2, width)
        y1 = max(y - mask_grid_size // 2, 0)
        y2 = min(y + mask_grid_size // 2, height)
        image_tensors[:, :, y1:y2, x1:x2] = -1  

        return {
            'image': image_tensors,  # Shape: [frame_dim, C, H, W]
            'id': self.dataset[index]['video_id'],
            'source': self.huggingface_dataset_path
        }


    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.dataset['video_id'])