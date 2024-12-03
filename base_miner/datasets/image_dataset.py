from typing import List, Tuple, Optional
from datasets import Dataset
from PIL import Image
from io import BytesIO
import bittensor as bt
import numpy as np
from torchvision.transforms import Compose

from bitmind.download_data import load_huggingface_dataset, download_image
from .base_dataset import BaseDataset


class ImageDataset(BaseDataset):
    def __init__(
        self,
        huggingface_dataset_path: Optional[str] = None,
        huggingface_dataset_split: str = 'train',
        huggingface_dataset_name: Optional[str] = None,
        huggingface_dataset: Optional[Dataset] = None,
        download_mode: Optional[str] = None,
        transforms: Optional[Compose] = None,
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
            transforms=transforms
        )

    def __getitem__(self, index: int) -> dict:
        """
        Get an item (image and ID) from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing 'image' (PIL image) and 'id' (str).
        """
        """
        Load an image from self.dataset. Expects self.dataset[i] to be a dictionary containing either 'image' or 'url'
        as a key.
            - The value associated with the 'image' key should be either a PIL image or a b64 string encoding of
            the image.
            - The value associated with the 'url' key should be a url that hosts the image (as in
            dalle-mini/open-images)

        Args:
            index (int): Index of the image in the dataset.

        Returns:
            dict: Dictionary containing 'image' (PIL image) and 'id' (str).
        """
        sample = self.dataset[int(index)]
        if 'url' in sample:
            image = download_image(sample['url'])
            image_id = sample['url']
        elif 'image_url' in sample:
            image = download_image(sample['image_url'])
            image_id = sample['image_url']
        elif 'image' in sample:
            if isinstance(sample['image'], Image.Image):
                image = sample['image']
            elif isinstance(sample['image'], bytes):
                image = Image.open(BytesIO(sample['image']))
            else:
                raise NotImplementedError

            image_id = ''
            if 'name' in sample:
                image_id = sample['name']
            elif 'filename' in sample:
                 image_id = sample['filename']

            image_id = image_id if image_id != '' else index

        else:
            raise NotImplementedError

        # remove alpha channel if download didnt 404
        if image is not None:
            image = image.convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)

        return {
            'image': image,
            'id': image_id,
            'source': self.huggingface_dataset_path
        }

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.dataset)

