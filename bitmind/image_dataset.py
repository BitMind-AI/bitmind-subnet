from typing import List, Tuple
from datasets import Dataset
from PIL import Image
from io import BytesIO
import bittensor as bt
import numpy as np

from bitmind.download_data import load_huggingface_dataset, download_image


class ImageDataset:

    def __init__(
        self,
        huggingface_dataset_path: str = None,
        huggingface_dataset_split: str = 'train',
        huggingface_dataset_name: str = None,
        huggingface_dataset: Dataset = None,
        download_mode: str = None
    ):
        """
        Args:
            huggingface_dataset_path (str): Path to the Hugging Face dataset. Can either be to a publicly hosted
                huggingface dataset (<organizatoin>/<dataset-name>) or a local directory (imagefolder:<path/to/directory>)
            huggingface_dataset_split (str): Split of the dataset to load (default: 'train').
                Make sure to check what splits are available for the datasets you're working with.
            huggingface_dataset_name (str): Name of the Hugging Face dataset (default: None).
                Some huggingface datasets provide various subets of different sizes, which can be accessed via thi
                parameter.
            create_splits (bool): Whether to create dataset splits (default: False).
                If the huggingface dataset hasn't been pre-split (i.e., it only contains "Train"), we split it here
                randomly.
            download_mode (str): Download mode for the dataset (default: None).
                can be None or "force_redownload"
        """
        assert huggingface_dataset_path is not None or huggingface_dataset is not None, \
            "Either huggingface_dataset_path or huggingface_dataset must be provided."
        
        if huggingface_dataset:
            self.dataset = huggingface_dataset
            self.huggingface_dataset_path = self.dataset.info.dataset_name
            self.huggingface_dataset_split = list(self.dataset.info.splits.keys())[0]
            self.huggingface_dataset_name = self.dataset.info.config_name

        else:
            self.huggingface_dataset_path = huggingface_dataset_path
            self.huggingface_dataset_name = huggingface_dataset_name
            self.dataset = load_huggingface_dataset(
                huggingface_dataset_path,
                huggingface_dataset_split,
                huggingface_dataset_name,
                download_mode)
        self.sampled_images_idx = []

    def __getitem__(self, index: int) -> dict:
        """
        Get an item (image and ID) from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing 'image' (PIL image) and 'id' (str).
        """
        return self._get_image(index)

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.dataset)

    def _get_image(self, index: int) -> dict:
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

        return {
            'image': image,
            'id': image_id,
            'source': self.huggingface_dataset_path
        }

    def sample(self, k: int = 1) -> Tuple[List[dict], List[int]]:
        """
        Randomly sample k images from self.dataset. Includes retries for failed downloads, in the case that
        self.dataset contains urls.

        Args:
            k (int): Number of images to sample (default: 1).

        Returns:
            Tuple[List[dict], List[int]]: A tuple containing a list of sampled images and their indices.
        """
        sampled_images = []
        sampled_idx = []
        while k > 0:
            attempts = len(self.dataset) // 2
            for i in range(attempts):
                image_idx = np.random.randint(0, len(self.dataset))
                if image_idx not in self.sampled_images_idx:
                    break
                if i >= attempts:
                    self.sampled_images_idx = []
            try:
                image = self._get_image(image_idx)
                if image['image'] is not None:
                    sampled_images.append(image)
                    sampled_idx.append(image_idx)
                    self.sampled_images_idx.append(image_idx)
                    k -= 1
            except Exception as e:
                bt.logging.error(e)
                continue

        return sampled_images, sampled_idx
