from typing import List
from collections import defaultdict
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import bittensor as bt
import numpy as np
import requests
import base64


def download_image(url):
    #print(f'downloading {url}')
    response = requests.get(url)
    if response.status_code == 200:
        image_data = BytesIO(response.content)
        return Image.open(image_data)

    else:
        #print(f"Failed to download image: {response.status_code}")
        return None


def load_huggingface_dataset(path, split=None, name=None, create_splits=False, download_mode=None):
    if 'imagefolder' in path:
        _, directory = path.split(':')
        dataset = load_dataset(path='imagefolder', data_dir=directory, split='train')
    else:
        dataset = load_dataset(path, name=name, download_mode=download_mode)#, split=split)

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


class ImageDataset:

    def __init__(
        self,
        huggingface_dataset_path: str,
        huggingface_datset_split: str = 'train',
        huggingface_datset_name: str = None,
        create_splits: bool = False,
        download_mode: str = None
    ):
        self.huggingface_dataset_path = huggingface_dataset_path
        self.huggingface_datset_name = huggingface_datset_name
        self.dataset = load_huggingface_dataset(
            huggingface_dataset_path, huggingface_datset_split, huggingface_datset_name, create_splits, download_mode)
        self.sampled_images_idx = []

    def __getitem__(self, index):
        return self._get_image(index)

    def __len__(self):
        return len(self.dataset)

    def _get_image(self, index: int):
        """
        Loads an image from self.dataset. Expects self.dataset[i] to be a dictionary containing
        either 'image' or 'url' as a key.
            - The value associated with the 'image' key should be either a PIL image or a
                b64 string encoding of the image.
            - The value associated with the 'url' key should be a url that hosts the image
                (as in dalle-mini/open-images)

        Returns:
            {'image': PIL.image, 'id': str}
        """
        sample = self.dataset[int(index)]
        if 'url' in sample:
            image = download_image(sample['url'])
            image_id = sample['url']
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

        # check for/remove alpha channel if download didnt 404
        if image is not None and 'A' in image.mode:
            image = image.convert('RGB')

        return {
            'image': image,
            'id': image_id,
        }

    def sample(self, k=1):
        """
        Randomly samples k images from self.dataset. Includes retries for failed downloads,
        in the case that self.dataset contains urls.

        Returns:
            a tuple containing a list of the sampled images and a list of ther indices
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
                print(e)
                continue

        return sampled_images, sampled_idx
