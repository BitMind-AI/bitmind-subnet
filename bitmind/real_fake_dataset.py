import numpy as np
from torchvision import transforms as T
import torch

class RealFakeDataset:

    def __init__(
        self,
        real_image_datasets: list,
        fake_image_datasets: list,
        transforms=None,
        fake_prob=0.5,
        source_label_mapping=None
    ):
        """
        Initialize the RealFakeDataset instance.

        Args:
            real_image_datasets (list): List of ImageDataset objects containing real images
            fake_image_datasets (list): List of ImageDataset objects containing real images
            transforms (transforms.Compose): Image transformations (default: None).
            fake_prob (float): Probability of selecting a fake image (default: 0.5).
            source_label_mapping (dict): A dictionary mapping dataset names to float labels.
        """
        self.real_image_datasets = real_image_datasets
        self.fake_image_datasets = fake_image_datasets
        self.transforms = transforms
        self.fake_prob = fake_prob
        self.source_label_mapping = source_label_mapping

        self._history = {
            'source': [],
            'index': [],
            'label': [],
        }
    
    def __getitem__(self, index: int) -> tuple:
        """
        Retrieve an item (image, label) from the dataset.
        By default, 50/50 chance of real or fake. This can be overidden by self.fake_prob

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the image, its label (1 : fake, 0 : real),
            and its source label (0 for real datasets and >= 1 for fake datasets).
        """
        if len(self._history['index']) > index:
            self.reset()

        if np.random.rand() > self.fake_prob:
            source = self.fake_image_datasets[np.random.randint(0, len(self.fake_image_datasets))]
            image = source[index]['image']
            label = 1.0
        else:
            source = self.real_image_datasets[np.random.randint(0, len(self.real_image_datasets))]
            imgs, idx = source.sample(1)
            image = imgs[0]['image']
            index = idx[0]
            label = 0.0

        self._history['source'].append(source.huggingface_dataset_path)
        self._history['label'].append(label)
        self._history['index'].append(index)
        
        try:
            if self.transforms is not None:
                image = self.transforms(image)
        except Exception as e:
            print(e)
            print(source.huggingface_dataset_path, index)

        if self.source_label_mapping:
            source_label = self.source_label_mapping[source.huggingface_dataset_path]
            return image, label, source_label
            
        return image, label
    
    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: Length of the dataset (minimum length between fake and real datasets, which  limits the number of
            images sampled each epoch to the length of the smallest dataset to avoid imbalance).
        """
        real_dataset_min = min([len(ds) for ds in self.real_image_datasets])
        fake_dataset_min = min([len(ds) for ds in self.fake_image_datasets])
        return min(fake_dataset_min, real_dataset_min)

    def reset(self):
        self._history = {
            'source': [],
            'index': [],
            'label': [],
        }