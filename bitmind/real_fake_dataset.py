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
        source_label_mapping=None,
        sampling_strategy='weighted'  # New parameter
    ):
        """
        Initialize the RealFakeDataset instance.

        Args:
            real_image_datasets (list): List of ImageDataset objects containing real images
            fake_image_datasets (list): List of ImageDataset objects containing fake images
            transforms (transforms.Compose): Image transformations (default: None).
            fake_prob (float): Probability of selecting a fake image (default: 0.5).
            source_label_mapping (dict): A dictionary mapping dataset names to float labels.
            sampling_strategy (str): Strategy for sampling. Options: 'weighted', 'balanced', 'full' (default: 'weighted').
        """
        self.real_image_datasets = real_image_datasets
        self.fake_image_datasets = fake_image_datasets
        self.transforms = transforms
        self.fake_prob = fake_prob
        self.source_label_mapping = source_label_mapping
        self.sampling_strategy = sampling_strategy

        self._history = {
            'source': [],
            'index': [],
            'label': [],
        }
        
        self._setup_sampling_weights()
        self._setup_sampling_strategy()

    def _setup_sampling_weights(self):
        """
        Set up sampling weights for each dataset based on its size.
        """
        self.real_weights = np.array([len(ds) for ds in self.real_image_datasets])
        self.fake_weights = np.array([len(ds) for ds in self.fake_image_datasets])
        
        self.real_weights = self.real_weights / self.real_weights.sum()
        self.fake_weights = self.fake_weights / self.fake_weights.sum()

    def _setup_sampling_strategy(self):
        """
        Set up the sampling strategy based on the chosen option.
        """
        if self.sampling_strategy == 'balanced':
            min_real = min(len(ds) for ds in self.real_image_datasets)
            min_fake = min(len(ds) for ds in self.fake_image_datasets)
            self.samples_per_dataset = min(min_real, min_fake)
        elif self.sampling_strategy == 'full':
            self.total_real = sum(len(ds) for ds in self.real_image_datasets)
            self.total_fake = sum(len(ds) for ds in self.fake_image_datasets)

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieve an item (image, label) from the dataset.
        Maintains balance between real and fake images based on fake_prob and sampling strategy.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the image, its label (1 : fake, 0 : real),
            and its source label (0 for real datasets and >= 1 for fake datasets).
        """
        if len(self._history['index']) > index:
            self.reset()

        if self.sampling_strategy == 'balanced':
            is_fake = index >= self.samples_per_dataset
        elif self.sampling_strategy == 'full':
            is_fake = index >= self.total_real
        else:  # weighted
            is_fake = np.random.rand() < self.fake_prob

        if is_fake:
            datasets = self.fake_image_datasets
            weights = self.fake_weights
            label = 1.0
        else:
            datasets = self.real_image_datasets
            weights = self.real_weights
            label = 0.0

        if self.sampling_strategy == 'balanced':
            source_index = index % len(datasets)
            source = datasets[source_index]
            valid_index = index % self.samples_per_dataset
        elif self.sampling_strategy == 'full':
            cumulative_sizes = np.cumsum([len(ds) for ds in datasets])
            source_index = np.searchsorted(cumulative_sizes, index % (cumulative_sizes[-1]))
            source = datasets[source_index]
            valid_index = index - (cumulative_sizes[source_index - 1] if source_index > 0 else 0)
        else:  # weighted
            source_index = np.random.choice(len(datasets), p=weights)
            source = datasets[source_index]
            valid_index = np.random.randint(0, len(source))

        if label == 1.0:  # fake image
            image = source[valid_index]['image']
        else:  # real image
            if self.sampling_strategy == 'weighted':
                imgs, idx = source.sample(1)
                image = imgs[0]['image']
                valid_index = idx[0]
            else:
                image = source[valid_index]['image']

        self._history['source'].append(source.huggingface_dataset_path)
        self._history['label'].append(label)
        self._history['index'].append(valid_index)
        
        try:
            if self.transforms is not None:
                image = self.transforms(image)
        except Exception as e:
            print(e)
            print(source.huggingface_dataset_path, valid_index)

        if self.source_label_mapping:
            source_label = self.source_label_mapping[source.huggingface_dataset_path]
            return image, label, source_label
            
        return image, label
    
    def __len__(self) -> int:
        """
        Return the length of the dataset based on the sampling strategy.

        Returns:
            int: Total length of the dataset.
        """
        if self.sampling_strategy == 'balanced':
            return self.samples_per_dataset * 2
        elif self.sampling_strategy == 'full':
            return self.total_real + self.total_fake
        else:  # weighted
            real_datasets_total = sum(len(ds) for ds in self.real_image_datasets)
            return int((real_datasets_total / (1 - self.fake_prob)) * 2)

    def reset(self):
        self._history = {
            'source': [],
            'index': [],
            'label': [],
        }
