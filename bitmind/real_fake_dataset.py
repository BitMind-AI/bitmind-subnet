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
        sampling_strategy='balanced',
        max_samples_per_dataset=20000,
        split='train'
    ):
        """d
        Initialize the RealFakeDataset instance.

        Args:
            real_image_datasets (list): List of ImageDataset objects containing real images
            fake_image_datasets (list): List of ImageDataset objects containing fake images
            transforms (transforms.Compose): Image transformations (default: None).
            fake_prob (float): Probability of selecting a fake image (default: 0.5).
            source_label_mapping (dict): A dictionary mapping dataset names to float labels.
            sampling_strategy (str): Strategy for sampling. Options: 'weighted', 'balanced', 'full', 'max_samples' (default: 'weighted').
            max_samples_per_dataset (int): Maximum number of samples to use from each dataset when using 'max_samples' strategy.
        """
        self.real_image_datasets = real_image_datasets
        self.fake_image_datasets = fake_image_datasets
        self.transforms = transforms
        self.fake_prob = fake_prob
        self.source_label_mapping = source_label_mapping
        self.sampling_strategy = sampling_strategy
        self.max_samples_per_dataset = max_samples_per_dataset
        self.split = split
        
        self._history = {
            'source': [],
            'index': [],
            'label': [],
        }
        
        self._setup_sampling_weights()
        self._setup_sampling_strategy()
        self._setup_shuffled_indices()
        self._print_dataset_info()
        self._verify_label_distribution()

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
            real_sizes = [len(ds) for ds in self.real_image_datasets]
            fake_sizes = [len(ds) for ds in self.fake_image_datasets]
            self.samples_per_dataset = min(min(real_sizes), min(fake_sizes))
            self.total_samples = self.samples_per_dataset * (len(self.real_image_datasets) + len(self.fake_image_datasets))
        elif self.sampling_strategy == 'full':
            self.total_real = sum(len(ds) for ds in self.real_image_datasets)
            self.total_fake = sum(len(ds) for ds in self.fake_image_datasets)
        elif self.sampling_strategy == 'max_samples':
            if self.max_samples_per_dataset is None:
                raise ValueError("max_samples_per_dataset must be set for 'max_samples' strategy")
            else:
                self.real_samples = [min(len(ds), self.max_samples_per_dataset) for ds in self.real_image_datasets]
                self.fake_samples = [min(len(ds), self.max_samples_per_dataset) for ds in self.fake_image_datasets]
                self.total_real = sum(self.real_samples)
                self.total_fake = sum(self.fake_samples)
                self.real_cumulative = np.cumsum([0] + self.real_samples[:-1])
                self.fake_cumulative = np.cumsum([0] + self.fake_samples[:-1])
                self.real_shuffled_indices = [np.random.permutation(samples) for samples in self.real_samples]
                self.fake_shuffled_indices = [np.random.permutation(samples) for samples in self.fake_samples]
    
    def _print_dataset_info(self):
        print(f"\nDataset information:")
        print(f"Total samples: {len(self)}")
        print("Real datasets:")
        for i, ds in enumerate(self.real_image_datasets):
            if self.sampling_strategy == 'max_samples':
                size = min(len(ds), self.max_samples_per_dataset)
            elif self.sampling_strategy == 'balanced':
                size = self.samples_per_dataset
            else:
                size = len(ds)
            print(f"  {i}: {ds.huggingface_dataset_path} - Size: {size}")
        print("Fake datasets:")
        for i, ds in enumerate(self.fake_image_datasets):
            if self.sampling_strategy == 'max_samples':
                size = min(len(ds), self.max_samples_per_dataset)
            elif self.sampling_strategy == 'balanced':
                size = self.samples_per_dataset
            else:
                size = len(ds)
            print(f"  {i}: {ds.huggingface_dataset_path} - Size: {size}")
            
    def _verify_label_distribution(self):
        """Verify the distribution of labels in the dataset."""
        total = len(self)
        
        if self.sampling_strategy == 'balanced':
            real_count = self.samples_per_dataset * len(self.real_image_datasets)
            fake_count = self.samples_per_dataset * len(self.fake_image_datasets)
        elif self.sampling_strategy == 'full':
            real_count = self.total_real
            fake_count = self.total_fake
        elif self.sampling_strategy == 'max_samples':
            real_count = sum(self.real_samples)
            fake_count = sum(self.fake_samples)
        elif self.sampling_strategy == 'weighted':
            real_count = int(total * (1 - self.fake_prob))
            fake_count = int(total * self.fake_prob)
        
        print(f"Label distribution in {self.split} set:")
        print(f"Real images: {real_count} ({real_count/total:.2%})")
        print(f"Fake images: {fake_count} ({fake_count/total:.2%})")
        if real_count == 0 or fake_count == 0:
            print("WARNING: One class is missing from the dataset!")
    
    def _setup_shuffled_indices(self):
        """
        Set up a shuffled list of indices for accessing samples.
        """
        self.shuffled_indices = np.arange(len(self))
        np.random.shuffle(self.shuffled_indices)

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieve an item (image, label) from the dataset using shuffled indices.
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self)}")

        if len(self._history['index']) > len(self):
            self.reset()

        shuffled_index = self.shuffled_indices[index]
        is_fake = self._determine_fake(shuffled_index)
        datasets, weights, label = self._get_dataset_info(is_fake)
        
        source, valid_index = self._get_source_and_index(shuffled_index, datasets, weights, is_fake)
        
        assert 0 <= valid_index < len(source), f"Invalid index: {valid_index} for source of length {len(source)}"
        
        image = self._get_image(source, valid_index, label)
        
        self._update_history(source, label, valid_index)
        
        if self.transforms:
            image = self._apply_transforms(image, source, valid_index)
        
        return self._prepare_output(image, label, source)

    def _setup_shuffled_indices(self):
        """Create and shuffle indices to ensure balanced sampling."""
        self.shuffled_indices = torch.randperm(len(self)).tolist()

    def _determine_fake(self, shuffled_index):
        if self.sampling_strategy == 'balanced':
            return shuffled_index >= self.samples_per_dataset * len(self.real_image_datasets)
        elif self.sampling_strategy == 'full':
            return shuffled_index >= self.total_real
        elif self.sampling_strategy == 'weighted':
            return np.random.rand() < self.fake_prob
        elif self.sampling_strategy == 'max_samples':
            return shuffled_index >= self.total_real

    def _get_dataset_info(self, is_fake):
        if is_fake:
            return self.fake_image_datasets, self.fake_weights, 1.0
        else:
            return self.real_image_datasets, self.real_weights, 0.0

    def _get_source_and_index(self, shuffled_index, datasets, weights, is_fake):
        if self.sampling_strategy == 'balanced':
            return self._balanced_sampling(shuffled_index, datasets)
        elif self.sampling_strategy == 'full':
            return self._full_sampling(shuffled_index, datasets)
        elif self.sampling_strategy == 'weighted':
            return self._weighted_sampling(datasets, weights)
        elif self.sampling_strategy == 'max_samples':
            return self._max_samples_sampling(shuffled_index, datasets, is_fake)

    def _balanced_sampling(self, shuffled_index, datasets):
        source_index = shuffled_index % len(datasets)
        source = datasets[source_index]
        valid_index = shuffled_index % self.samples_per_dataset
        return source, valid_index

    def _full_sampling(self, shuffled_index, datasets):
        cumulative_sizes = np.cumsum([len(ds) for ds in datasets])
        source_index = np.searchsorted(cumulative_sizes, shuffled_index % (cumulative_sizes[-1]))
        source = datasets[source_index]
        valid_index = shuffled_index - (cumulative_sizes[source_index - 1] if source_index > 0 else 0)
        return source, valid_index

    def _weighted_sampling(self, datasets, weights):
        source_index = np.random.choice(len(datasets), p=weights)
        source = datasets[source_index]
        valid_index = np.random.randint(0, len(source))
        return source, valid_index

    def _max_samples_sampling(self, shuffled_index, datasets, is_fake):
        samples = self.fake_samples if is_fake else self.real_samples
        cumulative = self.fake_cumulative if is_fake else self.real_cumulative
        shuffled_indices = self.fake_shuffled_indices if is_fake else self.real_shuffled_indices
        
        source_index = np.searchsorted(cumulative, shuffled_index, side='right') - 1
        source_index = max(0, min(source_index, len(datasets) - 1))  # Ensure source_index is within bounds
        source = datasets[source_index]
        
        # Calculate valid_index relative to the current dataset
        if source_index > 0:
            relative_index = shuffled_index - cumulative[source_index]
        else:
            relative_index = shuffled_index
        
        relative_index = relative_index % samples[source_index]
        # Use the pre-shuffled indices for this dataset
        valid_index = shuffled_indices[source_index][relative_index]
        
        return source, valid_index

    def _get_image(self, source, valid_index, label):
        if label == 1.0:  # fake image
            return source[valid_index]['image']
        else:  # real image
            if self.sampling_strategy == 'weighted':
                imgs, idx = source.sample(1)
                return imgs[0]['image']
            else:
                return source[valid_index]['image']

    def _update_history(self, source, label, valid_index):
        self._history['source'].append(source.huggingface_dataset_path)
        self._history['label'].append(label)
        self._history['index'].append(valid_index)

    def _apply_transforms(self, image, source, valid_index):
        try:
            if self.transforms is not None:
                return self.transforms(image)
        except Exception as e:
            print(e)
            print(source.huggingface_dataset_path, valid_index)
        return image

    def _prepare_output(self, image, label, source):
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
            return self.total_samples
        elif self.sampling_strategy == 'full':
            return self.total_real + self.total_fake
        elif self.sampling_strategy == 'weighted':
            real_datasets_total = sum(len(ds) for ds in self.real_image_datasets)
            return int((real_datasets_total / (1 - self.fake_prob)) * 2)
        elif self.sampling_strategy == 'max_samples':
            return self.total_real + self.total_fake

    def reset(self):
        self._history = {
            'source': [],
            'index': [],
            'label': [],
        }
        self._setup_shuffled_indices()  # Reshuffle indices on reset

    def on_epoch_end(self):
        """
        Method to be called at the end of each epoch to reshuffle the dataset.
        """
        self._setup_shuffled_indices()
        if self.sampling_strategy == 'max_samples':
            # Reshuffle indices for each dataset
            self.real_shuffled_indices = [np.random.permutation(samples) for samples in self.real_samples]
            self.fake_shuffled_indices = [np.random.permutation(samples) for samples in self.fake_samples]
