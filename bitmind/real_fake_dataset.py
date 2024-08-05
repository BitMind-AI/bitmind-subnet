import numpy as np


class RealFakeDataset:

    def __init__(
        self,
        real_image_datasets: list,
        fake_image_datasets: list,
        transforms=None,
        fake_prob=0.5,
    ):
        """
        Initialize the RealFakeDataset instance.

        Args:
            real_image_datasets (list): List of ImageDataset objects containing real images
            fake_image_datasets (list): List of ImageDataset objects containing real images
            transforms (transforms.Compose): Image transformations (default: None).
            fake_prob (float): Probability of selecting a fake image (default: 0.5).
        """
        self.real_image_datasets = real_image_datasets
        self.fake_image_datasets = fake_image_datasets
        self.transforms = transforms
        self.fake_prob = fake_prob

        self._history = {}

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieve an item (image, label) from the dataset.
        By default, 50/50 chance of real or fake. This can be overidden by self.fake_prob

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the image and its label.
        """
        if index in self._history:
            label = self._history[index]['label']
            datasets = self.real_image_datasets if label == 0 else self.fake_image_datasets
            for ds in datasets:
                if ds.huggingface_dataset_path == self._history[index]['source']:
                    source = ds
            # Update index to account for potential download failures in previous epochs
            # if a download failed, the current index maps to a working index in the
            # current dataset
            index = self._history[index]['sampled_index']
        else:
            if np.random.rand() > self.fake_prob:
                datasets = self.fake_image_datasets
                label = 1
            else:
                datasets = self.real_image_datasets
                label = 0
            source = datasets[np.random.randint(0, len(datasets))]
            self._history[index] = {
                'label': label,
                'source': source.huggingface_dataset_path
            }

        try:
            # this image index may have been updated at the beginning of this function to
            # account for failed downloads for url datasets. If this is the first time
            # accessing this index for a url dataset, a download will be attempted before
            # falling back to a random sample. The index that gave a successful download will
            # be tracked in self._history for access in subsequent epochs
            image = source[index]['image']
            self._history[index]['sampled_index'] = index
        except Exception as e:
            print(e)
            print(f"Error sampling image index {index} from dataset {source}, performing random sample instead")
            imgs, idx = source.sample(1)
            image = imgs[0]['image']
            self._history[index]['sampled_index'] = idx[0]

        try:
            if self.transforms is not None:
                image = self.transforms(image)
        except Exception as e:
            print(e)
            print("RealFakeDataset: Error transforming image", source.huggingface_dataset_path, index)

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
