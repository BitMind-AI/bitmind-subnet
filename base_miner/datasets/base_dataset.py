from abc import ABC, abstractmethod
from datasets import Dataset
from typing import Optional
from torchvision.transforms import Compose

from bitmind.download_data import load_huggingface_dataset


class BaseDataset(ABC):
    def __init__(
        self,
        huggingface_dataset_path: Optional[str] = None,
        huggingface_dataset_split: str = 'train',
        huggingface_dataset_name: Optional[str] = None,
        huggingface_dataset: Optional[Dataset] = None,
        download_mode: Optional[str] = None,        
        transforms: Optional[Compose] = None
    ):
        """Base class for dataset implementations.
        
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
        self.huggingface_dataset_path = None
        self.huggingface_dataset_split = huggingface_dataset_split
        self.huggingface_dataset_name = None
        self.dataset = None
        self.transforms = transforms
        
        if huggingface_dataset_path is None and huggingface_dataset is None:
            raise ValueError("Either huggingface_dataset_path or huggingface_dataset must be provided.")
        
        # If a dataset is directly provided, use it
        if huggingface_dataset is not None:
            self.dataset = huggingface_dataset
            self.huggingface_dataset_path = self.dataset.info.dataset_name
            self.huggingface_dataset_name = self.dataset.info.config_name
            try:
                self.huggingface_dataset_split = list(self.dataset.info.splits.keys())[0]
            except AttributeError as e:
                self.huggingface_data_split = None

        else:
            # Store the initialization parameters
            self.huggingface_dataset_path = huggingface_dataset_path
            self.huggingface_dataset_name = huggingface_dataset_name
            self.dataset = load_huggingface_dataset(
                huggingface_dataset_path,
                huggingface_dataset_split,
                huggingface_dataset_name,
                download_mode)

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Get an item from the dataset.
        
        Args:
            index (int): Index of the item to retrieve.
            
        Returns:
            dict: Dictionary containing the item data.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        pass
