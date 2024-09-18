import typing
from abc import ABC, abstractmethod
from pathlib import Path
import yaml
import torch
from PIL import Image


class DeepfakeDetector(ABC):
    def __init__(self, model_name: str, config = None, cuda: bool = False):
        self.model_name = model_name
        use_cuda = cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if config: self.load_and_apply_config(config)
        self.load_model()

    @abstractmethod
    def load_model(self):
        """
        Load the model. Specific loading implementations will be defined in subclasses.
        """
        pass

    def preprocess(self, image: Image) -> torch.Tensor:
        """
        Preprocess the image for model inference.
        
        Args:
            image (PIL.Image): The image to preprocess.
            extra_data (dict, optional): Any additional data required for preprocessing.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        # General preprocessing, to be overridden if necessary in subclasses
        pass

    @abstractmethod
    def __call__(self, image: Image) -> float:
        """
        Perform inference with the model.

        Args:
            image (PIL.Image): The preprocessed image.

        Returns:
            float: The model's prediction (or other relevant result).
        """

    def load_and_apply_config(self, detector_config):
        """
        Load detector configuration from YAML file and set corresponding attributes dynamically.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        detector_config_file = Path(__file__).resolve().parent / Path('configs/' + detector_config)
        try:
            with open(detector_config_file, 'r') as file:
                config_dict = yaml.safe_load(file)

            # Set class attributes dynamically from the config dictionary
            for key, value in config_dict.items():
                setattr(self, key, value)  # Dynamically create self.key = value
            
        except Exception as e:
            print(f"Error loading detector configurations from {detector_config_file}: {e}")
            raise