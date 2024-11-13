from huggingface_hub import hf_hub_download
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
import typing
import torch
import yaml

from base_miner.UCF.config.constants import CONFIGS_DIR, WEIGHTS_DIR


class DeepfakeDetector(ABC):
    """
    Abstract base class for detecting deepfake images via binary classification.

    This class is intended to be subclassed by detector implementations
    using different underying model architectures, routing via gates, or 
    configurations.
    
    Attributes:
        model_name (str): Name of the detector instance.
        config (str): Name of the YAML file in deepfake_detectors/config/ to load
                      instance attributes from.
        device (str): The type of device ('cpu' or 'cuda').
    """
    
    def __init__(self, model_name: str, config = None, device: str = 'cpu'):
        self.model_name = model_name
        self.device = torch.device(device if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        if config:
            self.load_and_apply_config(config)
            self.load_train_config()
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
        if Path(detector_config).exists():
            detector_config_file = Path(detector_config)
        else:
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

    def ensure_weights_are_available(self, weights_dir, weights_filename):
        """
        
        """
        destination_path = Path(weights_dir) / Path(weights_filename)
        if not Path(weights_dir).exists():
            Path(weights_dir).mkdir(parents=True, exist_ok=True)
        if not destination_path.exists():
            hf_hub_download(self.hf_repo, weights_filename, cache_dir=weights_dir)

    def load_train_config(self):
        destination_path = Path(CONFIGS_DIR) / Path(self.train_config)
        if destination_path.exists():
            print(f"Loaded local config from {destination_path}")
            with destination_path.open('r') as f:
                config = yaml.safe_load(f)
        else:
            local_config_path = hf_hub_download(self.hf_repo, self.train_config, cache_dir=CONFIGS_DIR)
            print(f"Downloaded {self.hf_repo}/{self.train_config} to {local_config_path}")
            with local_config_path.open('r') as f:
                config = yaml.safe_load(f)
        self.config = config
        return config