from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import yaml
import bittensor as bt
from PIL import Image
from huggingface_hub import hf_hub_download

from base_miner.DFB.config.constants import CONFIGS_DIR, WEIGHTS_DIR


class DeepfakeDetector(ABC):
    """Abstract base class for detecting deepfake images via binary classification.

    This class is intended to be subclassed by detector implementations
    using different underlying model architectures, routing via gates, or
    configurations.

    Attributes:
        model_name (str): Name of the detector instance.
        config (Optional[str]): Name of the YAML file in deepfake_detectors/config/
            to load instance attributes from.
        device (str): The type of device ('cpu' or 'cuda').
        hf_repo (str): Hugging Face repository name for model weights.
        train_config (str): Name of training configuration file.
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[str] = None,
        device: str = 'cpu'
    ) -> None:
        """Initialize the DeepfakeDetector.

        Args:
            model_name: Name of the detector instance.
            config: Optional name of configuration file to load.
            device: Device to run the model on ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.device = torch.device(
            device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        )

        if config:
            self.set_class_attrs(config)
            self.load_model_config()

        self.load_model()

    @abstractmethod
    def load_model(self) -> None:
        """Load the model weights and architecture.

        This method should be implemented by subclasses to define their specific
        model loading logic.
        """
        pass

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess the image for model inference.

        Args:
            image: The input image to preprocess.

        Returns:
            The preprocessed image as a tensor ready for model input.
        """
        # General preprocessing, to be overridden if necessary in subclasses
        pass

    @abstractmethod
    def __call__(self, image: Image.Image) -> float:
        """Perform inference with the model.

        Args:
            image: The preprocessed input image.

        Returns:
            The model's prediction score (typically between 0 and 1).
        """
        pass

    def set_class_attrs(self, detector_config: str) -> None:
        """Load detector configuration from YAML file and set attributes.

        Args:
            detector_config: Path to the YAML configuration file or filename
                in the configs directory.

        Raises:
            Exception: If there is an error loading or parsing the config file.
        """
        if Path(detector_config).exists():
            detector_config_file = Path(detector_config)
        else:
            detector_config_file = (
                Path(__file__).resolve().parent / Path('configs/' + detector_config)
            )

        try:
            with open(detector_config_file, 'r', encoding='utf-8') as file:
                config_dict = yaml.safe_load(file)

            # Set class attributes dynamically from the config dictionary
            for key, value in config_dict.items():
                setattr(self, key, value)

        except Exception as e:
            print(f"Error loading detector configurations from {detector_config_file}: {e}")
            raise

    def ensure_weights_are_available(
        self,
        weights_dir: str,
        weights_filename: str
    ) -> None:
        """Ensure model weights are downloaded and available locally.

        Downloads weights from Hugging Face Hub if not found locally.

        Args:
            weights_dir: Directory to store/find the weights.
            weights_filename: Name of the weights file.
        """
        destination_path = Path(weights_dir) / Path(weights_filename)
        if not Path(weights_dir).exists():
            Path(weights_dir).mkdir(parents=True, exist_ok=True)

        if not destination_path.exists():
            print(f"Downloading {weights_filename} from {self.hf_repo} "
                  f"to {weights_dir}")
            hf_hub_download(self.hf_repo, weights_filename, local_dir=weights_dir)

    def load_model_config(self):
        try:
            destination_path = Path(CONFIGS_DIR) / Path(self.train_config)
            if not destination_path.exists():
                local_config_path = hf_hub_download(self.hf_repo, self.train_config, local_dir=CONFIGS_DIR)
                print(f"Downloaded {self.hf_repo}/{self.train_config} to {local_config_path}")
                with Path(local_config_path).open('r') as f:
                    self.config = yaml.safe_load(f)
            else:
                print(f"Loaded local config from {destination_path}")
                with destination_path.open('r') as f:
                    self.config = yaml.safe_load(f)
        except Exception as e:
            # some models such as NPR don't have an additional config file
            bt.logging.warning("No additional train config loaded.")