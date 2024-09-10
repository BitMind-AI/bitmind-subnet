import torch
from PIL import Image
import typing
from abc import ABC, abstractmethod

class DeepfakeDetector(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
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
        pass