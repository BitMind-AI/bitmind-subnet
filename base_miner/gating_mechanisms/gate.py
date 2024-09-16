from PIL import Image
from abc import ABC, abstractmethod

class Gate(ABC):
    def __init__(self, gate_name: str, content_type: str):
        self.gate_name = gate_name
        self.content_type = content_type
    
    @abstractmethod
    def detect_content_type(self, image: Image) -> any:
        """Detect the content type of the image."""
        pass

    @abstractmethod
    def preprocess(self, image: Image) -> any:
        """Preprocess the image based on its content type."""
        return image

    @abstractmethod
    def __call__(self, image: Image) -> any:
        """
        Perform inference with the model.

        Args:
            image (PIL.Image): The preprocessed image.

        Returns:
            float: The model's prediction (or other relevant result).
        """
        pass