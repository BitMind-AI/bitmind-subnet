from PIL import Image
from abc import ABC, abstractmethod
import numpy as np


class Gate(ABC):
    """
    Abstract base class for image content detection and preprocessing.
    Used to route deepfake detection inference inputs to tailored models
    in a single agent or mixture-of-experts design.

    This class is intended to be subclassed by specific gate
    implementations that handle different content types.

    Attributes:
        gate_name (str): The name of the gate.
        content_type (str): The type of content handled by the gate.
    """
    
    def __init__(self, gate_name: str, content_type: str):
        self.gate_name = gate_name
        self.content_type = content_type

    @abstractmethod
    def preprocess(self, image: np.array) -> any:
        """Preprocess the image based on its content type."""
        return image

    @abstractmethod
    def __call__(self, image: Image) -> any:
        """
        Perform content classification and content-specific preprocessing.
        Used to route inputs to appropriate models for inference.

        Args:
            image (PIL.Image): The image to preprocess.
        """
        pass