import torch
import numpy as np
from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download
from base_miner.NPR.networks.resnet import resnet50
from bitmind.utils.image_transforms import get_base_transforms
from base_miner.deepfake_detectors import DeepfakeDetector
from base_miner import DETECTOR_REGISTRY
from base_miner.NPR.config.constants import WEIGHTS_DIR


@DETECTOR_REGISTRY.register_module(module_name='NPR')
class NPRDetector(DeepfakeDetector):
    """
    DeepfakeDetector subclass that initializes a pretrained NPR model
    for binary classification of fake and real images.
    
    Attributes:
        model_name (str): Name of the detector instance.
        config (str): Name of the YAML file in deepfake_detectors/config/ to load
                      attributes from.
        device (str): The type of device ('cpu' or 'cuda').
    """
    
    def __init__(self, model_name: str = 'NPR', config: str = 'npr.yaml', device: str = 'cpu'):
        super().__init__(model_name, config, device)
        self.transforms = get_base_transforms()

    def load_model(self):
        """
        Load the ResNet50 model with the specified weights for deepfake detection.
        """
        self.ensure_weights_are_available(WEIGHTS_DIR, self.weights)
        self.model = resnet50(num_classes=1)
        self.model.load_state_dict(torch.load(Path(WEIGHTS_DIR) / self.weights, map_location=self.device))
        self.model.eval()
    
    def preprocess(self, image: Image) -> torch.Tensor:
        """
        Preprocess the image using the base_transforms function.
        
        Args:
            image (PIL.Image): The image to preprocess.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        image_tensor = self.transforms(image).unsqueeze(0).float()
        return image_tensor

    def __call__(self, image: Image) -> float:
        """
        Perform inference with the model.

        Args:
            image (PIL.Image): The image to process.

        Returns:
            float: The prediction score indicating the likelihood of the image being a deepfake.
        """
        image_tensor = self.preprocess(image)
        with torch.no_grad():
            out = np.asarray(self.model(image_tensor).sigmoid().flatten())
        return out
