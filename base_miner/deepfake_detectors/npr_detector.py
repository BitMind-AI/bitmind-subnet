import torch
import numpy as np
from PIL import Image
from base_miner.NPR.networks.resnet import resnet50
from bitmind.image_transforms import base_transforms
from base_miner.deepfake_detectors import DETECTOR_REGISTRY, DeepfakeDetector

@DETECTOR_REGISTRY.register_module(module_name='NPR')
class NPRDetector(DeepfakeDetector):
    def __init__(self, weight_path: str, model_name: str = 'NPR'):
        self.weight_path = weight_path
        super().__init__(model_name)

    def load_model(self):
        """
        Load the ResNet50 model with the specified weights for deepfake detection.
        """
        self.model = resnet50(num_classes=1)
        print(f"Loading detector model from {self.weight_path}")
        self.model.load_state_dict(torch.load(self.weight_path, map_location='cpu'))
        self.model.eval()

    def preprocess(self, image: Image) -> torch.Tensor:
        """
        Preprocess the image using the base_transforms function.
        
        Args:
            image (PIL.Image): The image to preprocess.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        image_tensor = base_transforms(image).unsqueeze(0).float()
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