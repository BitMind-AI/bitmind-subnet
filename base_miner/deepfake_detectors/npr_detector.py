import torch
import numpy as np
from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download
from base_miner.NPR.networks.resnet import resnet50
from bitmind.image_transforms import base_transforms
from base_miner.deepfake_detectors import DeepfakeDetector
from base_miner import DETECTOR_REGISTRY
from base_miner.NPR.config.constants import WEIGHTS_DIR


@DETECTOR_REGISTRY.register_module(module_name='NPR')
class NPRDetector(DeepfakeDetector):
    def __init__(self, model_name: str = 'NPR', config: str = 'npr.yaml', cuda: bool = False):
        super().__init__(model_name, config, cuda)

    def load_model(self):
        """
        Load the ResNet50 model with the specified weights for deepfake detection.
        """
        self.ensure_weights_are_available(self.weights)
        self.model = resnet50(num_classes=1)
        print(f"Loading detector model from {Path(WEIGHTS_DIR) / self.weights}")
        self.model.load_state_dict(torch.load(Path(WEIGHTS_DIR) / self.weights, map_location=self.device))
        self.model.eval()

    def ensure_weights_are_available(self, weight_filename):
        destination_path = Path(WEIGHTS_DIR) / Path(weight_filename)
        if not destination_path.parent.exists():
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Created directory {destination_path.parent}.")
        if not destination_path.exists():
            model_path = hf_hub_download(self.hf_repo, weight_filename)
            model = torch.load(model_path)
            torch.save(model, destination_path)
            print(f"Downloaded {weight_filename} to {destination_path}.")
        else:
            print(f"{weight_filename} already present at {destination_path}.")
    
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
        print(out.shape)
        return out