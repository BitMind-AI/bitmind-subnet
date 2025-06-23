import json
import torch
import bittensor as bt
import numpy as np
from torchvision import models
import torchvision.transforms.functional as F


class Segmenter:
    """Handler for image segmentation models."""
    
    def __init__(self, config):
        self.config = config
        self.segmentation_model = None
        self.device = (
            self.config.device
            if hasattr(self.config, "device")
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.load_model()

    def load_model(self):
        """Load the segmentation model.

        MINER TODO:
            This class has a placeholder model to demonstrate the required outputs
            for validator requests. It has not been trained and will perform
            poorly. Your task is to train a performant image segmentation
            model and load it here. Happy mining!
        """
        bt.logging.info("Loading segmentation model...")
        
        ### REPLACE WITH YOUR OWN MODEL
        # Using DeepLabV3 with ResNet-50 backbone as placeholder
        self.segmentation_model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.segmentation_model = self.segmentation_model.to(self.device)
        self.segmentation_model.eval()

    def preprocess(self, image_tensor):
        """Preprocess image tensor for segmentation.

        MINER TODO:
            Update this function as necessary for your model

        Args:
            image_tensor (torch.Tensor): Input image tensor of shape (C, H, W)
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for model input
        """
        bt.logging.debug(
            json.dumps(
                {
                    "modality": "image_segmentation",
                    "shape": tuple(image_tensor.shape),
                    "dtype": str(image_tensor.dtype),
                    "min": torch.min(image_tensor).item(),
                    "max": torch.max(image_tensor).item(),
                },
                indent=2,
            )
        )

        # Ensure tensor is float and normalized to [0, 1]
        if image_tensor.dtype != torch.float32:
            image_tensor = image_tensor.float()
        
        if torch.max(image_tensor) > 1.0:
            image_tensor = image_tensor / 255.0

        # Add batch dimension and move to device
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Normalize using ImageNet stats (standard for pretrained models)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor

    def postprocess(self, model_output, original_shape):
        """Convert model output to segmentation confidence mask.
        
        Args:
            model_output: Raw model output
            original_shape: Original image shape (H, W)
            
        Returns:
            np.ndarray: Segmentation confidence mask of shape (H, W) with values in [0,1]
        """
        predictions = model_output['out']
        predictions = torch.softmax(predictions, dim=1)
        confidence = predictions[0, 1].cpu().numpy()
        
        # Ensure returned mask is the same dims as image sent by validator
        if confidence.shape != original_shape:
            from PIL import Image
            confidence_pil = Image.fromarray((confidence * 255).astype(np.uint8))
            confidence_pil = confidence_pil.resize((original_shape[1], original_shape[0]), Image.BILINEAR)
            confidence = np.array(confidence_pil).astype(np.float32) / 255
        
        return confidence

    def segment(self, image_tensor):
        """Perform image segmentation.

        MINER TODO: Update segmentation logic as necessary for your own model

        Args:
            image_tensor (torch.Tensor): Input image tensor of shape (C, H, W)

        Returns:
            np.ndarray: Segmentation mask of shape (H, W) with class indices
        """
        if self.segmentation_model is None:
            self.load_model()

        original_shape = image_tensor.shape[1:]  # (H, W)
        processed_tensor = self.preprocess(image_tensor)
        
        bt.logging.debug(
            f"Running segmentation on array shape {processed_tensor.shape}"
        )

        # MINER TODO: Update segmentation logic as necessary
        with torch.no_grad():
            output = self.segmentation_model(processed_tensor)
        heatmap = self.postprocess(output, original_shape)
        
        bt.logging.success(f"Segmentation completed, heatmap shape: {heatmap.shape}")
        bt.logging.success(f"Heatmap range: [{np.min(heatmap)}, {np.max(heatmap)}]")
        
        return heatmap
