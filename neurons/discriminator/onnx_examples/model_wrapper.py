import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple


class BaseModelWrapper(nn.Module, ABC):
    """    
    This class servers as an example interface for wrapping a pytorch model
    to preprocess its inputs before inference. This allows you to pre-package
    your model with all the required transforms.
    """
    
    def __init__(self, model: nn.Module, target_size: Tuple[int, int] = (224, 224)):
        super().__init__()
        self.model = model
        self.target_size = target_size
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input tensor and return model output.
        
        Args:
            x: Input tensor with raw pixel values [0, 255]
            
        Returns:
            Model output tensor
        """
        pass
    
    def normalize_to_01(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw pixel values [0, 255] to [0, 1] range."""
        return x.float() / 255.0
    
    def apply_imagenet_normalization(self, x: torch.Tensor, channels_dim: int = 1) -> torch.Tensor:
        """Apply ImageNet normalization to tensor."""
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        # Create view for broadcasting
        view_shape = [1] * x.dim()
        view_shape[channels_dim] = 3
        mean = mean.view(view_shape)
        std = std.view(view_shape)
        
        return (x - mean) / std


class ImageModelWrapper(BaseModelWrapper):
    """
    Preprocessing wrapper for image classification models.
    
    This is an example implementation. Miners should rewrite this function to preprocess data 
    according to their model's needs.
    """

    
    def __init__(self, model: nn.Module, crop_inputs: bool = True, 
                 resize_inputs: bool = True, target_size: Tuple[int, int] = (224, 224)):
        super().__init__(model, target_size)
        self.crop_inputs = crop_inputs
        self.resize_inputs = resize_inputs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process image input tensor.
        
        This is an example implementation. Miners should rewrite this function to preprocess data 
        according to their model's needs.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width] with values [0, 255]
            
        Returns:
            Model output tensor
        """
        # Example: Normalize pixel values from [0, 255] to [0, 1]
        x = self.normalize_to_01(x)
        
        # Example: Crop to square if requested
        if self.crop_inputs:
            _, _, height, width = x.shape  # B, C, H, W
            min_dim = min(height, width)
            y_start = (height - min_dim) // 2
            x_start = (width - min_dim) // 2
            y_end = y_start + min_dim
            x_end = x_start + min_dim
            x = x[:, :, y_start:y_end, x_start:x_end]
        
        # Example: Resize to target size
        if self.resize_inputs:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        # Example: Apply normalization
        x = self.apply_imagenet_normalization(x)
        
        # Apply your model
        return self.model(x)


class VideoModelWrapper(BaseModelWrapper):
    """
    Preprocessing wrapper for video classification models.
    
    This is an example implementation. Miners should rewrite this function to preprocess data 
    according to their model's needs.
    """
    
    def __init__(self, model: nn.Module, crop_inputs: bool = True, 
                 resize_inputs: bool = True, target_size: Tuple[int, int] = (224, 224),
                 num_frames: int = 16, channels_first: bool = True, num_classes: int = 3):
        super().__init__(model, target_size)
        self.crop_inputs = crop_inputs
        self.resize_inputs = resize_inputs
        self.num_frames = num_frames
        self.channels_first = channels_first  # True for [B,C,F,H,W], False for [B,F,C,H,W]
        self.num_classes = num_classes
        
        # Update model classifier if needed
        if hasattr(model, 'classifier') and hasattr(model.config, 'hidden_size'):
            model.classifier = nn.Linear(model.config.hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process video input tensor.
        
        This is an example implementation. Miners should rewrite this function to preprocess data 
        according to their model's needs.
        
        Args:
            x: Input tensor of shape [batch_size, frames, channels, height, width] with values [0, 255]
            
        Returns:
            Model output tensor
        """
        # Example: Normalize pixel values from [0, 255] to [0, 1]
        x = self.normalize_to_01(x)
        
        # Get dimensions for processing
        batch_size, frames, channels, height, width = x.shape
        
        # Example: Crop to square if requested
        if self.crop_inputs:
            min_dim = min(height, width)
            y_start = (height - min_dim) // 2
            x_start = (width - min_dim) // 2
            y_end = y_start + min_dim
            x_end = x_start + min_dim
            x = x[:, :, :, y_start:y_end, x_start:x_end]
            # Update height and width after cropping
            height, width = min_dim, min_dim
        
        # Example: Resize frames to target size
        if self.resize_inputs:
            # Process each frame individually to avoid dynamic reshaping issues
            resized_frames = []
            for i in range(frames):
                frame = x[:, i, :, :, :]  # [batch, channels, height, width]
                frame_resized = F.interpolate(frame, size=self.target_size, mode='bilinear', align_corners=False)
                resized_frames.append(frame_resized)
            
            # Stack frames back together
            x = torch.stack(resized_frames, dim=1)  # [batch, frames, channels, target_height, target_width]
        
        # Example: Apply normalization
        x = self.apply_imagenet_normalization(x, channels_dim=2)
        
        # Convert format based on model requirements
        if self.channels_first:
            # Convert to [batch, channels, frames, height, width] format for 3D convolutions
            x = x.permute(0, 2, 1, 3, 4)
        # else: keep as [batch, frames, channels, height, width] format
        
        # Apply your model
        output = self.model(x)
        
        # Handle HuggingFace model outputs (they return objects with logits attribute)
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            # Handle regular PyTorch model outputs
            logits = output

        # Example: Ensure output is properly shaped
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        return logits 