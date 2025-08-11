"""
Example: Converting Custom PyTorch Image Models to ONNX
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_wrapper import ImageModelWrapper


# ============================================================================
# Custom Image Model
# ============================================================================

class CustomImageModel(nn.Module):
    """Example custom image classification model."""
    
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x





# ============================================================================
# Example Usage
# ============================================================================

def convert_custom_image_model():
    """Convert a custom image model to ONNX."""
    print("Converting custom image model...")
    
    os.makedirs("models", exist_ok=True)
    
    # Create model and preprocessing wrapper
    model = CustomImageModel()
    wrapped_model = ImageModelWrapper(model, crop_inputs=False, resize_inputs=True, target_size=(384, 384))
    wrapped_model.eval()
    
    # Create dummy input
    dummy_input = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.uint8)
    
    # Export to ONNX
    onnx_path = "models/image_detector.onnx"
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'logits': {0: 'batch_size'}
        },
        opset_version=11,
        do_constant_folding=True,
        export_params=True,
        keep_initializers_as_inputs=False
    )
    print(f"✅ Image detector converted: {onnx_path}")


def main():
    """Run the custom image model conversion example."""
    print("🚀 Custom PyTorch Image Model Conversion Example")
    print("=" * 50)
    
    # Convert custom image model
    convert_custom_image_model()
    
    print("\n" + "=" * 50)
    print("✅ Custom image model conversion completed!")
    print("\n💡 Key Points:")
    print("   • Custom image models inherit from nn.Module")
    print("   • ImageModelWrapper handles image preprocessing")
    print("   • Direct torch.onnx.export() for maximum flexibility")
    print("   • Easy to customize preprocessing and export parameters")
    
    print("\n📁 Generated ONNX files:")
    import os
    if os.path.exists("models"):
        for file in os.listdir("models"):
            if file.endswith(".onnx"):
                size_mb = os.path.getsize(os.path.join("models", file)) / (1024 * 1024)
                print(f"   • models/{file} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main() 