"""
Example: Converting a Custom PyTorch Video Model to ONNX
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_wrapper import VideoModelWrapper


# ============================================================================
# Custom Video Model
# ============================================================================

class CustomVideoModel(nn.Module):
    """A custom video classification model."""
    
    def __init__(self, num_classes=3):
        super().__init__()
        # Simple 3D CNN architecture
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, (3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, (3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, (3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
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

def convert_custom_video_model():
    """Convert a custom video model to ONNX."""
    print("Converting custom video model...")
    
    # Ensure models directory exists
    import os
    os.makedirs("models", exist_ok=True)
    
    # Create model and preprocessing wrapper
    model = CustomVideoModel()
    wrapped_model = VideoModelWrapper(model, crop_inputs=True, resize_inputs=True, target_size=(224, 224), num_frames=32)
    wrapped_model.eval()
    
    # Create dummy input
    dummy_input = torch.randint(0, 256, (1, 32, 3, 224, 224), dtype=torch.uint8)
    
    # Export to ONNX
    onnx_path = "models/video_detector.onnx"
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'frames', 3: 'height', 4: 'width'},
            'logits': {0: 'batch_size'}
        },
        opset_version=11,
        do_constant_folding=True,
        export_params=True,
        keep_initializers_as_inputs=False
    )
    print(f"✅ Video detector converted: {onnx_path}")


def main():
    """Run the custom video model conversion example."""
    print("🚀 Custom PyTorch Video Model Conversion Example")
    print("=" * 50)
    
    # Convert custom video model
    convert_custom_video_model()
    
    print("\n" + "=" * 50)
    print("✅ Custom video model conversion completed!")
    print("\n💡 Key Points:")
    print("   • Custom video models inherit from nn.Module")
    print("   • VideoModelWrapper handles video preprocessing")
    print("   • Direct torch.onnx.export() for maximum flexibility")
    print("   • Easy to customize preprocessing and export parameters")
    print("   • Supports different frame counts and temporal processing")
    
    print("\n📁 Generated ONNX files:")
    import os
    if os.path.exists("models"):
        for file in os.listdir("models"):
            if file.endswith(".onnx"):
                size_mb = os.path.getsize(os.path.join("models", file)) / (1024 * 1024)
                print(f"   • models/{file} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main() 