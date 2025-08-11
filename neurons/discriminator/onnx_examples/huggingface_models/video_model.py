"""
Example: Converting HuggingFace Video Models to ONNX
"""

import os
import time
import torch
from typing import Optional, Tuple

from transformers import AutoModelForVideoClassification

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_wrapper import VideoModelWrapper


def convert_huggingface_video_model(model_name: str, output_path: str,
                                   crop_inputs: bool = True, resize_inputs: bool = True,
                                   target_size: Tuple[int, int] = (224, 224),
                                   num_frames: int = 16, channels_first: bool = False,
                                   num_classes: int = 3) -> str:
    """
    Convert a HuggingFace video model to ONNX format.
    
    Args:
        model_name: HuggingFace model name (e.g., "microsoft/videomae-base")
        output_path: Output path for ONNX file
        crop_inputs: Whether to crop inputs to square
        resize_inputs: Whether to resize inputs to target size
        target_size: Target size for resizing (width, height)
        num_frames: Number of frames to process
        channels_first: Whether to use [B,C,F,H,W] format (True) or [B,F,C,H,W] format (False)
        num_classes: Number of output classes
        
    Returns:
        Path to the generated ONNX file
    """
    print(f"Converting HuggingFace video model: {model_name}")
    print(f"Configuration: {num_classes} classes, {num_frames} frames, {target_size} size")
    start_time = time.time()
    
    # Load the model
    model = AutoModelForVideoClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    # Ensure label mappings align with the requested number of classes
    id2label = {i: f"LABEL_{i}" for i in range(num_classes)}
    label2id = {v: k for k, v in id2label.items()}
    model.config.id2label = id2label
    model.config.label2id = label2id

    model.eval()
    
    # Create preprocessing wrapper with proper configuration
    wrapped_model = VideoModelWrapper(
        model, 
        crop_inputs=crop_inputs, 
        resize_inputs=resize_inputs, 
        target_size=target_size,
        num_frames=num_frames,
        channels_first=channels_first,
        num_classes=num_classes
    )
    wrapped_model.eval()
    
    # Create dummy input (raw pixel values [0, 255])
    # VideoMAE expects [B,F,C,H,W] format
    dummy_input = torch.randint(0, 256, (1, num_frames, 3, 224, 224), dtype=torch.uint8)
    
    # Test the model with dummy input to ensure it works
    print("Testing model with dummy input...")
    with torch.no_grad():
        test_output = wrapped_model(dummy_input)
        print(f"Test output shape: {test_output.shape}")
        print(f"Expected shape: [1, {num_classes}]")
    
    # Export to ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'frames', 2: 'channels', 3: 'height', 4: 'width'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
        keep_initializers_as_inputs=False
    )
    
    elapsed_time = time.time() - start_time
    print(f"✅ HuggingFace video model converted successfully in {elapsed_time:.2f} seconds")
    print(f"📁 ONNX file saved to: {output_path}")
    
    return output_path


def convert_videomae(output_path="models/video_detector.onnx", num_classes: int = 3):
    """Convert VideoMAE to ONNX."""
    print("Converting VideoMAE...")
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    try:
        onnx_path = convert_huggingface_video_model(
            model_name="MCG-NJU/videomae-base",
            output_path=output_path,
            crop_inputs=True,
            resize_inputs=True,
            target_size=(224, 224),
            num_frames=16,
            channels_first=False,  # VideoMAE expects [B,F,C,H,W] format
            num_classes=num_classes
        )
        print(f"✅ VideoMAE detector converted: {onnx_path}")
        return True
        
    except Exception as e:
        print(f"❌ VideoMAE conversion failed: {e}")
        return False


def main():
    """Run the HuggingFace video model conversion example."""
    print("🚀 HuggingFace Video Model Conversion Example")
    print("=" * 50)
    
    # Convert VideoMAE with 3 classes
    output_path = "models/video_detector.onnx"
    num_classes = 3
    if convert_videomae(output_path, num_classes):
        print("\n" + "=" * 50)
        print("✅ HuggingFace video model conversion completed!")
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   • {output_path} ({size_mb:.1f} MB)")
        print(f"   • {num_classes} output classes")
    else:
        print("\n🔧 Troubleshooting:")
        print("   • Install transformers: pip install transformers")
        print("   • Check model configuration and input format")


if __name__ == "__main__":
    main() 