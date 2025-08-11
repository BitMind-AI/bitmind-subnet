"""
Example: Converting a HuggingFace Image Model to ONNX
"""

import os
import time
import torch
from typing import Optional, Tuple

from transformers import AutoModelForImageClassification

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_wrapper import ImageModelWrapper


def convert_huggingface_image_model(model_name: str, output_path: str,
                                   crop_inputs: bool = True, resize_inputs: bool = True,
                                   target_size: Tuple[int, int] = (224, 224)
                                   ) -> str:
    """
    Convert a HuggingFace image model to ONNX format.
    
    Args:
        model_name: HuggingFace model name (e.g., "microsoft/resnet-50")
        output_path: Output path for ONNX file
        crop_inputs: Whether to crop inputs to square
        resize_inputs: Whether to resize inputs to target size
        target_size: Target size for resizing (width, height)
        num_classes: Number of output classes (sets classifier to this size)
        
    Returns:
        Path to the generated ONNX file
    """
    print(f"Converting HuggingFace image model: {model_name}")
    start_time = time.time()
    
    # Load the model configured for desired number of classes
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    
    # Ensure label mappings align with the requested number of classes
    id2label = {i: f"LABEL_{i}" for i in range(3)}
    label2id = {v: k for k, v in id2label.items()}
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.eval()
    
    # Create preprocessing wrapper
    wrapped_model = ImageModelWrapper(
        model, 
        crop_inputs=crop_inputs, 
        resize_inputs=resize_inputs, 
        target_size=target_size
    )
    wrapped_model.eval()
    
    # Create dummy input (raw pixel values [0, 255])
    dummy_input = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.uint8)
    
    # Export to ONNX
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
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
    
    elapsed_time = time.time() - start_time
    print(f"✅ HuggingFace image model converted successfully in {elapsed_time:.2f} seconds")
    print(f"📁 ONNX file saved to: {output_path}")
    
    return output_path


def convert_resnet50(output_path="models/image_detector.onnx", num_classes: int = 3):
    """Convert ResNet50 to ONNX."""
    print("Converting ResNet50...")
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    try:
        onnx_path = convert_huggingface_image_model(
            model_name="microsoft/resnet-50",
            output_path=output_path,
            crop_inputs=True,
            resize_inputs=True,
            target_size=(224, 224),
            num_classes=num_classes
        )
        print(f"✅ ResNet50 detector converted: {onnx_path}")
        return True
        
    except Exception as e:
        print(f"❌ ResNet50 conversion failed: {e}")
        return False


def main():
    """Run the HuggingFace image model conversion example."""
    print("🚀 HuggingFace Image Model Conversion Example")
    print("=" * 50)
    
    # Convert ResNet50
    convert_resnet50()
    
    print("\n" + "=" * 50)
    print("✅ HuggingFace image model conversion completed!")
    print("\n💡 Key Points:")
    print("   • Use AutoModelForImageClassification for HuggingFace models")
    print("   • ImageModelWrapper handles preprocessing")
    print("   • Direct torch.onnx.export() for maximum flexibility")
    
    print("\n🔧 Troubleshooting:")
    print("   • Install transformers: pip install transformers")
    print("   • Check model configuration and input format")
    
    print("\n📁 Generated ONNX files:")
    if os.path.exists("models"):
        for file in os.listdir("models"):
            if file.endswith(".onnx"):
                file_path = os.path.join("models", file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   • models/{file} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main() 