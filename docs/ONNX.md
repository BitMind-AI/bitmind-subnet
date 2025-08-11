# ONNX Model Creation Guide

This guide explains how to create ONNX models for discriminative mining using the example scripts in `neurons/discriminator/onnx_examples`.

## Key Requirements

**⚠️ IMPORTANT**: Your ONNX models must meet these requirements:

1. **Dynamic Axes**: Use dynamic axes for everything except Channels
   - Image models: `{0: 'batch_size', 2: 'height', 3: 'width'}`
   - Video models: `{0: 'batch_size', 1: 'frames', 3: 'height', 4: 'width'}`

2. **Pixel Range**: Accept raw pixel values in range `[0-255]`
   - The model wrapper handles all preprocessing (normalization, resizing, etc.)
   - Your model should expect raw uint8 inputs

3. **Output Format**: Return logits for 3 classes `[real, synthetic, semisynthetic]` with the shape (batch_size, 3)

## Example Scripts

The `neurons/discriminator/onnx_examples` directory contains working examples:

### PyTorch Models
- `pytorch_models/image_model.py` - Custom image classification model
- `pytorch_models/video_model.py` - Custom video classification model

### HuggingFace Models  
- `huggingface_models/image_model.py` - Convert HuggingFace image models (e.g., ResNet50)
- `huggingface_models/video_model.py` - Convert HuggingFace video models (e.g., VideoMAE)

## Quick Start

1. **Navigate to the examples directory:**
   ```bash
   cd neurons/discriminator/onnx_examples
   ```

2. **Run an example script:**
   ```bash
   # For custom PyTorch models
   python pytorch_models/image_model.py
   python pytorch_models/video_model.py
   
   # For HuggingFace models
   python huggingface_models/image_model.py
   python huggingface_models/video_model.py
   ```

3. **Check the output:**
   ```bash
   ls models/
   # Should see: image_detector.onnx, video_detector.onnx
   ```

## Model Wrapper

The `model_wrapper.py` file contains example preprocessing wrappers that handle:
- Pixel normalization (`[0-255]` → `[0-1]`)
- Image resizing and cropping
- ImageNet normalization
- Video frame processing

**Follow the ** to ensure your models work correctly with the network.

## Custom Models

To create your own model:

1. **Inherit from `nn.Module`** and implement your architecture
2. **Wrap with `ImageModelWrapper` or `VideoModelWrapper`**
3. **Export with dynamic axes** as shown in the examples
4. **Test with raw uint8 inputs** to ensure compatibility

## Example Export Code

```python
# Create dummy input with raw pixel values
dummy_input = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.uint8)

# Export with dynamic axes
torch.onnx.export(
    wrapped_model,
    dummy_input,
    "models/image_detector.onnx",
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
```

## Next Steps

Once you have your ONNX files, follow the [Discriminative Mining Guide](Discriminative-Mining.md) to push them to the network.
