# Discriminative Mining Guide

## Before You Proceed

Follow the [Installation Guide](Installation.md) to set up your environment before proceeding with mining operations.

## Discriminative Mining Overview

- Miners are tasked with training multiclass classifiers that discern between genuine and AI-generated content, and are rewarded based on their accuracy. 
- For each challenge, a miner's model is presented an image, video, or audio and is required to respond with a multiclass prediction [$p_{real}$, $p_{synthetic}$, $p_{semisynthetic}$] indicating whether the media is real, fully generated, or partially modified by AI. 

## Model Preparation

Discriminative miners can submit models in two formats:
- **ONNX Format** - Traditional ONNX Runtime models (single .onnx file)
- **Safetensors Format** - Custom PyTorch architectures with native weights

**📖 [How to Create ONNX Models](ONNX.md)** - Guide for creating compatible ONNX models

You can submit models for any combination of modalities:
- `image_detector.zip` - Image classification model
- `video_detector.zip` - Video classification model  
- `audio_detector.zip` - Audio classification model

## Pushing Your Model

First, activate the virtual environment:
```bash
source .venv/bin/activate
```

Push your models to the network using the `push` command:

```bash
# Upload all three models
gascli d push \
  --image-model image_detector.zip \
  --video-model video_detector.zip \
  --audio-model audio_detector.zip \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name

# Or upload individual models
gascli d push \
  --image-model image_detector.zip \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name
```

### Command Options

The `push` command accepts several parameters:

```bash
gascli d push \
  --image-model image_detector.zip \
  --video-model video_detector.zip \
  --audio-model audio_detector.zip \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name \
  --netuid 34 \
  --chain-endpoint wss://test.finney.opentensor.ai:443/ \
  --retry-delay 60
```

**Parameters:**
- `--image-model`: Path to image detector zip file
- `--video-model`: Path to video detector zip file
- `--audio-model`: Path to audio detector zip file
- `--wallet-name`: Bittensor wallet name (default: "default")
- `--wallet-hotkey`: Bittensor hotkey name (default: "default") 
- `--netuid`: Subnet UID (default: 34)
- `--chain-endpoint`: Subtensor network endpoint (default: "wss://test.finney.opentensor.ai:443/")
- `--retry-delay`: Retry delay in seconds (default: 60)

At least one model (image, video, or audio) must be provided.

---

## Model Formats

### ONNX Format (Traditional)

Package your ONNX model in a ZIP file:

```bash
# Package image model
zip image_detector.zip image_detector.onnx

# Package video model  
zip video_detector.zip video_detector.onnx

# Package audio model
zip audio_detector.zip audio_detector.onnx
```

The ZIP should contain a single `.onnx` file.

### Safetensors Format (PyTorch Native Weights)

For custom PyTorch architectures, package these three required files:

```
model_submission.zip
├── model_config.yaml    # Metadata and preprocessing config
├── model.py             # Model architecture with load_model() function
└── model.safetensors    # Trained weights
```

#### model_config.yaml

Configuration file specifying model metadata and preprocessing:

```yaml
name: "my-detector-v1"
version: "1.0.0"
modality: "image"  # or "video", "audio"

preprocessing:
  # For image/video models:
  resize: [224, 224]
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  
  # For audio models (instead of resize/normalize):
  # sample_rate: 16000
  # duration_seconds: 6.0

model:
  num_classes: 2
  weights_file: "model.safetensors"
```

**Required fields:**
- `name`: Model identifier (alphanumeric, hyphens, underscores only)
- `modality`: One of `image`, `video`, or `audio`
- `model.num_classes`: Number of output classes (1-100)

#### model.py

Python file defining your model architecture. **Must define a `load_model()` function:**

```python
import torch
import torch.nn as nn
from safetensors.torch import load_file

class MyDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Input: [B, C, H, W] for image/video, [B, samples] for audio
        # Output: [B, num_classes] logits
        x = self.backbone(x)
        x = x.flatten(1)
        return self.classifier(x)

def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Required entry point - called by gasbench.
    
    Args:
        weights_path: Path to the .safetensors file
        num_classes: Number of output classes from config
        
    Returns:
        Loaded PyTorch model ready for inference
    """
    model = MyDetector(num_classes=num_classes)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    return model
```

**Allowed imports:**
- `torch`, `torch.nn`, `torch.nn.functional`
- `torchvision`, `torchvision.models`, `torchvision.transforms`
- `torchaudio`
- `numpy`, `math`, `functools`, `typing`
- `safetensors`, `safetensors.torch`
- `timm`, `einops`

**Blocked imports (security):**
- `os`, `sys`, `subprocess`, `pathlib` - system access
- `requests`, `socket`, `urllib` - network access
- `pickle`, `marshal` - serialization/code execution
- `eval`, `exec`, `__import__` - dynamic code execution

#### Creating safetensors weights

Save your trained model weights using safetensors:

```python
from safetensors.torch import save_file

# After training your model
state_dict = model.state_dict()
save_file(state_dict, "model.safetensors")
```

#### Packaging safetensors submission

```bash
zip my_detector.zip model_config.yaml model.py model.safetensors
```

---

### What Happens During Push

1. **Model Validation**: The system checks that the zip files are present and valid
2. **Model Upload**: Your model zip files are uploaded to the cloud inference system
3. **Blockchain Registration**: Model metadata is registered on the Bittensor blockchain
4. **Verification**: The system verifies the registration was successful

### Getting Help

```bash
gascli discriminator --help        # Miner help
gascli d push --help               # Push command help
```

**Note**: Remember to activate the virtual environment first with `source .venv/bin/activate` before running any `gascli` commands.