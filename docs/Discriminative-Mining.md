# Discriminative Mining Guide

## Before You Proceed

Follow the [Installation Guide](Installation.md) to set up your environment before proceeding with mining operations.

## Discriminative Mining Overview

- Miners are tasked with training multiclass classifiers that discern between genuine and AI-generated content, and are rewarded based on their accuracy. 
- For each challenge, a miner's model is presented an image or video and is required to respond with a multiclass prediction [$p_{real}$, $p_{synthetic}$, $p_{semisynthetic}$] indicating whether the media is real, fully generated, or partially modified by AI. 

## Model Preparation

Discriminative miners need to prepare ONNX models for classification tasks and package them as zip files. You'll need:

**📖 [How to Create ONNX Models](ONNX.md)** - Complete guide for creating compatible ONNX models

- `image_detector.zip` - Zip file containing your image classification ONNX model
- `video_detector.zip` - Zip file containing your video classification ONNX model

Each zip file should contain the corresponding ONNX model file (`image_detector.onnx` or `video_detector.onnx`).

## Pushing Your Model

First, activate the virtual environment:
```bash
source .venv/bin/activate
```

Once you have your ONNX models packaged as zip files, push them to the network using the `push` command. You can upload models one at a time or both together:

```bash
# Upload both models together
gascli d push \
  --image-model image_detector.zip \
  --video-model video_detector.zip \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name

# Or upload just the image model
gascli d push \
  --image-model image_detector.zip \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name

# Or upload just the video model
gascli d push \
  --video-model video_detector.zip \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name
```

### Command Options

The `push` command accepts several parameters:

```bash
gascli d push \
  --image-model image_detector.zip \
  --video-model video_detector.zip \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name \
  --netuid 34 \
  --chain-endpoint wss://test.finney.opentensor.ai:443/ \
  --retry-delay 60
```

**Parameters:**
- `--image-model`: Path to image detector zip file (optional, but at least one model required)
- `--video-model`: Path to video detector zip file (optional, but at least one model required)
- `--wallet-name`: Bittensor wallet name (default: "default")
- `--wallet-hotkey`: Bittensor hotkey name (default: "default") 
- `--netuid`: Subnet UID (default: 34)
- `--chain-endpoint`: Subtensor network endpoint (default: "wss://test.finney.opentensor.ai:443/")
- `--retry-delay`: Retry delay in seconds (default: 60)


### Packaging Your Models

Before pushing, you need to package your ONNX models into zip files. The zip format helps keep the system flexible should we expand to other model formats, or ones that require multiple files. Currently, each zip file should contain the corresponding ONNX model:

```bash
# Package image model
zip image_detector.zip image_detector.onnx

# Package video model
zip video_detector.zip video_detector.onnx
```

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