# Discriminative Mining Guide

## Before You Proceed

Follow the [Installation Guide](Installation.md) to set up your environment before proceeding with mining operations.

## Discriminative Mining Overview

- Miners are tasked with training multiclass classifiers that discern between genuine and AI-generated content, and are rewarded based on their accuracy. 
- For each challenge, a miner's model is presented an image or video and is required to respond with a multiclass prediction [$p_{real}$, $p_{synthetic}$, $p_{semisynthetic}$] indicating whether the media is real, fully generated, or partially modified by AI. 

## Model Preparation

Discriminative miners need to prepare ONNX models for classification tasks. You'll need two ONNX files:

**📖 [How to Create ONNX Models](ONNX.md)** - Complete guide for creating compatible ONNX models

- `image_detector.onnx` - For image classification tasks  
- `video_detector.onnx` - For video classification tasks

## Pushing Your Model

First, activate the virtual environment:
```bash
source .venv/bin/activate
```

Once you have your ONNX models ready, push them to the network using the `push` command. You can upload models one at a time or both together:

```bash
# Upload both models together
gascli d push \
  --image-model image_detector.onnx \
  --video-model video_detector.onnx \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name

# Or upload just the image model
gascli d push \
  --image-model image_detector.onnx \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name

# Or upload just the video model
gascli d push \
  --video-model video_detector.onnx \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name
```

### Command Options

The `push` command accepts several parameters:

```bash
gascli d push \
  --image-model image_detector.onnx \
  --video-model video_detector.onnx \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name \
  --netuid 34 \
  --chain-endpoint wss://test.finney.opentensor.ai:443/ \
  --retry-delay 60
```

**Parameters:**
- `--image-model`: Path to image ONNX model (optional, but at least one model required)
- `--video-model`: Path to video ONNX model (optional, but at least one model required)
- `--wallet-name`: Bittensor wallet name (default: "default")
- `--wallet-hotkey`: Bittensor hotkey name (default: "default") 
- `--netuid`: Subnet UID (default: 34)
- `--chain-endpoint`: Subtensor network endpoint (default: "wss://test.finney.opentensor.ai:443/")
- `--retry-delay`: Retry delay in seconds (default: 60)


### What Happens During Push

1. **Model Validation**: The system checks that all required ONNX files are present
2. **Model Upload**: Your models are uploaded to the cloud inference system
3. **Blockchain Registration**: Model metadata is registered on the Bittensor blockchain
4. **Verification**: The system verifies the registration was successful

### Getting Help

```bash
gascli discriminator --help        # Miner help
gascli d push --help               # Push command help
```

**Note**: Remember to activate the virtual environment first with `source .venv/bin/activate` before running any `gascli` commands.