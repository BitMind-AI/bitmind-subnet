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

Place these files in a directory (e.g., `models/`) or create a zip file containing them.

## Pushing Your Model

First, activate the virtual environment:
```bash
source .venv/bin/activate
```

Once you have your ONNX models ready, push them to the network using the `push-discriminator` command:

```bash
# Push from a directory containing ONNX files
gascli miner push-discriminator --onnx-dir models/ 

# Or push from a pre-existing zip file
gascli m push-discriminator --model-zip models.zip
```

### Command Options

The `push-discriminator` command accepts several parameters:

```bash
gascli miner push-discriminator \
  --onnx-dir models/ \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name \
  --netuid 379 \
  --chain-endpoint wss://test.finney.opentensor.ai:443/ \
  --retry-delay 60
```

**Parameters:**
- `--onnx-dir` or `--model-zip`: Path to your ONNX models
- `--wallet-name`: Bittensor wallet name (default: "default")
- `--wallet-hotkey`: Bittensor hotkey name (default: "default") 
- `--netuid`: Subnet UID (default: 379)
- `--chain-endpoint`: Subtensor network endpoint (default: "wss://test.finney.opentensor.ai:443/")
- `--retry-delay`: Retry delay in seconds (default: 60)

### What Happens During Push

1. **Model Validation**: The system checks that all required ONNX files are present
2. **Model Upload**: Your models are uploaded to the cloud inference system
3. **Blockchain Registration**: Model metadata is registered on the Bittensor blockchain
4. **Verification**: The system verifies the registration was successful

### Getting Help

```bash
gascli miner --help              # Miner help
gascli miner push-discriminator --help  # Push command help
```

**Note**: Remember to activate the virtual environment first with `source .venv/bin/activate` before running any `gascli` commands.