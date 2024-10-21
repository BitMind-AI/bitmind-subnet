#!/bin/bash

# Load environment variables from .env file
set -a
source validator.env
set +a

# Login to Weights & Biases
if ! wandb login $WANDB_API_KEY; then
  echo "Failed to login to Weights & Biases with the provided API key."
  exit 1
fi

# Login to Hugging Face
if ! huggingface-cli login --token $HUGGING_FACE_TOKEN; then
  echo "Failed to login to Hugging Face with the provided token."
  exit 1
fi

echo "Verifying access to synthetic image generation models. This may take a few minutes."
if ! python3 bitmind/validator/verify_models.py; then
  echo "Failed to verify diffusion models. Please check the configurations or model access permissions."
  exit 1
fi

# Check if the process is already running
if pm2 list | grep -q "bitmind_validator"; then
  echo "Process 'bitmind_validator' is already running. Deleting it..."
  pm2 delete bitmind_validator
fi

# Start the process with arguments from environment variables
pm2 start neurons/validator.py --name bitmind_validator -- \
  --netuid $NETUID \
  --subtensor.network $SUBTENSOR_NETWORK \
  --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --axon.port $VALIDATOR_AXON_PORT