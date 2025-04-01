#!/bin/bash

# Load environment variables from .env file & set defaults
set -a
source validator.env
set +a

: ${VALIDATOR_PROXY_PORT:=10913}
: ${DEVICE:=cuda}

VALIDATOR_PROCESS_NAME="bitmind_validator"
DATA_GEN_PROCESS_NAME="bitmind_data_generator"
CACHE_UPDATE_PROCESS_NAME="bitmind_cache_updater"

# Clear cache if specified 
while [[ $# -gt 0 ]]; do
  case $1 in
    --clear-cache)
      rm -rf ~/.cache/sn34
      shift
      ;;
    *)
      shift
      ;;
  esac
done

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

# STOP VALIDATOR PROCESS
if pm2 list | grep -q "$VALIDATOR_PROCESS_NAME"; then
  echo "Process '$VALIDATOR_PROCESS_NAME' is already running. Deleting it..."
  pm2 delete $VALIDATOR_PROCESS_NAME
fi

# STOP REAL DATA CACHE UPDATER PROCESS
if pm2 list | grep -q "$CACHE_UPDATE_PROCESS_NAME"; then
  echo "Process '$CACHE_UPDATE_PROCESS_NAME' is already running. Deleting it..."
  pm2 delete $CACHE_UPDATE_PROCESS_NAME
fi

# STOP SYNTHETIC DATA GENERATOR PROCESS
if pm2 list | grep -q "$DATA_GEN_PROCESS_NAME"; then
  echo "Process '$DATA_GEN_PROCESS_NAME' is already running. Deleting it..."
  pm2 delete $DATA_GEN_PROCESS_NAME
fi


WANDB_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/wandb"
echo "Pruning $WANDB_DIR"
python3 bitmind/validator/scripts/prune_wandb_cache --dir $WANDB_DIR

echo "Verifying access to synthetic image generation models. This may take a few minutes."
if ! python3 bitmind/validator/verify_models.py; then
  echo "Failed to verify diffusion models. Please check the configurations or model access permissions."
  exit 1
fi

echo "Starting validator process"
pm2 start neurons/validator.py --name $VALIDATOR_PROCESS_NAME -- \
  --netuid $NETUID \
  --subtensor.network $SUBTENSOR_NETWORK \
  --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --axon.port $VALIDATOR_AXON_PORT \
  --proxy.port $VALIDATOR_PROXY_PORT

echo "Starting real data cache updater process"
pm2 start bitmind/validator/scripts/run_cache_updater.py --name $CACHE_UPDATE_PROCESS_NAME

echo "Starting synthetic data generation process"
pm2 start bitmind/validator/scripts/run_data_generator.py --name $DATA_GEN_PROCESS_NAME -- \
  --device $DEVICE
