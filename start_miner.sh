#!/bin/bash

# Load environment variables from .env file
set -a
source miner.env
set +a

# Check if the process is already running
if pm2 list | grep -q "bitmind_miner"; then
  echo "Process 'bitmind_miner' is already running. Deleting it..."
  pm2 delete bitmind_miner
fi

# Start the process with arguments from environment variables
pm2 start python --name bitmind_miner -- neurons/camo_miner.py \
  --neuron.model_path $MODEL_PATH \
  --netuid $NETUID \
  --subtensor.network $SUBTENSOR_NETWORK \
  --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --axon.port $MINER_AXON_PORT \
  --blacklist.force_validator_permit $BLACKLIST_FORCE_VALIDATOR_PERMIT
