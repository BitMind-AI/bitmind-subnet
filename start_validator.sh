#!/bin/bash

# Load environment variables from .env file
set -a
source validator.env
set +a

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
