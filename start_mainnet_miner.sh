#!/bin/bash

# Check if the process is already running
if pm2 list | grep -q "bitmind_miner"; then
  echo "Process 'bitmind_miner' is already running. Deleting it..."
  pm2 delete bitmind_miner
fi

pm2 start python --name bitmind_miner -- neurons/miner.py --neuron.model_path ./mining_models/base.pth --netuid 34 --subtensor.network finney --wallet.name default --wallet.hotkey default --axon.port 8091 --blacklist.force_validator_permit True
