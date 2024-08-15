#!/bin/bash

# Check if the process is already running
if pm2 list | grep -q "bitmind_validator"; then
  echo "Process 'bitmind_validator' is already running. Deleting it..."
  pm2 delete bitmind_validator
fi

pm2 start python --name bitmind_validator -- neurons/validator.py --netuid 34 --subtensor.network finney --wallet.name default --wallet.hotkey default --axon.port 8092
