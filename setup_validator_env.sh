#!/bin/bash

# Update system and install required packages
sudo apt update -y
sudo apt install python3-pip nano libgl1 npm ffmpeg -y
sudo npm install pm2@latest -g
sudo apt install -y unzip

# Install Python dependencies
pip install -e .
pip install -r requirements-validator.txt

# Check if validator.env exists
if [ -f "validator.env" ]; then
    echo "File 'validator.env' already exists. Skipping creation."
else
    echo "# Default options:

# Subtensor Network Configuration:
NETUID=34                                      # Network User ID options: 34, 168
SUBTENSOR_NETWORK=finney                       # Networks: finney, test, local
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
                                                # Endpoints:
                                                # - wss://entrypoint-finney.opentensor.ai:443
                                                # - wss://test.finney.opentensor.ai:443/

# Wallet Configuration:
WALLET_NAME=default
WALLET_HOTKEY=default

# Note: If you're using RunPod, you must select a port >= 70000 for symmetric mapping
# Validator Port Setting:
VALIDATOR_AXON_PORT=8092
VALIDATOR_PROXY_PORT=10913
DEVICE=cuda

# API Keys:
WANDB_API_KEY=your_wandb_api_key_here
HUGGING_FACE_TOKEN=your_hugging_face_token_here" > validator.env
    echo "File 'validator.env' created."
fi
