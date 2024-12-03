#!/bin/bash

# Update system and install required packages
sudo apt update -y
sudo apt install python3-pip -y
sudo apt install nano -y
sudo apt install libgl1 -y
sudo apt install npm -y
sudo npm install pm2@latest -g 
sudo apt install build-essential cmake -y
sudo apt install libopenblas-dev liblapack-dev -y
sudo apt install libx11-dev libgtk-3-dev -y
sudo apt install unzip

# Install Python dependencies
pip install -e .

echo "# Default options:
IMAGE_DETECTOR=CAMO                            # Options: CAMO, UCF, NPR, None
IMAGE_DETECTOR_CONFIG=camo.yaml                # Configs live in base_miner/deepfake_detectors/configs
                                               # Supply a filename or relative path

VIDEO_DETECTOR=TALL                            # Options: TALL, None
VIDEO_DETECTOR_CONFIG=tall.yaml                # Configs live in base_miner/deepfake_detectors/configs
                                               # Supply a filename or relative path

IMAGE_DETECTOR_DEVICE=cpu                         # Options: cpu, cuda
VIDEO_DETECTOR_DEVICE=cpu

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

# Miner Settings:
MINER_AXON_PORT=8091
BLACKLIST_FORCE_VALIDATOR_PERMIT=True          # Default setting to force validator permit for blacklisting" > miner.env

