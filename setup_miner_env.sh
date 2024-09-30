#!/bin/bash

sudo apt update -y
sudo apt install python3-pip -y
sudo apt install nano -y
sudo apt install libgl1 -y
sudo apt install npm -y
sudo npm install pm2@latest -g 
sudo apt install build-essential cmake -y
sudo apt install libopenblas-dev liblapack-dev -y
sudo apt install libx11-dev libgtk-3-dev -y

pip install -e .
pip install -r requirements-miner.txt

echo "# Default options:
DETECTOR=CAMO                                  # Options: CAMO, UCF, NPR
DETECTOR_CONFIG=camo.yaml                      # Configurations: camo.yaml, ucf.yaml, npr.yaml
DEVICE=cpu                                     # Options: cpu, cuda
NETUID=34                                      # Network User ID options: 34, 168

# Subtensor Network Configuration:
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

