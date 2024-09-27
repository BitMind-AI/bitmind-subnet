#!/bin/bash

sudo apt update -y
sudo apt install python3-pip -y
sudo apt install nano -y
sudo apt install libgl1 -y
sudo apt install npm -y
sudo npm install pm2@latest -g

pip install -e .
pip install -r requirements-validator.sh

echo "# Default options:
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

# Validator Port Setting:
VALIDATOR_AXON_PORT=8092" > validator.env

