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

# Install Python dependencies
pip install -e .
pip install -r requirements-miner.txt

# Setup environment variables for mining configuration
echo "MODEL_PATH=./mining_models/base.pth
NEURON_PATH=./neurons/npr_miner.py
NETUID=34
SUBTENSOR_NETWORK=finney
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
WALLET_NAME=default
WALLET_HOTKEY=default
MINER_AXON_PORT=8091
BLACKLIST_FORCE_VALIDATOR_PERMIT=True" > miner.env