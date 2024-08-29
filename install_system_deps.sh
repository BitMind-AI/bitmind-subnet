sudo apt update -y
sudo apt install python3-pip -y
sudo apt install nano -y
sudo apt install libgl1 -y
sudo apt install npm -y
sudo npm install pm2@latest -g 
sudo apt install build-essential cmake -y
sudo apt install libopenblas-dev liblapack-dev -y
sudo apt install libx11-dev libgtk-3-dev -y

echo "MODEL_PATH=./mining_models/base.pth
NETUID=34
SUBTENSOR_NETWORK=finney
WALLET_NAME=default
WALLET_HOTKEY=default
MINER_AXON_PORT=8091
BLACKLIST_FORCE_VALIDATOR_PERMIT=True" > miner.env

echo "NETUID=34
SUBTENSOR_NETWORK=finney
WALLET_NAME=default
WALLET_HOTKEY=default
VALIDATOR_AXON_PORT=8092" > validator.env