#!/bin/bash

###################################
# LOAD ENV FILE
###################################
set -a
source .env.validator  
set +a

###################################
# LOG IN TO THIRD PARTY SERVICES
###################################
# Login to Weights & Biases
if ! wandb login $WANDB_API_KEY; then
  echo "Failed to login to Weights & Biases with the provided API key."
  exit 1
fi
echo "Logged into W&B with API key provided in .env.validator"

# Login to Hugging Face
if ! huggingface-cli login --token $HUGGING_FACE_TOKEN; then
  echo "Failed to login to Hugging Face with the provided token."
  exit 1
fi
echo "Logged into W&B with token provided in .env.validator"

###################################
# PREPARE CLI ARGS
###################################
: ${PROXY_PORT:=10913}
: ${PROXY_EXTERNAL_PORT:=$PROXY_PORT}
: ${DEVICE:=cuda}

if [[ "$CHAIN_ENDPOINT" == *"test"* ]]; then
  NETUID=168
  NETWORK="test"
elif [[ "$CHAIN_ENDPOINT" == *"finney"* ]]; then
  NETUID=34
  NETWORK="finney"
fi

case "$LOGLEVEL" in
  "trace")
    LOG_PARAM="--logging.trace"
    ;;
  "debug")
    LOG_PARAM="--logging.debug"
    ;;
  "info")
    LOG_PARAM="--logging.info"
    ;;
  *)
    # Default to info if LOGLEVEL is not set or invalid
    LOG_PARAM="--logging.info"
    ;;
esac

# Set auto-update parameter based on AUTO_UPDATE
if [ "$AUTO_UPDATE" = true ]; then
  AUTO_UPDATE_PARAM=""
else
  AUTO_UPDATE_PARAM="--autoupdate-off"
fi

if [ "$HEARTBEAT" = true ]; then
  HEARTBEAT_PARAM="--heartbeat"
else
  HEARTBEAT_PARAM=""
fi

###################################
# STOP AND WAIT FOR CLEANUP
###################################
VALIDATOR="sn34-validator"
GENERATOR="sn34-generator"
PROXY="sn34-proxy"

# Stop any existing processes
if pm2 list | grep -q "$VALIDATOR"; then
    echo "'$VALIDATOR' is already running. Deleting it..."
    pm2 delete $VALIDATOR
    sleep 1
fi

if pm2 list | grep -q "$GENERATOR"; then
    echo "'$GENERATOR' is already running. Deleting it..."
    pm2 delete $GENERATOR
    sleep 2
fi

if pm2 list | grep -q "$PROXY"; then
    echo "'$PROXY' is already running. Deleting it..."
    pm2 delete $PROXY
    sleep 1
fi


###################################
# START PROCESSES 
###################################
SN34_CACHE_DIR=$(eval echo "$SN34_CACHE_DIR")

echo "Starting validator and generator | chain_endpoint: $CHAIN_ENDPOINT | netuid: $NETUID"

# Run data generator
pm2 start neurons/generator.py \
  --interpreter python3 \
  --kill-timeout 2000 \
  --name $GENERATOR \
  -- \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --netuid $NETUID \
  --subtensor.chain_endpoint $CHAIN_ENDPOINT \
  --cache-dir $SN34_CACHE_DIR \
  --device $DEVICE

# Run validator
pm2 start neurons/validator.py \
  --interpreter python3 \
  --kill-timeout 1000 \
  --name $VALIDATOR \
  -- \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --netuid $NETUID \
  --subtensor.chain_endpoint $CHAIN_ENDPOINT \
  --epoch-length 360 \
  --cache-dir $SN34_CACHE_DIR \
  --proxy.port $PROXY_PORT \
  $LOG_PARAM \
  $AUTO_UPDATE_PARAM \
  $HEARTBEAT_PARAM

# Run validator proxy
pm2 start neurons/proxy.py \
  --interpreter python3 \
  --kill-timeout 1000 \
  --name $PROXY \
  -- \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --netuid $NETUID \
  --subtensor.chain_endpoint $CHAIN_ENDPOINT \
  --proxy.port $PROXY_PORT \
  --proxy.external_port $PROXY_EXTERNAL_PORT \
  $LOG_PARAM
