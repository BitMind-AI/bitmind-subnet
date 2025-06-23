#!/bin/bash

###################################
# LOAD ENV FILE
###################################
set -a
source .env.miner
set +a

###################################
# PREPARE CLI ARGS 
###################################
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
FORCE_VPERMIT_PARAM=""
if [ "$FORCE_VPERMIT" = false ]; then
  FORCE_VPERMIT_PARAM="--no-force-validator-permit"
fi


###################################
# RESTART PROCESSES 
###################################
# Set script based on MINER_TYPE
if [ "${MINER_TYPE,,}" = "detector" ]; then
    SCRIPT="neurons.miner.detection_miner"
elif [ "${MINER_TYPE,,}" = "segmenter" ]; then
    SCRIPT="neurons.miner.segmentation_miner"
else
    echo "Invalid MINER_TYPE: $MINER_TYPE. Must be 'detector' or 'segmenter'"
    exit 1
fi

NAME="sn34-$MINER_TYPE"

# Stop any existing processes
if pm2 list | grep -q "$NAME"; then
    echo "'$NAME' is already running. Deleting it..."
    pm2 delete $NAME
fi

echo "Starting $NAME | chain_endpoint: $CHAIN_ENDPOINT | netuid: $NETUID"

# Run data generator
pm2 start python3 \
  --name $NAME \
  -- \
  -m $SCRIPT \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --netuid $NETUID \
  --subtensor.chain_endpoint $CHAIN_ENDPOINT \
  --axon.port $AXON_PORT \
  --axon.external_ip $AXON_EXTERNAL_IP \
  --device $DEVICE \
  $FORCE_VPERMIT_PARAM
 
