#!/bin/bash

set -a
source miner.env
set +a

if pm2 list | grep -q "bitmind_miner"; then
  echo "Process 'bitmind_miner' is already running. Deleting it..."
  pm2 delete bitmind_miner
fi

pm2 start neurons/miner.py --name bitmind_miner -- \
  --neuron.image_detector ${IMAGE_DETECTOR:-None} \
  --neuron.image_detector_config ${IMAGE_DETECTOR_CONFIG:-None} \
  --neuron.image_detector_device ${IMAGE_DETECTOR_DEVICE:-None} \
  --neuron.video_detector ${VIDEO_DETECTOR:-None} \
  --neuron.video_detector_config ${VIDEO_DETECTOR_CONFIG:-None} \
  --neuron.video_detector_device ${VIDEO_DETECTOR_DEVICE:-None} \
  --netuid $NETUID \
  --subtensor.network $SUBTENSOR_NETWORK \
  --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --axon.port $MINER_AXON_PORT \
  --blacklist.force_validator_permit $BLACKLIST_FORCE_VALIDATOR_PERMIT
