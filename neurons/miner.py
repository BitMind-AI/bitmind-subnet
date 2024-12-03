# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: aliang322, benliang99, dubm
# Copyright © 2023 Bitmind

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from PIL import Image
import bittensor as bt
import torch
import base64
import time
import typing
import io
import os
import sys
import numpy as np

from base_miner import DETECTOR_REGISTRY
from bitmind.base.miner import BaseMinerNeuron
from bitmind.protocol import ImageSynapse, VideoSynapse, decode_video_synapse
from bitmind.utils.config import get_device


class Miner(BaseMinerNeuron):

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward_image,
            blacklist_fn=self.blacklist_image,
            priority_fn=self.priority_image,
        ).attach(
            forward_fn=self.forward_video,
            blacklist_fn=self.blacklist_video,
            priority_fn=self.priority_video,
        )
        bt.logging.info(f"Axon created: {self.axon}")

        bt.logging.info("Loading image detection model if configured")
        self.load_image_detector()
        bt.logging.info("Loading video detection model if configured")
        self.load_video_detector()

    def load_image_detector(self):
        if (str(self.config.neuron.image_detector).lower() == 'none' or
            str(self.config.neuron.image_detector_config).lower() == 'none'):
            bt.logging.warning("No image detector configuration provided, skipping.")
            self.image_detector = None
            return

        if self.config.neuron.image_detector_device == 'auto':
            bt.logging.warning("Automatic device configuration enabled for image detector")
            self.config.neuron.image_detector_device = get_device()

        self.image_detector = DETECTOR_REGISTRY[self.config.neuron.image_detector](
            config=self.config.neuron.image_detector_config,
            device=self.config.neuron.image_detector_device
        )
        bt.logging.info(f"Loaded image detection model: {self.config.neuron.image_detector}")

    def load_video_detector(self):
        if (str(self.config.neuron.video_detector).lower() == 'none' or
            str(self.config.neuron.video_detector_config).lower() == 'none'):
            bt.logging.warning("No video detector configuration provided, skipping.")
            self.video_detector = None
            return

        if self.config.neuron.video_detector_device == 'auto':
            bt.logging.warning("Automatic device configuration enabled for video detector")
            self.config.neuron.video_detector_device = get_device()

        self.video_detector = DETECTOR_REGISTRY[self.config.neuron.video_detector](
            config=self.config.neuron.video_detector_config,
            device=self.config.neuron.video_detector_device
        )
        bt.logging.info(f"Loaded video detection model: {self.config.neuron.video_detector}")

    async def forward_image(
        self, synapse: ImageSynapse
    ) -> ImageSynapse:
        """
        Perform inference on image

        Args:
            synapse (bt.Synapse): The synapse object containing the list of b64 encoded images in the
            'images' field.

        Returns:
            bt.Synapse: The synapse object with the 'predictions' field populated with a list of probabilities

        """
        if self.image_detector is None:
            bt.logging.info("Image detection model not configured; skipping image challenge")
        else:
            bt.logging.info("Received image challenge!")
            try:
                image_bytes = base64.b64decode(synapse.image)
                image = Image.open(io.BytesIO(image_bytes))
                synapse.prediction = self.image_detector(image)
            except Exception as e:
                bt.logging.error("Error performing inference")
                bt.logging.error(e)
            bt.logging.info(f"PREDICTION: {synapse.prediction}")
        return synapse

    async def forward_video(
        self, synapse: VideoSynapse
    ) -> VideoSynapse:
        """
        Perform inference on video 
        Args:
            synapse (bt.Synapse): The synapse object containing the list of b64 encoded images in the
            'images' field.

        Returns:
            bt.Synapse: The synapse object with the 'predictions' field populated with a list of probabilities

        """
        if self.video_detector is None:
            bt.logging.info("Video detection model not configured; skipping video challenge")
        else:
            bt.logging.info("Received video challenge!")
            try:
                frames_tensor = decode_video_synapse(synapse)
                synapse.prediction = self.video_detector(frames_tensor)
            except Exception as e:
                bt.logging.error("Error performing inference")
                bt.logging.error(e)
            bt.logging.info(f"PREDICTION: {synapse.prediction}")
        return synapse

    async def blacklist_image(self, synapse: ImageSynapse) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def blacklist_video(self, synapse: VideoSynapse) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_image(self, synapse: ImageSynapse) -> float:
        return await self.priority(synapse)

    async def priority_video(self, synapse: VideoSynapse) -> float:
        return await self.priority(synapse)

    def save_state(self):
        pass


# This is the main function, which runs the miner.
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running | uid {miner.uid} | {time.time()}")
            time.sleep(5)

