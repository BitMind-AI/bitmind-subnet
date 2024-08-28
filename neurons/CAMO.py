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
from ultralytics import YOLO

script_directory = os.path.dirname(os.path.realpath(__file__))
base_ucf_path = os.path.join(script_directory, '../base_miner/UCF/')
resolved_path = os.path.abspath(base_ucf_path)
sys.path.append(resolved_path)
predictor_path = os.path.join(base_ucf_path, 'preprocessing', 'dlib_tools',
                              'shape_predictor_81_face_landmarks.dat')

from pretrained_ucf import UCF
from bitmind.base.miner import BaseMinerNeuron
from bitmind.protocol import ImageSynapse

UCF_CONFIG_PATH = os.path.join(base_ucf_path, 'config', 'ucf.yaml')
UCF_WEIGHTS_PATH = os.path.join(base_ucf_path, 'weights')
UCF_DFB_CHECKPOINT_NAME = "ucf_bitmind_face.pth"
UCF_BITMIND_CHECKPOINT_NAME = "ucf-bitmind-18k.pth"

class Miner(BaseMinerNeuron):

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        
        # Dictionary to hold different types of detectors
        self.detectors = {}

        # Attempt to load face and general detectors
        self.detectors['face'] = self.load_detector(
            UCF_CONFIG_PATH,
            UCF_WEIGHTS_PATH,
            UCF_DFB_CHECKPOINT_NAME,
            predictor_path,
            'face'
        )
        
        self.detectors['general'] = self.load_detector(
            UCF_CONFIG_PATH,
            UCF_WEIGHTS_PATH,
            UCF_BITMIND_CHECKPOINT_NAME,
            predictor_path,
            'general'
        )

        self.object_detector = YOLO("yolov8x.pt")
        

    def load_detector(self, config_path, weights_dir, checkpoint_name, predictor_path, detector_type):
        """
        Load a detector model with given parameters.
        
        Args:
            config_path (str): Path to the configuration file for the detector.
            weights_dir (str): Directory where the model weights are stored.
            checkpoint_name (str): Name of the model checkpoint file.
            predictor_path (str): Path to any additional required files, like a shape predictor.
            detector_type (str): A descriptive name for the type of detector being loaded.

        Returns:
            Loaded detector object if successful, None otherwise.
        """
        try:
            bt.logging.info(f"Loading {detector_type} detection model from {weights_dir}")
            detector = UCF(
                config_path=config_path,
                weights_dir=weights_dir,
                ucf_checkpoint_name=checkpoint_name,
                predictor_path=predictor_path
            )
            return detector
        except Exception as e:
            bt.logging.error(f"Error loading {detector_type} model")
            bt.logging.error(str(e))
            return None


    async def classify_image(self, image, use_object_detection=True):
        """
        Classify the image to determine its content type.

        Args:
            image (PIL.Image): The image to analyze.

        Returns:
            str: 'face' if the image contains at least one face, otherwise 'general'.
        """

        _, num_faces = self.detectors['face'].detect_faces(image)
        
        if use_object_detection:
            try:
                results = self.object_detector(image)
            except Exception as e:
                print(f"Error in object detection: {e}")
                return 'general'
        
            detected_classes = []
            try:
                for result in results:
                    for box in result.boxes:
                        try:
                            if box.conf.item() is not None and box.conf.item() > 0.5:
                                detected_classes.append(result.names[box.cls.item()])
                        except Exception as e:
                            print(f"Error processing object detection box: {e}")
                            continue
            except Exception as e:
                print(f"Error processing object detection results: {e}")
                return 'general'
        
            try:
                if 'person' in detected_classes:
                    if num_faces:
                        return 'face'
                return 'general'
            except Exception as e:
                print(f"Error checking detected classes: {e}")
                return 'general'
                
        else:
            try:
                if num_faces:
                    return 'face'
                return 'general'
            except Exception as e:
                print(f"Error in non-object detection branch: {e}")
                return 'general'

    async def forward(
        self, synapse: ImageSynapse
    ) -> ImageSynapse:
        """
        Loads the deepfake detection model (a PyTorch binary classifier) from the path specified in --neuron.model_path.
        Processes the incoming ImageSynapse and passes the image to the loaded model for classification.
        The model is loaded here, rather than in __init__, so that miners may (backup) and overwrite
        their model file as a means of updating their miner's predictor.

        Args:
            synapse (ImageSynapse): The synapse object containing the list of b64 encoded images in the
            'images' field.

        Returns:
            ImageSynapse: The synapse object with the 'predictions' field populated with a list of probabilities

        """
        try:
            image_bytes = base64.b64decode(synapse.image)
            image = Image.open(io.BytesIO(image_bytes))

            # Determine image content type.
            image_type = await self.classify_image(image)

            if image_type == "face":
                image_tensor = self.detectors['face'].preprocess(image)
                pred = self.detectors['face'].infer(image_tensor)
            else:
                image_tensor = self.detectors['general'].preprocess(image)
                pred = self.detectors['general'].infer(image_tensor)

            synapse.prediction = pred
            
        except Exception as e:
            bt.logging.error("Error performing inference")
            bt.logging.error(e)

        bt.logging.info(f"PREDICTION: {synapse.prediction}")
        return synapse

    async def blacklist(
        self, synapse: ImageSynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (ImageSynapse): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: ImageSynapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (ImageSynapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.

        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority

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