from PIL import Image
import bittensor as bt
import base64
import io
import os
import sys
from typing import List
import asyncio
from io import BytesIO
import time

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
UCF_DFB_CHECKPOINT_NAME = "ucf_best.pth"
UCF_BITMIND_CHECKPOINT_NAME = "ucf_bitmind_best.pth"

class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        try:
            face_model_path = os.path.join(UCF_WEIGHTS_PATH, UCF_DFB_CHECKPOINT_NAME)
            bt.logging.info(f"Loading face detection model from {face_model_path}")
            self.face_model = UCF(config_path=UCF_CONFIG_PATH,
                                  weights_dir=UCF_WEIGHTS_PATH,
                                  ucf_checkpoint_name=UCF_DFB_CHECKPOINT_NAME,
                                  predictor_path=predictor_path)
        except Exception as e:
            bt.logging.error("Error loading face model", exc_info=True)

        try:
            general_model_path = os.path.join(UCF_WEIGHTS_PATH, UCF_BITMIND_CHECKPOINT_NAME)
            bt.logging.info(f"Loading general detection model from {general_model_path}")
            self.general_model = UCF(config_path=UCF_CONFIG_PATH,
                                     weights_dir=UCF_WEIGHTS_PATH,
                                     ucf_checkpoint_name=UCF_BITMIND_CHECKPOINT_NAME,
                                     predictor_path=predictor_path)
        except Exception as e:
            bt.logging.error("Error loading general model", exc_info=True)

    async def forward(self, synapse: ImageSynapse) -> ImageSynapse:
        try:
            image_bytes = base64.b64decode(synapse.image)
            image = Image.open(io.BytesIO(image_bytes))
            faces, num_faces = self.face_model.detect_faces(image)

            if faces:
                print("Face detected")
                image_tensor = self.face_model.preprocess(image, faces=faces)
                pred = self.face_model.infer(image_tensor)
            else:
                print("No faces detected")
                image_tensor = self.general_model.preprocess(image)
                pred = self.general_model.infer(image_tensor)

            synapse.prediction = pred
        except Exception as e:
            bt.logging.error("Error performing inference", exc_info=True)

        bt.logging.info(f"PREDICTION: {synapse.prediction}")
        return synapse

if __name__ == "__main__":
    start_time = time.time()
    model = Miner()    
    latency = time.time() - start_time
    print(f"Latency for loading models: {latency:.4f} seconds")
    image_dir = '../base_miner/UCF/sample_images'  # Update this path to your image directory
    # Loop over all .jpg files in the directory
    total_time = 0.0
    num_images = 0
    
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            print(f"Processing {filename}...")

            # Read the image and encode it in base64
            with open(image_path, "rb") as image_file:
                b64_encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Create the ImageSynapse object
            synapse = ImageSynapse(image=b64_encoded_image)

            start_time = time.time()
            # Run the forward function
            result_synapse = asyncio.run(model.forward(synapse))

            end_time = time.time()
            latency = end_time - start_time
            total_time += latency
            num_images += 1

            print(f"Prediction for {filename}: {result_synapse.prediction}")
            print(f"Latency for {filename}: {latency:.4f} seconds")

    if num_images > 0:
        average_latency = total_time / num_images
        print(f"Processed {num_images} images")
        print(f"Average latency per image: {average_latency:.4f} seconds")
    else:
        print("No images processed.")
