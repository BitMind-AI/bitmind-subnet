import unittest
import torch
import numpy as np
from PIL import Image
import os
import sys
import base64
import io
import asyncio

# Miner class located in the parent directory
directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(directory)
sys.path.append(parent_directory)

from miner import Miner
from bitmind.base.miner import BaseMinerNeuron
from bitmind.protocol import ImageSynapse

class TestMiner(unittest.TestCase):

    def setUp(self):
        """Set up the necessary components for testing the Miner."""

        self.miner = Miner.__new__(Miner) # Create an instance of the Miner class without initialization
        self.miner.config = self.miner.config()
        self.script_dir = os.path.dirname(__file__)
        self.image_path = os.path.join(self.script_dir, 'sample_image.jpg')
        
        # Load a sample image and convert it to base64 for the synapse object
        with open(self.image_path, "rb") as img_file:
            self.image = Image.open(self.image_path)
            self.image_bytes = img_file.read()
            self.image_base64 = base64.b64encode(self.image_bytes).decode('utf-8')

    def test_init_detector(self):
        """Test if the models load properly with the given weight paths."""
        self.miner.load_detector()
        print(f"Detector: {self.miner.deepfake_detector}, Type:{type(self.miner.deepfake_detector)}")
        self.assertIsNotNone(self.miner.deepfake_detector, "Detector should not be None")
    
    def test_deepfake_detector(self):
        """Test the deepfake detection functionality."""
        # Test the deepfake detector directly
        self.miner.load_detector()
        prediction = self.miner.deepfake_detector(self.image)
        print(f"Prediction: {prediction}")

        # Check that the prediction is not None and within valid range (assuming it's a probability)
        self.assertIsNotNone(prediction, "Prediction should not be None")
        self.assertIsInstance(prediction, np.ndarray, "Prediction should be a numpy array")
        self.assertTrue(0 <= prediction <= 1, "Prediction should be between 0 and 1")

    def test_forward_synapse(self):
        """Test the forward method in the Miner class using a mock ImageSynapse."""
        # Create a mock synapse object with base64 encoded image
        self.miner.load_detector()
        synapse = ImageSynapse(image=self.image_base64)
        
        # Run the detector through the synapse and verify the output prediction
        pred = asyncio.run(self.miner.forward(synapse)).prediction
        print(f"Synapse prediction: {pred}")

        self.assertIsNotNone(pred, "Prediction in the synapse should not be None")
        self.assertIsInstance(pred, float, "Synapse prediction should be a f")
        self.assertTrue(0 <= pred <= 1, "Synapse prediction should be between 0 and 1")

if __name__ == '__main__':
    unittest.main()