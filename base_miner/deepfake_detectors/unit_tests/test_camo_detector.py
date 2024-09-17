import unittest
import torch
import numpy as np
from PIL import Image
import os
import sys
from camo_detector import CAMODetector


#CAMODetector class located in the parent directory
directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(directory)
sys.path.append(parent_directory)

class TestCAMODetector(unittest.TestCase):
    def setUp(self):
        """Set up the necessary information to test CAMODetector."""
        self.script_dir = os.path.dirname(__file__)
        # Set the path of the sample image
        self.image_path = os.path.join(self.script_dir, 'sample_image.jpg')
        self.camo_detector = CAMODetector()

    def test_load_model(self):
        """Test if the models load properly with the given weight paths."""
        self.assertIsNotNone(self.camo_detector.detectors['face'], "Face detector should not be None")
        self.assertIsNotNone(self.camo_detector.detectors['general'], "General detector should not be None")

    def test_load_gates(self):
        """Test if the models load properly with the given weight paths."""
        self.assertIsNotNone(self.camo_detector.gate, "Gate should not be None")
        self.assertIsNotNone(self.camo_detector.gate.gates, "GatingMechanism gates not be None")

    def test_call(self):
        """Test the __call__ method for inference on a given image."""
        image = Image.open(self.image_path)
        prediction = self.camo_detector(image)
        print(f"Prediction: {prediction}")
        self.assertIsNotNone(prediction, "Inference output should not be None")
        self.assertIsInstance(prediction, np.ndarray, "Output should be a np.ndarray containing a float value")
        self.assertTrue(0 <= prediction <= 1, "Output should be between 0 and 1")

if __name__ == '__main__':
    unittest.main()