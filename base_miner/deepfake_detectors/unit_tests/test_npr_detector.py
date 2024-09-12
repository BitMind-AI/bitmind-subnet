import unittest
import torch
from PIL import Image
import os
import sys
import numpy as np

# NPRDetector class located in the parent directory
directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(directory)
sys.path.append(parent_directory)

from npr_detector import NPRDetector

class TestNPRDetector(unittest.TestCase):
    def setUp(self):
        """Set up the necessary information to test NPRDetector."""
        self.script_dir = os.path.dirname(__file__)
        # Set the path of the sample image
        self.image_path = os.path.join(self.script_dir, 'sample_image.jpg')
        self.npr_detector = NPRDetector(weight_path=os.path.join(directory, 'base_npr_weights.pth'))

    def test_load_model(self):
        """Test if the model loads properly with the given weight path."""
        self.assertIsNotNone(self.npr_detector.model, "Model should not be None")

    def test_preprocess(self):
        """Test image preprocessing."""
        image = Image.open(self.image_path)
        tensor = self.npr_detector.preprocess(image)
        print(f"Preprocessed tensor: {tensor}")
        self.assertIsInstance(tensor, torch.Tensor, "Output should be a torch.Tensor")
        self.assertEqual(tensor.dim(), 4, "Tensor should have a dimension of 4")
        self.assertEqual(tensor.shape[1], 3, "Tensor should have 3 channels")

    def test_inference(self):
        """Test model inference on a preprocessed image."""
        image = Image.open(self.image_path)
        prediction = self.npr_detector(image)
        print(f"Prediction: {prediction}, Type: {type(prediction)}")
        self.assertIsInstance(prediction, np.ndarray, "Output should be a np.ndarray containing a float value")
        self.assertTrue(0 <= prediction <= 1, "Output should be between 0 and 1")

if __name__ == '__main__':
    unittest.main()