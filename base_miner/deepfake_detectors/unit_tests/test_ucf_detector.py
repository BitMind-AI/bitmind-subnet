import unittest
import os
import sys
from PIL import Image
import numpy as np

#UCFDetector class located in the parent directory
directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(directory)
sys.path.append(parent_directory)

from ucf_detector import UCFDetector

class TestUCFDetector(unittest.TestCase):
    def setUp(self):
        """Set up the necessary information to test UCFDetector."""
        # Set up a test instance of the UCFDetector class
        self.ucf_detector = UCFDetector()
        # Get the path of the current script
        self.script_dir = os.path.dirname(__file__)
        # Set the path of the sample image
        self.image_path = os.path.join(self.script_dir, 'sample_image.jpg')

    def test_load_config(self):
        """Test if the configuration is loaded properly."""
        self.assertIsNotNone(self.ucf_detector.config, "Config should not be None")

    def test_ensure_weights(self):
        """Test if the weights are checked and downloaded if missing."""
        self.assertTrue((self.ucf_detector.weights_dir / self.ucf_detector.ucf_checkpoint_name).exists(),
                        "Model weights should be available after initialization.")

    def test_model_loading(self):
        """Test if the model is loaded properly."""
        self.assertIsNotNone(self.ucf_detector.model, "Model should not be None")

    def test_infer_general(self):
        """Test a basic inference to ensure model outputs are correct."""
        self.ucf_detector = UCFDetector()
        image = Image.open(self.image_path)
        preprocessed_image = self.ucf_detector.preprocess(image)
        output = self.ucf_detector.infer(preprocessed_image)
        print(f"General Output: {output}")
        self.assertIsNotNone(output, "Inference output should not be None")
        self.assertIsInstance(output, np.ndarray, "Output should be a np.ndarray containing a float value")

    def test_infer_general_call(self):
        """Test the __call__ method to ensure inference is correct."""
        self.ucf_detector = UCFDetector()
        image = Image.open(self.image_path)
        output = self.ucf_detector(image)
        print(f"General __call__ method output: {output}")
        self.assertIsNotNone(output, "Inference output should not be None")
        self.assertIsInstance(output, np.ndarray, "Output should be a np.ndarray containing a float value")
    
    def test_face_load(self):
        """Test a basic inference to ensure model outputs are correct."""
        self.ucf_detector = UCFDetector(gate="FACE")
        self.assertIsNotNone(self.ucf_detector.gate, "Gate should not be None")
    
    def test_infer_face(self):
        """Test a basic inference to ensure model outputs are correct."""
        print("Face gate test")
        self.ucf_detector = UCFDetector(gate="FACE")
        image = Image.open(self.image_path)
        preprocessed_image = self.ucf_detector.preprocess(image)
        output = self.ucf_detector.infer(preprocessed_image)
        print(f"Face Output: {output}")
        self.assertIsNotNone(output, "Inference output should not be None")
        self.assertIsInstance(output, np.ndarray, "Output should be a np.ndarray containing a float value")

    def test_infer_face_call(self):
        """Test the __call__ method to ensure inference is correct."""
        self.ucf_detector = UCFDetector(gate="FACE")
        image = Image.open(self.image_path)
        output = self.ucf_detector(image)
        print(f"Face __call__ method output: {output}")
        self.assertIsNotNone(output, "Inference output should not be None")
        self.assertIsInstance(output, np.ndarray, "Output should be a np.ndarray containing a float value")

if __name__ == '__main__':
    unittest.main()