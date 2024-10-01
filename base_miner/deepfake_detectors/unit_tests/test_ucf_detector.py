import unittest
import os
import sys
from PIL import Image
import numpy as np
from pathlib import Path
from base_miner.UCF.config.constants import WEIGHTS_DIR
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
        self.ucf_detector_face = UCFDetector(config='ucf_face.yaml')
        # Set the path of the sample image
        self.image_path = os.path.join(os.path.dirname(__file__), 'sample_image.jpg')

    def test_load_config(self):
        """Test if the configuration is loaded properly."""
        self.assertIsNotNone(self.ucf_detector.train_config, "Generaliist config should not be None")
        self.assertIsNotNone(self.ucf_detector_face.train_config, "Face config should not be None")

    def test_ensure_weights(self):
        """Test if the weights are checked and downloaded if missing."""
        self.assertTrue((Path(WEIGHTS_DIR) / self.ucf_detector.weights).exists(),
                        "Model weights should be available after initialization.")
        self.assertTrue((Path(WEIGHTS_DIR) / self.ucf_detector.train_config['pretrained'].split('/')[-1]).exists(),
                        "Backbone weights should be available after initialization.")
        self.assertTrue((Path(WEIGHTS_DIR) / self.ucf_detector_face.weights).exists(),
                        "Face model weights should be available after initialization.")
        self.assertTrue((Path(WEIGHTS_DIR) / self.ucf_detector_face.train_config['pretrained'].split('/')[-1]).exists(),
                        "Face backbone weights should be available after initialization.")

    def test_model_loading(self):
        """Test if the model is loaded properly."""
        self.assertIsNotNone(self.ucf_detector.model, "Generalist model should not be None")
        self.assertIsNotNone(self.ucf_detector_face.model, "Face model should not be None")

    def test_infer_general(self):
        """Test a basic inference to ensure model outputs are correct."""
        image = Image.open(self.image_path)
        preprocessed_image = self.ucf_detector.preprocess(image)
        output = self.ucf_detector.infer(preprocessed_image)
        print(f"General Output: {output}")
        self.assertIsNotNone(output, "Inference output should not be None")
        self.assertIsInstance(output, np.ndarray, "Output should be a np.ndarray containing a float value")

    def test_infer_general_call(self):
        """Test the __call__ method to ensure inference is correct."""
        image = Image.open(self.image_path)
        output = self.ucf_detector(image)
        print(f"General __call__ method output: {output}")
        self.assertIsNotNone(output, "Inference output should not be None")
        self.assertIsInstance(output, np.ndarray, "Output should be a np.ndarray containing a float value")
    
    def test_infer_face(self):
        """Test a basic inference to ensure model outputs are correct."""
        image = Image.open(self.image_path)
        preprocessed_image = self.ucf_detector_face.preprocess(image)
        output = self.ucf_detector_face.infer(preprocessed_image)
        print(f"Face Output: {output}")
        self.assertIsNotNone(output, "Inference output should not be None")
        self.assertIsInstance(output, np.ndarray, "Output should be a np.ndarray containing a float value")

    def test_infer_face_call(self):
        """Test the __call__ method to ensure inference is correct."""
        image = Image.open(self.image_path)
        output = self.ucf_detector_face(image)
        print(f"Face __call__ method output: {output}")
        self.assertIsNotNone(output, "Inference output should not be None")
        self.assertIsInstance(output, np.ndarray, "Output should be a np.ndarray containing a float value")

if __name__ == '__main__':
    unittest.main()