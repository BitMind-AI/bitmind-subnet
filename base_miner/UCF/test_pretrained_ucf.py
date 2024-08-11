import unittest
from PIL import Image
from pretrained_ucf import UCF

class TestUCF(unittest.TestCase):
    def setUp(self):
        """Set up the necessary information to test UCF."""
        # Set up a test instance of the UCF class
        self.ucf = UCF(config_path="./config/ucf.yaml", weights_dir="weights/")

    def test_load_config(self):
        """Test if the configuration is loaded properly."""
        self.assertIsNotNone(self.ucf.config, "Config should not be None")

    def test_ensure_weights(self):
        """Test if the weights are checked and downloaded if missing."""
        self.assertTrue((self.ucf.weights_dir / self.ucf.ucf_checkpoint_name).exists(),
                        "Model weights should be available after initialization.")

    def test_model_loading(self):
        """Test if the model is loaded properly."""
        self.assertIsNotNone(self.ucf.model, "Model should not be None")

    def test_infer_faces(self):
        """Test a basic inference to ensure model outputs are correct."""
        # Assuming an image is available at the specified path
        image_path = './sample_images/celebahq_0.jpg'
        image = Image.open(image_path)
        faces, num_faces = self.ucf.detect_faces(image)
        preprocessed_image = self.ucf.preprocess(image, faces=faces)
        output = self.ucf.infer(preprocessed_image)
        print(f"Face crop and align output: {output}")
        self.assertIsNotNone(output, "Inference output should not be None")

    def test_infer_general(self):
        """Test a basic inference to ensure model outputs are correct."""
        # Assuming an image is available at the specified path
        image_path = './sample_images/celebahq_0.jpg'
        image = Image.open(image_path)
        preprocessed_image = self.ucf.preprocess(image)
        output = self.ucf.infer(preprocessed_image)
        print(f"General output: {output}")
        self.assertIsNotNone(output, "Inference output should not be None")

if __name__ == '__main__':
    unittest.main()