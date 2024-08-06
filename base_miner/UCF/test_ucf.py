# Adapted from: https://github.com/SCLBD/DeepfakeBench/blob/main/training/test.py

"""
Evaluate a pretained DeepfakeBench model.
"""
import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from PIL import Image
from detectors import DETECTOR
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_DIR = "./weights/"
HUGGING_FACE_REPO_NAME = "bitmind/ucf"
UCF_CHECKPOINT_NAME = "ucf_best.pth"
BACKBONE_CHECKPOINT_NAME = "xception_best.pth"
DETECTOR_CONFIG_PATH = "./configs/ucf.yaml"
UCF_WEIGHTS_PATH = WEIGHTS_DIR + UCF_CHECKPOINT_NAME
BACKBONE_WEIGHTS_PATH = WEIGHTS_DIR + BACKBONE_CHECKPOINT_NAME

def ensure_weights_are_available(repo_name, model_filename, destination_path):
    """
    Ensures that model weights are available at the destination path.
    If not, it downloads them from the Hugging Face repository.

    Args:
    repo_name (str): Name of the Hugging Face repository.
    model_filename (str): Name of the model file in the repository.
    destination_path (str): Local path to save the model weights.
    """
    if not os.path.exists(destination_path):
        # Download the file and save it directly to the destination_path
        model_path = hf_hub_download(repo_name, model_filename)
        model = torch.load(model_path)  # Load the model weights from the downloaded path
        torch.save(model, destination_path)  # Save it to the specified path
        print(f"Downloaded {model_filename} to {destination_path}.")
    else:
        print(f"{model_filename} already present at {destination_path}.")

def load_config(detector_path):
    """ Load configuration from YAML file. """
    with open(detector_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def init_seed(seed_value):
    """ Initialize random seed for reproducibility. """
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def load_model(config, weights_path):
    """ Load the model based on configuration and weights. """
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    model.eval()
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        print('===> Load checkpoint done!')
    except FileNotFoundError:
        print('Failed to load the pretrained weights.')
    return model

def preprocess(image, device):
    """ Preprocess the image for model inference. """
    image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to Tensor, scales to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Normalizes each channel according to ImageNet statistics
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(device)


def infer(image_tensor, model):
    """ Perform inference using the model. """
    with torch.no_grad():
        predictions = model({'image': image_tensor}, inference=True)
        probability = predictions['prob'].squeeze().item()
    return probability

def main():
     # Check if the weights already exist, if not download them
    ensure_weights_are_available(HUGGING_FACE_REPO_NAME, UCF_CHECKPOINT_NAME, UCF_WEIGHTS_PATH)
    ensure_weights_are_available(HUGGING_FACE_REPO_NAME, BACKBONE_CHECKPOINT_NAME, BACKBONE_WEIGHTS_PATH)

    # parse args and load config
    config = load_config(DETECTOR_CONFIG_PATH)

    # Set manual seed for reproducibility
    if config.get('manualSeed'):
        init_seed(config['manualSeed'])

    # Enable CUDNN benchmark for performance optimization if configured
    if config.get('cudnn'):
        cudnn.benchmark = True

    model = load_model(config, UCF_WEIGHTS_PATH)
    image = Image.open("sample_images/fake.jpg")
    image_tensor = preprocess(image, device)
    probability = infer(image_tensor, model)
    print(f"Probability that the image is a deepfake: {probability}")

if __name__ == '__main__':
    main()