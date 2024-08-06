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
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from detectors import DETECTOR
import yaml
import cv2
import dlib
from preprocessing.preprocess import extract_aligned_face_dlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHAPE_PREDICTOR_PATH = './dlib_tools/shape_predictor_81_face_landmarks.dat'
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

WEIGHTS_DIR = "./weights/"
HUGGING_FACE_REPO_NAME = "bitmind/ucf"
UCF_CHECKPOINT_NAME = "ucf_best.pth"
BACKBONE_CHECKPOINT_NAME = "xception_best.pth"
DETECTOR_CONFIG_PATH = "./config/ucf.yaml"
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
    directory = os.path.dirname(destination_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory {directory}.")
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
    aligned_image, _, _ = extract_aligned_face_dlib(face_detector, shape_predictor, image)
    if aligned_image is not None:
        aligned_image.show()
    else:
        print("No face detected.")

    transform = transforms.Compose([
        transforms.resize((256, 256)),
        transforms.ToTensor(),  # Converts image to Tensor, scales to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor.to(device)

def infer(image_tensor, model):
    """ Perform inference using the model. """
    with torch.no_grad():
        model({'image': image_tensor}, inference=True)
    return model.prob[-1]

def process_images_in_folder(folder_path, model):
    """Process all images in the specified folder and predict if they are deepfakes."""
    images = [img for img in os.listdir(folder_path) if img.endswith('.jpg')]
    results = {}
    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        image = Image.open(image_path)
        image_tensor = preprocess(image, device)
        probability = infer(image_tensor, model)
        results[image_name] = probability
        rounded_prob = np.round(probability).astype(int)
        classification = 'fake' if rounded_prob == 1 else 'real'
        results[image_name] = (probability, classification)
        print(f"Probability that {image_name} is a deepfake: {probability:.4f}\
               - Classified as {classification}")
    return results

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
    folder_path = "sample_images/"
    process_images_in_folder(folder_path, model)

if __name__ == '__main__':
    main()