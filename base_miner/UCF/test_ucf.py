# Adapted from: https://github.com/SCLBD/DeepfakeBench/blob/main/training/test.py

"""
Evaluate a pretained DeepfakeBench model.
"""
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
from detectors import DETECTOR
import yaml

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate a pretrained deepfake detection model.')
parser.add_argument('--detector_path', type=str, default='./configs/ucf.yaml', help='Path to detector YAML configuration file.')
parser.add_argument('--weights_path', type=str, default='./weights/ucf_best.pth', help='Path to the weights file.')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # parse args and load config
    config = load_config(args.detector_path)

    # Set manual seed for reproducibility
    if config.get('manualSeed'):
        init_seed(config['manualSeed'])

    # Enable CUDNN benchmark for performance optimization if configured
    if config.get('cudnn'):
        cudnn.benchmark = True

    model = load_model(config, args.weights_path)
    image = Image.open("test_data/fake.jpg")
    image_tensor = preprocess(image, device)
    probability = infer(image_tensor, model)
    print(f"Probability that the image is a deepfake: {probability}")

if __name__ == '__main__':
    main()