# Adapted from: https://github.com/SCLBD/DeepfakeBench/blob/main/training/test.py

"""
Evaluate a pretained DeepfakeBench model.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Example to ignore INFO and WARN messages
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore FutureWarnings
import io
import time
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
from pretrained_ucf import UCF

UCF_CONFIG_PATH = "./config/ucf.yaml"
UCF_WEIGHTS_PATH = "./weights/"
UCF_CHECKPOINT_NAME = "ucf_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_images_in_folder(folder_path, face_detector, predictor, model, device, config):
    """Process all images in the specified folder and predict if they are deepfakes, and record processing times."""
    images = [img for img in os.listdir(folder_path) if img.endswith('.jpg')]
    results = {}
    total_time = 0
    print(f"Processing {len(images)} images...")
    for image_name in images:
        start_time = time.time()  # Start timing for this image
        image_path = os.path.join(folder_path, image_name)
        with open(image_path, 'rb') as file:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
        #image = np.array(image)
        
        faces, num_faces = model.detect_faces(image)
        image_tensor = model.preprocess(image, faces=faces)
        
        probability = model.infer(image_tensor)
        rounded_prob = np.round(probability).astype(int)
        classification = 'fake' if rounded_prob == 1 else 'real'
        results[image_name] = (probability, classification)
        end_time = time.time()  # End timing for this image
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        print(f"Processed {image_name} in {elapsed_time:.2f} seconds - Probability: {probability:.4f}, Classified as: {classification}")

    average_time = total_time / len(images) if images else 0
    print(f"Average processing time per image: {average_time:.2f} seconds")

    return results

def main(): 
    model = UCF(config_path=UCF_CONFIG_PATH,
                weights_dir=UCF_WEIGHTS_PATH,
                ucf_checkpoint_name=UCF_CHECKPOINT_NAME)
    folder_path = "sample_images/"
    process_images_in_folder(folder_path, model.face_detector, model.face_predictor, model, model.device, model.config)

if __name__ == '__main__':
    main()