import os
import random
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
import yaml
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
from detectors import DETECTOR

class UCF:
    def __init__(self, config_path="./config/ucf.yaml", weights_dir="./weights/", weights_hf_repo_name="bitmind/ucf"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = Path(config_path)
        self.weights_dir = Path(weights_dir)
        self.hugging_face_repo_name = weights_hf_repo_name
        self.ucf_checkpoint_name = "ucf_best.pth"
        self.backbone_checkpoint_name = "xception_best.pth"
        self.config = self.load_config()
        self.init_cudnn()
        self.init_seed()
        self.ensure_weights_are_available(self.ucf_checkpoint_name)
        self.ensure_weights_are_available(self.backbone_checkpoint_name)
        self.model = self.load_model(self.ucf_checkpoint_name)

    def ensure_weights_are_available(self, model_filename):
        destination_path = self.weights_dir / model_filename
        if not destination_path.parent.exists():
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory {destination_path.parent}.")
        if not destination_path.exists():
            model_path = hf_hub_download(self.hugging_face_repo_name, model_filename)
            model = torch.load(model_path)
            torch.save(model, destination_path)
            logging.info(f"Downloaded {model_filename} to {destination_path}.")
        else:
            logging.info(f"{model_filename} already present at {destination_path}.")

    def load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"The config file at {self.config_path} does not exist.")
        with self.config_path.open('r') as file:
            return yaml.safe_load(file)
        
    def init_cudnn(self):
        if self.config.get('cudnn'):
            cudnn.benchmark = True

    def init_seed(self):
        seed_value = self.config.get('manualSeed')
        if seed_value:
            random.seed(seed_value)
            torch.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)

    def load_model(self, weights_filename):
        model_class = DETECTOR[self.config['model_name']]
        model = model_class(self.config).to(self.device)
        model.eval()
        weights_path = self.weights_dir / weights_filename
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(checkpoint, strict=True)
            logging.info('Loaded checkpoint successfully.')
        except FileNotFoundError:
            logging.error('Failed to load the pretrained weights.')
        return model
    
    def preprocess(self, image, res=256):
        """Preprocess the image for model inference."""
        # Ensure image is in RGB format
        image = image.convert('RGB')
        
        # Resize the image using Lanczos resampling
        image = image.resize((res, res), Image.LANCZOS)
        
        # Define the transformation pipeline
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts image to Tensor, scales to [0, 1]
            # Use ImageNet mean and std since the backbone, Xception, was pretrained on it
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
        return image_tensor.to(self.device)

    def infer(self, image_tensor):
        """ Perform inference using the model. """
        with torch.no_grad():
            self.model({'image': image_tensor}, inference=True)
        return self.model.prob[-1]