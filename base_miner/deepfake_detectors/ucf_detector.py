import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ignore INFO and WARN messages

import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import yaml
from PIL import Image
from huggingface_hub import hf_hub_download
import gc

from base_miner.UCF.config.constants import CONFIGS_DIR, WEIGHTS_DIR
from base_miner.gating_mechanisms import FaceGate

from base_miner.UCF.detectors import DETECTOR
from base_miner.deepfake_detectors import DeepfakeDetector
from base_miner import DETECTOR_REGISTRY, GATE_REGISTRY

import bittensor as bt

@DETECTOR_REGISTRY.register_module(module_name='UCF')
class UCFDetector(DeepfakeDetector):
    """
    DeepfakeDetector subclass that initializes a pretrained UCF model
    for binary classification of fake and real images.
    
    Attributes:
        model_name (str): Name of the detector instance.
        config (str): Name of the YAML file in deepfake_detectors/config/ to load
                      attributes from.
        device (str): The type of device ('cpu' or 'cuda').
    """
    
    def __init__(self, model_name: str = 'UCF', config: str = 'ucf.yaml', device: str = 'cpu'):
        super().__init__(model_name, config, device)
    
    def ensure_weights_are_available(self, weight_filename):
        destination_path = Path(WEIGHTS_DIR) / Path(weight_filename)
        if not destination_path.parent.exists():
            destination_path.parent.mkdir(parents=True, exist_ok=True)
        if not destination_path.exists():
            model_path = hf_hub_download(self.hf_repo, weight_filename)
            model = torch.load(model_path, map_location=self.device)
            torch.save(model, destination_path)

    def load_train_config(self):
        destination_path = Path(CONFIGS_DIR) / Path(self.train_config)
    
        if not destination_path.exists():
            local_config_path = hf_hub_download(self.hf_repo, self.train_config)
            print(f"Downloaded {self.hf_repo}/{self.train_config} to {local_config_path}")
            config_dict = {}
            with open(local_config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            with open(destination_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            with destination_path.open('r') as f:
                return yaml.safe_load(f)
        else:
            print(f"Loaded local config from {destination_path}")
            with destination_path.open('r') as f:
                return yaml.safe_load(f)

    def init_cudnn(self):
        if self.train_config.get('cudnn'):
            cudnn.benchmark = True

    def init_seed(self):
        seed_value = self.train_config.get('manualSeed')
        if seed_value:
            random.seed(seed_value)
            torch.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)

    def load_model(self):
        self.train_config = self.load_train_config()
        self.init_cudnn()
        self.init_seed()
        self.ensure_weights_are_available(self.weights)
        self.ensure_weights_are_available(self.train_config['pretrained'].split('/')[-1])
        model_class = DETECTOR[self.train_config['model_name']]
        bt.logging.info(f"Loaded config from training run: {self.train_config}")
        self.model = model_class(self.train_config).to(self.device)
        self.model.eval()
        weights_path = Path(WEIGHTS_DIR) / self.weights
        checkpoint = torch.load(weights_path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint, strict=True)
        except RuntimeError as e:
            if 'size mismatch' in str(e):
                # Create a custom error message
                custom_message = (
                            "\n\n Error: Incorrect specific_task_num in model config. The 'specific_task_num' "
                            "in 'config_path' yaml should match the value used during training. "
                            "A mismatch results in an incorrect output layer shape for UCF's learned disentanglement"
                            " of different forgery methods/sources.\n\n"
                            "Solution: Use the same config.yaml to intialize UCFDetector ('config_path' arg) "
                            "as output during training (config.yaml saved alongside weights in the training run's "
                            "logs directory). Or simply modify your config.yaml to ensure 'specific_task_num' equals "
                            "the value set during training (defaults to num fake training datasets + 1).\n"
                        )
                raise RuntimeError(custom_message) from e
            else: raise e
    
    def preprocess(self, image, res=256):
        """Preprocess the image for model inference.
        
        Returns:
            torch.Tensor: The preprocessed image tensor, ready for model inference.
        """
        # Convert image to RGB format to ensure consistent color handling.
        image = image.convert('RGB')
    
        # Define transformation sequence for image preprocessing.
        transform = transforms.Compose([
            transforms.Resize((res, res), interpolation=Image.LANCZOS),  # Resize image to specified resolution.
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
            transforms.Normalize(mean=self.train_config['mean'], std=self.train_config['std'])  # Normalize the image tensor.
        ])
        
        # Apply transformations and add a batch dimension for model inference.
        image_tensor = transform(image).unsqueeze(0)
        
        # Move the image tensor to the specified device (e.g., GPU).
        return image_tensor.to(self.device)

    def infer(self, image_tensor):
        """ Perform inference using the model. """
        with torch.no_grad():
            self.model({'image': image_tensor}, inference=True)
        return self.model.prob[-1]

    def __call__(self, image: Image) -> float:
        image_tensor = self.preprocess(image)
        return self.infer(image_tensor)
    
    def free_memory(self):
        """ Frees up memory by setting model and large data structures to None. """
        if self.model is not None:
            self.model.cpu()  # Move model to CPU to free up GPU memory (if applicable)
            del self.model
            self.model = None

        if self.face_detector is not None:
            del self.face_detector
            self.face_detector = None

        if self.face_predictor is not None:
            del self.face_predictor
            self.face_predictor = None

        gc.collect()

        # If using GPUs and PyTorch, clear the cache as well
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
