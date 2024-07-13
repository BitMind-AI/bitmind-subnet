import os
import json
import gc
import logging
import time
from typing import List

import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from transformers import pipeline
from diffusers import DiffusionPipeline
import warnings

import bittensor as bt
from bitmind.constants import PROMPT_GENERATOR_NAMES, PROMPT_GENERATOR_ARGS, DIFFUSER_NAMES, DIFFUSER_ARGS

class SyntheticImageGenerator:
    def __init__(self, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        if self.device.type == 'cpu':
            raise RuntimeError("This script requires a GPU because it uses torch.float16.")

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
        import tensorflow as tf 

        logging.basicConfig(level=logging.INFO)
        warnings.filterwarnings("ignore", category=FutureWarning, module='diffusers')

    def generate_image(self, annotation: dict, diffuser: DiffusionPipeline, save_dir: str):
        """Generate images from annotations using a diffuser and save to the specified directory."""
        with torch.no_grad():
            prompt = annotation['description']
            index = annotation.get('index', "missing_index")
            generated_image = diffuser(prompt=prompt).images[0]
            img = ToPILImage()(generated_image) if isinstance(generated_image, torch.Tensor) else generated_image
            safe_prompt = prompt[:50].replace(' ', '_').replace('/', '_').replace('\\', '\\\\')
            
            img_filename = f"{save_dir}/{safe_prompt}-{index}.png"
            return img, img_filename