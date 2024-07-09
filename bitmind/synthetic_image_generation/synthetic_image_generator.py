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
    def __init__(self, annotations_dir: str, output_dir: str, device: str = 'auto'):
        self.annotations_dir = annotations_dir
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        if self.device.type == 'cpu':
            raise RuntimeError("This script requires a GPU because it uses torch.float16.")

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
        import tensorflow as tf 

        logging.basicConfig(level=logging.INFO)
        warnings.filterwarnings("ignore", category=FutureWarning, module='diffusers')

    def generate_image(self, annotation: dict, diffuser: DiffusionPipeline):
        """ Generate images from annotations using a diffuser and save to the specified directory. """
        with torch.no_grad():
            prompt = annotation['description']
            index = annotation.get('index', f"missing_index")    
            generated_image = diffuser(prompt=prompt).images[0]
            img = ToPILImage()(generated_image) if isinstance(generated_image, torch.Tensor) else generated_image
            img_filename = f"{save_dir}/{prompt[:50].replace(' ', '_').replace('/', '_').replace('\\', '_')}-{index}.png"
            return img, img_filename