import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import numpy as np
import bittensor as bt
import random
import time
import gc

from bitmind.constants import (
    HUGGINGFACE_CACHE_DIR, 
    TARGET_IMAGE_SIZE,
    PAINTER_ARGS,
    PAINTER_GENERATE_ARGS,
    PAINTER_NAMES,
    PAINTER_PIPELINE
)

class InpaintingImageGenerator:
    def __init__(self, model_name=PAINTER_NAMES[0], gpu_id=0):
        if model_name not in PAINTER_NAMES:
            raise ValueError(f"Invalid painter name '{model_name}'. Options are {PAINTER_NAMES}")
            
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.pipeline = None

    def load_model(self):
        """Loads the inpainting model to GPU"""
        if self.pipeline is None:
            bt.logging.info(f"Loading inpainting model ({self.model_name})...")
            pipeline_class = globals()[PAINTER_PIPELINE[self.model_name]]
            painter_args = PAINTER_ARGS[self.model_name].copy()
            
            self.pipeline = pipeline_class.from_pretrained(
                self.model_name,
                cache_dir=HUGGINGFACE_CACHE_DIR,
                **painter_args
            ).to(f"cuda:{self.gpu_id}")
            self.pipeline.set_progress_bar_config(disable=True)

    def clear_gpu(self):
        """Clears GPU memory"""
        if self.pipeline is not None:
            del self.pipeline
            gc.collect()
            torch.cuda.empty_cache()
            self.pipeline = None

    def generate_random_mask(self, image_size):
        """Generates a random mask for inpainting"""
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Random mask generation parameters
        num_shapes = random.randint(1, 3)
        width, height = image_size
        
        for _ in range(num_shapes):
            # Randomly choose between rectangle and ellipse
            shape_type = random.choice(['rectangle', 'ellipse'])
            
            # Generate random size (20-50% of image dimension)
            w = random.randint(int(width * 0.2), int(width * 0.5))
            h = random.randint(int(height * 0.2), int(height * 0.5))
            
            # Random position
            x = random.randint(0, width - w)
            y = random.randint(0, height - h)
            
            if shape_type == 'rectangle':
                draw.rectangle([x, y, x + w, y + h], fill=255)
            else:
                draw.ellipse([x, y, x + w, y + h], fill=255)
        
        return mask

    def process_image(self, image, prompt=None):
        """
        Processes an image using inpainting.
        
        Args:
            image (PIL.Image): Input image to be inpainted
            prompt (str, optional): Prompt to guide inpainting. If None, uses a generic prompt.
            
        Returns:
            dict: Contains the inpainted image, mask, and metadata
        """
        try:
            self.load_model()
            
            # Resize image if needed
            if image.size != TARGET_IMAGE_SIZE:
                image = image.resize(TARGET_IMAGE_SIZE)
            
            # Generate mask
            mask = self.generate_random_mask(image.size)
            
            # Default prompt if none provided
            if prompt is None:
                prompt = "Complete the image in a photorealistic style"
            
            # Get generation args for this painter
            gen_args = PAINTER_GENERATE_ARGS.get(self.model_name, {}).copy()
            
            # Generate inpainted image
            start_time = time.time()
            output = self.pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask,
                **gen_args
            ).images[0]
            gen_time = time.time() - start_time
            
            return {
                'image': output,
                'mask': mask,
                'prompt': prompt,
                'gen_time': gen_time
            }
            
        except Exception as e:
            bt.logging.error(f"Error in inpainting: {str(e)}")
            raise
        finally:
            self.clear_gpu() 