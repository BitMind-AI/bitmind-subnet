import gc
import time
from pathlib import Path
from typing import Dict, Optional, Any, Union, Tuple

import bittensor as bt
import numpy as np
import torch
from PIL import Image, ImageDraw

from bitmind.validator.config import (
    HUGGINGFACE_CACHE_DIR,
    TEXT_MODERATION_MODEL,
    IMAGE_ANNOTATION_MODEL,
    I2I_MODELS,
    I2I_MODEL_NAMES,
    TARGET_IMAGE_SIZE
)
from bitmind.synthetic_data_generation.prompt_utils import truncate_prompt_if_too_long
from bitmind.synthetic_data_generation.image_annotation_generator import ImageAnnotationGenerator


class InPaintingGenerator:
    """
    A class for generating image-to-image transformations using inpainting models.
    
    This class supports generating prompts from input images and applying
    inpainting transformations using various models.
    """
    
    def __init__(
        self,
        i2i_model_name: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        output_dir: Optional[Union[str, Path]] = None,
        device: str = 'cuda'
    ) -> None:
        """
        Initialize the I2IGenerator.

        Args:
            i2i_model_name: Name of the image-to-image model.
            output_dir: Directory to write generated data.
            device: Device identifier.
        """
        if i2i_model_name not in I2I_MODEL_NAMES:
            raise ValueError(
                f"Invalid model name '{i2i_model_name}'. "
                f"Options are {I2I_MODEL_NAMES}"
            )
            
        self.i2i_model_name = i2i_model_name
        self.i2i_model = None
        self.device = device
        
        self.image_annotation_generator = ImageAnnotationGenerator(
            model_name=IMAGE_ANNOTATION_MODEL,
            text_moderation_model_name=TEXT_MODERATION_MODEL
        )

    def generate(
        self,
        image: Image.Image,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an image-to-image transformation based on input image.

        Args:
            image: Input image for transformation.
            custom_prompt: Optional custom prompt to use instead of generating one.

        Returns:
            Dictionary containing generated data information.
        """
        # Resize input image to target size at the start
        image = image.resize(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        if custom_prompt is None:
            prompt = self.generate_prompt(image, clear_gpu=True)
        else:
            prompt = custom_prompt
            
        bt.logging.info("Generating i2i transformation...")
        gen_data = self.run_i2i(prompt, image)
        self.clear_gpu()
        return gen_data

    def generate_prompt(
        self, 
        image: Image.Image,
        clear_gpu: bool = True
    ) -> str:
        """Generate a prompt based on the input image."""
        bt.logging.info("Generating prompt from image")
        self.image_annotation_generator.load_models()
        prompt = self.image_annotation_generator.generate(image)
        if clear_gpu:
            self.image_annotation_generator.clear_gpu()
        return prompt

    def run_i2i(
        self,
        prompt: str,
        original_image: Image.Image,
    ) -> Dict[str, Any]:
        """
        Generate image-to-image transformation based on a text prompt.
        """
        self.load_i2i_model()
        
        original_image = original_image.convert('RGB')
        
        # Use larger image size (1024x1024 or keep original if smaller)
        target_size = (1024, 1024)
        if original_image.size[0] > target_size[0] or original_image.size[1] > target_size[1]:
            original_image = original_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Create random mask at same size as image
        mask = self.create_random_mask(original_image.size)
        
        try:
            truncated_prompt = truncate_prompt_if_too_long(prompt, self.i2i_model)
            generator = torch.Generator(device=self.device).manual_seed(0)
            
            start_time = time.time()
            gen_output = self.i2i_model(
                prompt=truncated_prompt,
                image=original_image,
                mask_image=mask,
                guidance_scale=7.5,
                num_inference_steps=50,
                strength=0.99,
                generator=generator,
            )
            gen_time = time.time() - start_time
            
            # Ensure output is in RGB mode
            output_image = gen_output.images[0]
            output_image = output_image.convert('RGB')
            gen_output.images[0] = output_image

        except Exception as e:
            bt.logging.error(f"I2I generation error: {e}")
            raise RuntimeError(f"Failed to generate i2i image: {e}")

        return {
            'prompt': truncated_prompt,
            'prompt_long': prompt,
            'gen_output': gen_output,
            'mask': mask,
            'original_image': original_image,
            'time': time.time(),
            'model_name': self.i2i_model_name,
            'gen_time': gen_time
        }

    def create_random_mask(self, size: Tuple[int, int]) -> Image.Image:
        """
        Create a random mask for i2i transformation.
        """
        w, h = size
        mask = Image.new('RGB', size, 'black')
        
        if np.random.rand() < 0.5:
            # Rectangular mask with smoother edges
            width = np.random.randint(w//4, w//2)
            height = np.random.randint(h//4, h//2)
            
            # Center the rectangle with some random offset
            x1 = (w - width) // 2 + np.random.randint(-width//4, width//4)
            y1 = (h - height) // 2 + np.random.randint(-height//4, height//4)
            
            # Create mask with PIL draw for smoother edges
            draw = ImageDraw.Draw(mask)
            draw.rounded_rectangle(
                [x1, y1, x1 + width, y1 + height],
                radius=min(width, height) // 10,  # Smooth corners
                fill='white'
            )
        else:
            # Circular mask with feathered edges
            draw = ImageDraw.Draw(mask)
            center_x = w//2
            center_y = h//2
            
            # Make radius proportional to image size
            radius = min(w, h) // 4
            
            # Add small random offset to center
            center_x += np.random.randint(-radius//4, radius//4)
            center_y += np.random.randint(-radius//4, radius//4)
            
            # Draw multiple circles with decreasing opacity for feathered edge
            for r in range(radius, radius-10, -1):
                opacity = int(255 * (r - (radius-10)) / 10)
                draw.ellipse(
                    [center_x-r, center_y-r, center_x+r, center_y+r],
                    fill=(255, 255, 255, opacity)
                )
        
        return mask

    def load_i2i_model(self) -> None:
        """Load the Hugging Face image-to-image model."""
        if self.i2i_model is not None:
            return

        bt.logging.info(f"Loading {self.i2i_model_name}")
        
        model_config = I2I_MODELS[self.i2i_model_name]
        pipeline_cls = model_config['pipeline_cls']
        pipeline_args = model_config['from_pretrained_args']
        
        self.i2i_model = pipeline_cls.from_pretrained(
            self.i2i_model_name,
            cache_dir=HUGGINGFACE_CACHE_DIR,
            **pipeline_args
        )
        
        self.i2i_model.to(self.device)
        bt.logging.info(f"Loaded {self.i2i_model_name}")

    def clear_gpu(self) -> None:
        """Clear GPU memory by deleting models and running garbage collection."""
        if self.i2i_model is not None:
            bt.logging.info("Clearing i2i model from GPU memory")
            self.i2i_model.to('cpu')
            del self.i2i_model
            self.i2i_model = None
            gc.collect()
            torch.cuda.empty_cache() 