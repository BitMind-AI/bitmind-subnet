import gc
import time
from pathlib import Path
from typing import Dict, Optional, Any, Union, Tuple
import json

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
    TARGET_IMAGE_SIZE,
    select_random_i2i_model
)
from bitmind.synthetic_data_generation.prompt_utils import truncate_prompt_if_too_long
from bitmind.synthetic_data_generation.image_annotation_generator import ImageAnnotationGenerator
from bitmind.validator.cache import ImageCache


class InPaintingGenerator:
    """
    A class for generating image-to-image transformations using inpainting models.
    
    This class supports generating prompts from input images and applying
    inpainting transformations using various models.
    """
    
    def __init__(
        self,
        i2i_model_name: Optional[str] = None,
        use_random_i2i_model: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        image_cache: Optional[ImageCache] = None,
        device: str = 'cuda'
    ) -> None:
        """
        Initialize the I2IGenerator.

        Args:
            i2i_model_name: Name of the image-to-image model.
            use_random_i2i_model: Whether to randomly select models for generation.
            output_dir: Directory to write generated data.
            image_cache: Optional image cache instance.
            device: Device identifier.
        """
        if not use_random_i2i_model and i2i_model_name not in I2I_MODEL_NAMES:
            raise ValueError(
                f"Invalid model name '{i2i_model_name}'. "
                f"Options are {I2I_MODEL_NAMES}"
            )

        self.use_random_i2i_model = use_random_i2i_model
        self.i2i_model_name = i2i_model_name
        self.i2i_model = None
        self.device = device

        if self.use_random_i2i_model and i2i_model_name is not None:
            bt.logging.warning(
                "i2i_model_name will be ignored (use_random_i2i_model=True)"
            )
            self.i2i_model_name = None

        self.image_annotation_generator = ImageAnnotationGenerator(
            model_name=IMAGE_ANNOTATION_MODEL,
            text_moderation_model_name=TEXT_MODERATION_MODEL
        )
        self.image_cache = image_cache
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            (self.output_dir / "image").mkdir(parents=True, exist_ok=True)

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
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate image-to-image transformation based on a text prompt.

        Args:
            prompt: The text prompt used to inspire the generation.
            original_image: The source image to be inpainted.
            model_name: Optional model name to use for generation.

        Returns:
            Dictionary containing generated data and metadata.

        Raises:
            RuntimeError: If generation fails.
        """
        if model_name is not None:
            self.i2i_model_name = model_name
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
            
            bt.logging.info(f"Generating inpainting from prompt: {truncated_prompt}")
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
            bt.logging.info(f"Finished generation in {gen_time/60:.2f} minutes")
            
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

    def load_i2i_model(self, model_name: Optional[str] = None) -> None:
        """Load a Hugging Face image-to-image inpainting model to a specific GPU."""
        if model_name is not None:
            self.i2i_model_name = model_name
        elif self.use_random_i2i_model or model_name == 'random':
            model_name = select_random_i2i_model()
            self.i2i_model_name = model_name

        bt.logging.info(f"Loading {self.i2i_model_name}")
        
        pipeline_cls = I2I_MODELS[self.i2i_model_name]['pipeline_cls']
        pipeline_args = I2I_MODELS[self.i2i_model_name]['from_pretrained_args']

        self.i2i_model = pipeline_cls.from_pretrained(
            pipeline_args.get('base', self.i2i_model_name),
            cache_dir=HUGGINGFACE_CACHE_DIR,
            **pipeline_args,
            add_watermarker=False
        )

        self.i2i_model.set_progress_bar_config(disable=True)

        # Load scheduler if specified
        if 'scheduler' in I2I_MODELS[self.i2i_model_name]:
            sched_cls = I2I_MODELS[self.i2i_model_name]['scheduler']['cls']
            sched_args = I2I_MODELS[self.i2i_model_name]['scheduler']['from_config_args']
            self.i2i_model.scheduler = sched_cls.from_config(
                self.i2i_model.scheduler.config,
                **sched_args
            )

        # Configure model optimizations
        model_config = I2I_MODELS[self.i2i_model_name]
        if model_config.get('enable_model_cpu_offload', False):
            bt.logging.info(f"Enabling cpu offload for {self.i2i_model_name}")
            self.i2i_model.enable_model_cpu_offload()
        if model_config.get('enable_sequential_cpu_offload', False):
            bt.logging.info(f"Enabling sequential cpu offload for {self.i2i_model_name}")
            self.i2i_model.enable_sequential_cpu_offload()
        if model_config.get('vae_enable_slicing', False):
            bt.logging.info(f"Enabling vae slicing for {self.i2i_model_name}")
            try:
                self.i2i_model.vae.enable_slicing()
            except Exception:
                try:
                    self.i2i_model.enable_vae_slicing()
                except Exception:
                    bt.logging.warning(f"Could not enable vae slicing for {self.i2i_model}")
        if model_config.get('vae_enable_tiling', False):
            bt.logging.info(f"Enabling vae tiling for {self.i2i_model_name}")
            try:
                self.i2i_model.vae.enable_tiling()
            except Exception:
                try:
                    self.i2i_model.enable_vae_tiling()
                except Exception:
                    bt.logging.warning(f"Could not enable vae tiling for {self.i2i_model}")

        self.i2i_model.to(self.device)
        bt.logging.info(f"Loaded {self.i2i_model_name} using {pipeline_cls.__name__}.")

    def clear_gpu(self) -> None:
        """Clear GPU memory by deleting models and running garbage collection."""
        if self.i2i_model is not None:
            bt.logging.info("Clearing i2i model from GPU memory")
            self.i2i_model.to('cpu')
            del self.i2i_model
            self.i2i_model = None
            gc.collect()
            torch.cuda.empty_cache() 

    def batch_generate(self, batch_size: int = 5) -> None:
        """
        Generate inpainting transformations in batches.
        
        Args:
            batch_size: Number of images to process in each batch.
        """
        prompts = []
        bt.logging.info(f"Generating {batch_size} prompts")
        for i in range(batch_size):
            image_sample = self.image_cache.sample()
            bt.logging.info(f"Sampled image {i+1}/{batch_size} for captioning: {image_sample['path']}")
            prompts.append(self.generate_prompt(image=image_sample['image'], clear_gpu=i==batch_size-1))
            bt.logging.info(f"Caption {i+1}/{batch_size} generated: {prompts[-1]}")

        # Randomly select model if enabled
        if self.use_random_i2i_model:
            model_name = select_random_i2i_model()
        else:
            model_name = self.i2i_model_name

        for i, prompt in enumerate(prompts):
            bt.logging.info(f"Started generation {i+1}/{batch_size} | Model: {model_name} | Prompt: {prompt}")
            
            # Generate inpainted image from current prompt
            start = time.time()
            image_sample = self.image_cache.sample()
            output = self.run_i2i(prompt, image_sample['image'], model_name)
            
            bt.logging.info(f'Writing to cache {self.output_dir}')
            base_path = Path(self.output_dir) / 'image' / str(output['time'])
            metadata = {k: v for k, v in output.items() if k != 'gen_output'}
            base_path.with_suffix('.json').write_text(json.dumps(metadata))
            
            out_path = base_path.with_suffix('.png')
            output['gen_output'].images[0].save(out_path)
            bt.logging.info(f"Wrote to {out_path}")