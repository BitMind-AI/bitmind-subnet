import gc
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Any, Union
from itertools import zip_longest

import bittensor as bt
import numpy as np
import random
import torch
from diffusers.utils import export_to_video
from PIL import Image

from bitmind.validator.config import (
    HUGGINGFACE_CACHE_DIR,
    TEXT_MODERATION_MODEL,
    IMAGE_ANNOTATION_MODEL,
    MODELS,
    MODEL_NAMES,
    T2V_MODEL_NAMES,
    T2I_MODEL_NAMES,
    I2I_MODEL_NAMES,
    TARGET_IMAGE_SIZE,
    select_random_model,
    get_task,
    get_modality
)
from bitmind.synthetic_data_generation.image_utils import create_random_mask
from bitmind.synthetic_data_generation.prompt_utils import truncate_prompt_if_too_long
from bitmind.synthetic_data_generation.prompt_generator import PromptGenerator
from bitmind.validator.cache import ImageCache
from bitmind.validator.model_utils import (
    load_hunyuanvideo_transformer,
    load_annimatediff_motion_adapter,
    JanusWrapper,
    create_pipeline_generator,
    enable_model_optimizations
)


future_warning_modules_to_ignore = [
    'diffusers',
    'transformers.tokenization_utils_base'
]

for module in future_warning_modules_to_ignore:
    warnings.filterwarnings("ignore", category=FutureWarning, module=module)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True  
torch.set_float32_matmul_precision('high')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class SyntheticDataGenerator:
    """
    A class for generating synthetic images and videos based on text prompts.

    This class supports different prompt generation strategies and can utilize
    various text-to-video (t2v) and text-to-image (t2i) models.

    Attributes:
        use_random_model: Whether to randomly select a t2v or t2i for each
            generation task.
        prompt_type: The type of prompt generation strategy ('random', 'annotation').
        prompt_generator_name: Name of the prompt generation model.
        model_name: Name of the t2v, t2i, or i2i model.
        prompt_generator: The vlm/llm pipeline for generating input prompts for t2i/t2v models
        output_dir: Directory to write generated data.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_random_model: bool = True,
        prompt_type: str = 'annotation',
        output_dir: Optional[Union[str, Path]] = None,
        image_cache: Optional[ImageCache] = None,
        device: str = 'cuda'
    ) -> None:
        """
        Initialize the SyntheticDataGenerator.

        Args:
            model_name: Name of the generative image/video model
            use_random_model: Whether to randomly select models for generation.
            prompt_type: The type of prompt generation strategy.
            output_dir: Directory to write generated data.
            device: Device identifier.
            image_cache: Optional image cache instance.

        Raises:
            ValueError: If an invalid model name is provided.
            NotImplementedError: If an unsupported prompt type is specified.
        """
        if not use_random_model and model_name not in MODEL_NAMES:
            raise ValueError(
                f"Invalid model name '{model_name}'. "
                f"Options are {MODEL_NAMES}"
            )

        self.use_random_model = use_random_model
        self.model_name = model_name
        self.model = None
        self.device = device

        if self.use_random_model and model_name is not None:
            bt.logging.warning(
                "model_name will be ignored (use_random_model=True)"
            )
            self.model_name = None

        self.prompt_type = prompt_type
        self.image_cache = image_cache
        if self.prompt_type == 'annotation' and self.image_cache is None:
            raise ValueError(f"image_cache cannot be None if prompt_type == 'annotation'")

        self.prompt_generator = PromptGenerator(
            vlm_name=IMAGE_ANNOTATION_MODEL,
            llm_name=TEXT_MODERATION_MODEL
        )

        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            (self.output_dir / "t2v").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "t2i").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "i2i").mkdir(parents=True, exist_ok=True)

    def batch_generate(self, batch_size: int = 5) -> None:
        """
        Asynchronously generate synthetic data in batches.
        
        Args:
            batch_size: Number of prompts to generate in each batch.
        """
        prompts = []
        images = []
        bt.logging.info(f"Generating {batch_size} prompts")
        
        # Generate all prompts first
        for i in range(batch_size):
            image_sample = self.image_cache.sample()
            images.append(image_sample['image'])
            bt.logging.info(f"Sampled image {i+1}/{batch_size} for captioning: {image_sample['path']}")
            prompts.append(self.generate_prompt(image=image_sample['image'], clear_gpu=i==batch_size-1))
            bt.logging.info(f"Caption {i+1}/{batch_size} generated: {prompts[-1]}")
            
        # If specific model is set, use only that model
        if not self.use_random_model and self.model_name:
            model_names = [self.model_name]
        else:
            # shuffle and interleave models to add stochasticity
            i2i_model_names = random.sample(I2I_MODEL_NAMES, len(I2I_MODEL_NAMES))
            t2i_model_names = random.sample(T2I_MODEL_NAMES, len(T2I_MODEL_NAMES))
            t2v_model_names = random.sample(T2V_MODEL_NAMES, len(T2V_MODEL_NAMES))
            model_names = [
                m for triple in zip_longest(t2v_model_names, t2i_model_names, i2i_model_names) 
                for m in triple if m is not None
            ]

        # Generate for each model/prompt combination
        for model_name in model_names:
            modality = get_modality(model_name)
            task = get_task(model_name)
            for i, prompt in enumerate(prompts):
                bt.logging.info(f"Started generation {i+1}/{batch_size} | Model: {model_name} | Prompt: {prompt}")

                # Generate image/video from current model and prompt
                output = self._run_generation(prompt, task=task, model_name=model_name, image=images[i])

                bt.logging.info(f'Writing to cache {self.output_dir}')
                base_path = self.output_dir / task / str(output['time'])
                metadata = {k: v for k, v in output.items() if k != 'gen_output' and 'image' not in k}
                base_path.with_suffix('.json').write_text(json.dumps(metadata))

                if modality == 'image':
                    out_path = base_path.with_suffix('.png')
                    output['gen_output'].images[0].save(out_path)
                elif modality == 'video':
                    bt.logging.info("Writing to cache")
                    out_path = str(base_path.with_suffix('.mp4'))
                    export_to_video(
                        output['gen_output'].frames[0],
                        out_path,
                        fps=30
                    )
                bt.logging.info(f"Wrote to {out_path}")

    def generate(
        self,
        image: Optional[Image.Image] = None,
        task: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate synthetic data based on input parameters.

        Args:
            image: Input image for annotation-based generation.
            modality: Type of media to generate ('image' or 'video').

        Returns:
            Dictionary containing generated data information.

        Raises:
            ValueError: If real_image is None when using annotation prompt type.
            NotImplementedError: If prompt type is not supported.
        """
        prompt = self.generate_prompt(image, clear_gpu=True)
        bt.logging.info("Generating synthetic data...")
        gen_data = self._run_generation(prompt, task, model_name, image)
        self.clear_gpu()
        return gen_data

    def generate_prompt(
        self, 
        image: Optional[Image.Image] = None,
        clear_gpu: bool = True
    ) -> str:
        """Generate a prompt based on the specified strategy."""
        bt.logging.info("Generating prompt")
        if self.prompt_type == 'annotation':
            if image is None:
                raise ValueError(
                    "image can't be None if self.prompt_type is 'annotation'"
                )
            self.prompt_generator.load_models()
            prompt = self.prompt_generator.generate(image)
            if clear_gpu:
                self.prompt_generator.clear_gpu()
        else:
            raise NotImplementedError(f"Unsupported prompt type: {self.prompt_type}")
        return prompt

    def _run_generation(
        self,
        prompt: str,
        task: Optional[str] = None,
        model_name: Optional[str] = None,
        image: Optional[Image.Image] = None,
        generate_at_target_size: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate synthetic data based on a text prompt.

        Args:
            prompt: The text prompt used to inspire the generation.
            task: The generation task type ('t2i', 't2v', 'i2i', or None).
            model_name: Optional model name to use for generation.
            image: Optional input image for image-to-image generation.
            generate_at_target_size: If True, generate at TARGET_IMAGE_SIZE dimensions.

        Returns:
            Dictionary containing generated data and metadata.

        Raises:
            RuntimeError: If generation fails.
        """
        self.load_model(model_name)
        model_config = MODELS[self.model_name]
        task = get_task(model_name) if task is None else task      

        bt.logging.info("Preparing generation arguments")
        gen_args = model_config.get('generate_args', {}).copy()
        mask_center = None

        # prep inpainting-specific generation args
        if task == 'i2i':
            # Use larger image size for better inpainting quality
            target_size = (1024, 1024)
            if image.size[0] > target_size[0] or image.size[1] > target_size[1]:
                image = image.resize(target_size, Image.Resampling.LANCZOS)

            gen_args['mask_image'], mask_center = create_random_mask(image.size)
            gen_args['image'] = image

        # Prepare generation arguments
        for k, v in gen_args.items():
            if isinstance(v, dict):
                if "min" in v and "max" in v:
                    gen_args[k] = np.random.randint(v['min'], v['max'])
                if "options" in v:
                    gen_args[k] = random.choice(v['options'])

        try:
            if generate_at_target_size:
                gen_args['height'] = TARGET_IMAGE_SIZE[0]
                gen_args['width'] = TARGET_IMAGE_SIZE[1]
            elif 'resolution' in gen_args:
                gen_args['height'] = gen_args['resolution'][0]
                gen_args['width'] = gen_args['resolution'][1]
                del gen_args['resolution']

            truncated_prompt = truncate_prompt_if_too_long(prompt, self.model)
            bt.logging.info(f"Generating media from prompt: {truncated_prompt}")
            bt.logging.info(f"Generation args: {gen_args}")

            start_time = time.time()
            
            # Create pipeline-specific generator
            generate = create_pipeline_generator(model_config, self.model)
            
            # Handle autocast if needed
            if model_config.get('use_autocast', True):
                pretrained_args = model_config.get('from_pretrained_args', {})
                torch_dtype = pretrained_args.get('torch_dtype', torch.bfloat16)
                with torch.autocast(self.device, torch_dtype, cache_enabled=False):
                    gen_output = generate(truncated_prompt, **gen_args)
            else:
                gen_output = generate(truncated_prompt, **gen_args)
                
            gen_time = time.time() - start_time

        except Exception as e:
            if generate_at_target_size:
                bt.logging.error(
                    f"Attempt with custom dimensions failed, falling back to "
                    f"default dimensions. Error: {e}"
                )
                try:
                    gen_output = self.model(prompt=truncated_prompt)
                    gen_time = time.time() - start_time
                except Exception as fallback_error:
                    bt.logging.error(
                        f"Failed to generate image with default dimensions after "
                        f"initial failure: {fallback_error}"
                    )
                    raise RuntimeError(
                        f"Both attempts to generate image failed: {fallback_error}"
                    )
            else:
                bt.logging.error(f"Image generation error: {e}")
                raise RuntimeError(f"Failed to generate image: {e}")

        print(f"Finished generation in {gen_time/60} minutes")
        return {
            'prompt': truncated_prompt,
            'prompt_long': prompt,
            'gen_output': gen_output,  # image or video
            'time': time.time(),
            'model_name': self.model_name,
            'gen_time': gen_time,
            'mask_image': gen_args.get('mask_image', None),
            'mask_center': mask_center,
            'image': gen_args.get('image', None)
        }

    def load_model(self, model_name: Optional[str] = None, modality: Optional[str] = None) -> None:
        """Load a Hugging Face text-to-image or text-to-video model."""
        if model_name is not None:
            self.model_name = model_name
        elif self.use_random_model or model_name == 'random':
            self.model_name = select_random_model(modality)

        bt.logging.info(f"Loading {self.model_name}")
        
        model_config = MODELS[self.model_name]
        pipeline_cls = model_config['pipeline_cls']
        pipeline_args = model_config['from_pretrained_args'].copy()

        # Handle custom loading functions passed as tuples
        for k, v in pipeline_args.items():
            if isinstance(v, tuple) and callable(v[0]):
                pipeline_args[k] = v[0](**v[1])

        # Get model_id if specified, otherwise use model_name
        model_id = pipeline_args.pop('model_id', self.model_name)

        # Handle multi-stage pipeline
        if isinstance(pipeline_cls, dict):
            self.model = {}
            for stage_name, stage_cls in pipeline_cls.items():
                stage_args = pipeline_args.get(stage_name, {})
                base_model = stage_args.get('base', model_id)
                stage_args_filtered = {k:v for k,v in stage_args.items() if k != 'base'}
                
                bt.logging.info(f"Loading {stage_name} from {base_model}")
                self.model[stage_name] = stage_cls.from_pretrained(
                    base_model,
                    cache_dir=HUGGINGFACE_CACHE_DIR,
                    **stage_args_filtered,
                    add_watermarker=False
                )
                
                enable_model_optimizations(
                    model=self.model[stage_name],
                    device=self.device,
                    enable_cpu_offload=model_config.get('enable_model_cpu_offload', False),
                    enable_sequential_cpu_offload=model_config.get('enable_sequential_cpu_offload', False),
                    enable_vae_slicing=model_config.get('vae_enable_slicing', False),
                    enable_vae_tiling=model_config.get('vae_enable_tiling', False),
                    stage_name=stage_name
                )

                # Disable watermarker
                self.model[stage_name].watermarker = None
        else:
            # Single-stage pipeline
            self.model = pipeline_cls.from_pretrained(
                model_id,
                cache_dir=HUGGINGFACE_CACHE_DIR,
                **pipeline_args,
                add_watermarker=False
            )

            # Load scheduler if specified
            if 'scheduler' in model_config:
                sched_cls = model_config['scheduler']['cls']
                sched_args = model_config['scheduler']['from_config_args']
                self.model.scheduler = sched_cls.from_config(
                    self.model.scheduler.config,
                    **sched_args
                )

            enable_model_optimizations(
                model=self.model,
                device=self.device,
                enable_cpu_offload=model_config.get('enable_model_cpu_offload', False),
                enable_sequential_cpu_offload=model_config.get('enable_sequential_cpu_offload', False),
                enable_vae_slicing=model_config.get('vae_enable_slicing', False),
                enable_vae_tiling=model_config.get('vae_enable_tiling', False)
            )

            # Disable watermarker
            self.model.watermarker = None

        bt.logging.info(f"Loaded {self.model_name}")

    def clear_gpu(self) -> None:
        """Clear GPU memory by deleting models and running garbage collection."""
        if self.model is not None:
            bt.logging.info(
                "Deleting previous text-to-image or text-to-video model, "
                "freeing memory"
            )
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()

