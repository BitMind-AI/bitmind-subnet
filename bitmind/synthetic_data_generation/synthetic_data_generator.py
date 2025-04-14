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
    I2V_MODEL_NAMES,
    TARGET_IMAGE_SIZE,
    select_random_model,
    get_task,
    get_modality,
    get_output_media_type,
    MediaType,
    Modality
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
            (self.output_dir / Modality.IMAGE / MediaType.SYNTHETIC).mkdir(parents=True, exist_ok=True)
            (self.output_dir / Modality.IMAGE / MediaType.SEMISYNTHETIC).mkdir(parents=True, exist_ok=True)
            (self.output_dir / Modality.VIDEO / MediaType.SYNTHETIC).mkdir(parents=True, exist_ok=True)
            (self.output_dir / Modality.VIDEO / MediaType.SEMISYNTHETIC).mkdir(parents=True, exist_ok=True)

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
            task = get_task(self.model_name) if self.model_name else None
            prompts.append(self.generate_prompt(
                image=image_sample['image'], 
                clear_gpu=i==batch_size-1,
                task=task
            ))
            bt.logging.info(f"Caption {i+1}/{batch_size} generated: {prompts[-1]}")
            
        # If specific model is set, use only that model
        if not self.use_random_model and self.model_name:
            model_names = [self.model_name]
        else:
            # shuffle and interleave models to add stochasticity
            i2i_model_names = random.sample(I2I_MODEL_NAMES, len(I2I_MODEL_NAMES))
            t2i_model_names = random.sample(T2I_MODEL_NAMES, len(T2I_MODEL_NAMES))
            t2v_model_names = random.sample(T2V_MODEL_NAMES, len(T2V_MODEL_NAMES))
            i2v_model_names = random.sample(I2V_MODEL_NAMES, len(I2V_MODEL_NAMES))
            
            model_names = [
                m for quad in zip_longest(t2v_model_names, t2i_model_names, 
                                        i2i_model_names, i2v_model_names) 
                for m in quad if m is not None
            ]

        # Generate for each model/prompt combination
        for model_name in model_names:
            modality = get_modality(model_name)
            task = get_task(model_name)
            media_type = get_output_media_type(model_name)

            for i, prompt in enumerate(prompts):
                bt.logging.info(f"Started generation {i+1}/{batch_size} | Model: {model_name} | Prompt: {prompt}")

                # Generate image/video from current model and prompt
                output = self._run_generation(prompt, task=task, model_name=model_name, image=images[i])

                model_output_dir = self.output_dir / modality / media_type / model_name.split('/')[1]
                model_output_dir.mkdir(parents=True, exist_ok=True)
                base_path = model_output_dir / str(output['time'])

                bt.logging.info(f'Writing to cache {model_output_dir}')

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
        prompt = self.generate_prompt(image, clear_gpu=True, task=task)
        bt.logging.info("Generating synthetic data...")
        gen_data = self._run_generation(prompt, task, model_name, image)
        self.clear_gpu()
        return gen_data

    def generate_prompt(
        self, 
        image: Optional[Image.Image] = None,
        clear_gpu: bool = True,
        task: Optional[str] = None
    ) -> str:
        """Generate a prompt based on the specified strategy."""
        bt.logging.info("Generating prompt")
        if self.prompt_type == 'annotation':
            if image is None:
                raise ValueError(
                    "image can't be None if self.prompt_type is 'annotation'"
                )
            self.prompt_generator.load_models()
            prompt = self.prompt_generator.generate(image, task=task)
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
            task: The generation task type ('t2i', 't2v', 'i2i', 'i2v', or None).
            model_name: Optional model name to use for generation.
            image: Optional input image for image-to-image or image-to-video generation.
            generate_at_target_size: If True, generate at TARGET_IMAGE_SIZE dimensions.

        Returns:
            Dictionary containing generated data and metadata.

        Raises:
            RuntimeError: If generation fails.
        """
        # Clear CUDA cache before loading model
        torch.cuda.empty_cache()
        gc.collect()
        
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
        # prep image-to-video generation args
        elif task == 'i2v':
            if image is None:
                raise ValueError("image cannot be None for image-to-video generation")
            # Get target size from gen_args if specified, otherwise use default
            target_size = (
                gen_args.get('height', 768),
                gen_args.get('width', 768)
            )
            if image.size[0] > target_size[0] or image.size[1] > target_size[1]:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            gen_args['image'] = image

        # Prepare generation arguments
        for k, v in gen_args.items():
            if isinstance(v, dict):
                if "min" in v and "max" in v:
                    # For i2v, use minimum values to save memory
                    if task == 'i2v':
                        gen_args[k] = v['min']
                    else:
                        gen_args[k] = np.random.randint(v['min'], v['max'])
                if "options" in v:
                    gen_args[k] = random.choice(v['options'])
            # Ensure num_frames is always an integer
            if k == 'num_frames' and isinstance(v, dict):
                if "min" in v:
                    gen_args[k] = v['min']
                elif "max" in v:
                    gen_args[k] = v['max']
                else:
                    gen_args[k] = 24  # Default value

        try:
            if generate_at_target_size:
                gen_args['height'] = TARGET_IMAGE_SIZE[0]
                gen_args['width'] = TARGET_IMAGE_SIZE[1]
            elif 'resolution' in gen_args:
                gen_args['height'] = gen_args['resolution'][0]
                gen_args['width'] = gen_args['resolution'][1]
                del gen_args['resolution']

            # Ensure num_frames is an integer before generation
            if 'num_frames' in gen_args:
                gen_args['num_frames'] = int(gen_args['num_frames'])

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
                    # Clear CUDA cache before generation
                    torch.cuda.empty_cache()
                    gc.collect()
                    gen_output = generate(truncated_prompt, **gen_args)
            else:
                # Clear CUDA cache before generation
                torch.cuda.empty_cache()
                gc.collect()
                gen_output = generate(truncated_prompt, **gen_args)
                
            gen_time = time.time() - start_time

        except Exception as e:
            if generate_at_target_size:
                bt.logging.error(
                    f"Attempt with custom dimensions failed, falling back to "
                    f"default dimensions. Error: {e}"
                )
                try:
                    # Clear CUDA cache before retry
                    torch.cuda.empty_cache()
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
        pipeline_args = model_config.get('from_pretrained_args', {}).copy()

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
            
            # Load LoRA weights if specified
            if 'lora_model_id' in model_config:
                bt.logging.info(f"Loading LoRA weights from {model_config['lora_model_id']}")
                lora_loading_args = model_config.get('lora_loading_args', {})
                self.model.load_lora_weights(
                    model_config['lora_model_id'], 
                    **lora_loading_args
                )

            # Load scheduler if specified
            if 'scheduler' in model_config:
                sched_cls = model_config['scheduler']['cls']
                sched_args = model_config['scheduler'].get('from_config_args', {})
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

    def generate_from_prompt(
        self,
        prompt: str,
        task: Optional[str] = None,
        image: Optional[Image.Image] = None,
        generate_at_target_size: bool = False
    ) -> Dict[str, Any]:
        """Generate synthetic data based on a provided prompt.
        
        Args:
            prompt: The text prompt to use for generation
            task: Optional task type ('t2i', 't2v', 'i2i', 'i2v')
            image: Optional input image for i2i or i2v generation
            generate_at_target_size: If True, generate at TARGET_IMAGE_SIZE dimensions
            
        Returns:
            Dictionary containing generated data information
        """
        bt.logging.info(f"Generating synthetic data from provided prompt: {prompt}")
        
        # Default to t2i if task is not specified
        if task is None:
            task = 't2i'
        
        # If model_name is not specified, select one based on the task
        if self.model_name is None and self.use_random_model:
            bt.logging.warning(f"No model configured. Using random model.")
            if task == 't2i':
                model_candidates = T2I_MODEL_NAMES
            elif task == 't2v':
                model_candidates = T2V_MODEL_NAMES
            elif task == 'i2i':
                model_candidates = I2I_MODEL_NAMES
            elif task == 'i2v':
                model_candidates = I2V_MODEL_NAMES
            else:
                raise ValueError(f"Unsupported task: {task}")
            
            self.model_name = random.choice(model_candidates)
        
        # Validate input image for tasks that require it
        if task in ['i2i', 'i2v'] and image is None:
            raise ValueError(f"Input image is required for {task} generation")
        
        # Run the generation with the provided prompt
        gen_data = self._run_generation(
            prompt=prompt, 
            task=task, 
            model_name=self.model_name,
            image=image,
            generate_at_target_size=generate_at_target_size
        )
        
        # Clean up GPU memory
        self.clear_gpu()
        
        return gen_data

