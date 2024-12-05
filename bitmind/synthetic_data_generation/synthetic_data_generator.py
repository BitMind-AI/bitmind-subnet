import gc
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Any, Union

import bittensor as bt
import numpy as np
import torch
from diffusers.utils import export_to_video
from PIL import Image

from bitmind.validator.config import (
    HUGGINGFACE_CACHE_DIR,
    TEXT_MODERATION_MODEL,
    IMAGE_ANNOTATION_MODEL,
    T2VIS_MODELS,
    T2VIS_MODEL_NAMES,
    T2V_MODEL_NAMES,
    T2I_MODEL_NAMES,
    TARGET_IMAGE_SIZE,
    select_random_t2vis_model,
    get_modality
)
from bitmind.synthetic_data_generation.prompt_utils import truncate_prompt_if_too_long
from bitmind.synthetic_data_generation.image_annotation_generator import ImageAnnotationGenerator
from bitmind.validator.cache import ImageCache


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
        use_random_t2vis_model: Whether to randomly select a t2v or t2i for each
            generation task.
        prompt_type: The type of prompt generation strategy ('random', 'annotation').
        prompt_generator_name: Name of the prompt generation model.
        t2vis_model_name: Name of the t2v or t2i model.
        image_annotation_generator: The generator object for annotating images if required.
        output_dir: Directory to write generated data.
    """

    def __init__(
        self,
        t2vis_model_name: Optional[str] = None,
        use_random_t2vis_model: bool = True,
        prompt_type: str = 'annotation',
        output_dir: Optional[Union[str, Path]] = None,
        image_cache: Optional[ImageCache] = None,
        device: str = 'cuda'
    ) -> None:
        """
        Initialize the SyntheticDataGenerator.

        Args:
            t2vis_model_name: Name of the text-to-video or text-to-image model.
            use_random_t2vis_model: Whether to randomly select models for generation.
            prompt_type: The type of prompt generation strategy.
            output_dir: Directory to write generated data.
            device: Device identifier.
            run_as_daemon: Whether to run generation in the background.
            image_cache: Optional image cache instance.

        Raises:
            ValueError: If an invalid model name is provided.
            NotImplementedError: If an unsupported prompt type is specified.
        """
        if not use_random_t2vis_model and t2vis_model_name not in T2VIS_MODEL_NAMES:
            raise ValueError(
                f"Invalid model name '{t2vis_model_name}'. "
                f"Options are {T2VIS_MODEL_NAMES}"
            )

        self.use_random_t2vis_model = use_random_t2vis_model
        self.t2vis_model_name = t2vis_model_name
        self.t2vis_model = None
        self.device = device

        if self.use_random_t2vis_model and t2vis_model_name is not None:
            bt.logging.warning(
                "t2vis_model_name will be ignored (use_random_t2vis_model=True)"
            )
            self.t2vis_model_name = None

        self.prompt_type = prompt_type
        if self.prompt_type == 'annotation':
            self.image_annotation_generator = ImageAnnotationGenerator(
                model_name=IMAGE_ANNOTATION_MODEL,
                text_moderation_model_name=TEXT_MODERATION_MODEL
            )
        else:
            raise NotImplementedError(f"Unsupported prompt type: {self.prompt_type}")

        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            (self.output_dir / "video").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "image").mkdir(parents=True, exist_ok=True)

        self.image_cache = image_cache

    def batch_generate(self, batch_size: int = 5) -> None:
        """
        Asynchronously generate synthetic data in batches.
        
        Args:
            batch_size: Number of prompts to generate in each batch.
        """
        prompts = []
        bt.logging.info(f"Generating {batch_size} prompts")
        for i in range(batch_size):
            image_sample = self.image_cache.sample()
            bt.logging.info(f"Sampled image {i+1}/{batch_size} for captioning: {image_sample['path']}")
            prompts.append(self.generate_prompt(image=image_sample['image'], clear_gpu=i==batch_size-1))
            bt.logging.info(f"Caption {i+1}/{batch_size} generated: {prompts[-1]}")


        # shuffle and interleave models
        t2i_model_names = random.sample(T2I_MODEL_NAMES, len(T2I_MODEL_NAMES))
        t2v_model_names = random.sample(T2V_MODEL_NAMES, len(T2V_MODEL_NAMES))
        model_names = [m for pair in zip(t2v_model_names, t2i_model_names) for m in pair]
        for model_name in model_names:
            modality = get_modality(model_name)
            for i, prompt in enumerate(prompts):
                bt.logging.info(f"Started generation {i+1}/{batch_size} | Model: {model_name} | Prompt: {prompt}")

                # Generate image/video from current model and prompt
                start = time.time()
                output = self.run_t2vis(prompt, modality, t2vis_model_name=model_name)

                bt.logging.info(f'Writing to cache {self.output_dir}')
                base_path = self.output_dir / modality / str(output['time'])
                metadata = {k: v for k, v in output.items() if k != 'gen_output'}
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
        modality: str = 'image',
        t2vis_model_name: Optional[str] = None
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
        gen_data = self.run_t2vis(prompt, modality, t2vis_model_name)
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
            self.image_annotation_generator.load_models()
            prompt = self.image_annotation_generator.generate(image)
            if clear_gpu:
                self.image_annotation_generator.clear_gpu()
        else:
            raise NotImplementedError(f"Unsupported prompt type: {self.prompt_type}")
        return prompt

    def run_t2vis(
        self,
        prompt: str,
        modality: str,
        t2vis_model_name: Optional[str] = None,        
        generate_at_target_size: bool = False,

    ) -> Dict[str, Any]:
        """
        Generate synthetic data based on a text prompt.

        Args:
            prompt: The text prompt used to inspire the generation.
            generate_at_target_size: If True, generate at TARGET_IMAGE_SIZE dimensions.
            t2vis_model_name: Optional model name to use for generation.

        Returns:
            Dictionary containing generated data and metadata.

        Raises:
            RuntimeError: If generation fails.
        """
        self.load_t2vis_model(t2vis_model_name)
        model_config = T2VIS_MODELS[self.t2vis_model_name]

        bt.logging.info("Preparing generation arguments")
        gen_args = model_config.get('generate_args', {}).copy()
        
        # Process generation arguments
        for k, v in gen_args.items():
            if isinstance(v, dict):
                gen_args[k] = np.random.randint(
                    gen_args[k]['min'],
                    gen_args[k]['max']
                )
            for dim in ('height', 'width'):
                if isinstance(gen_args.get(dim), list):
                    gen_args[dim] = np.random.choice(gen_args[dim])

        try:
            if generate_at_target_size:
                gen_args['height'] = TARGET_IMAGE_SIZE[0]
                gen_args['width'] = TARGET_IMAGE_SIZE[1]

            truncated_prompt = truncate_prompt_if_too_long(
                prompt,
                self.t2vis_model
            )

            bt.logging.info(f"Generating media from prompt: {truncated_prompt}")
            bt.logging.info(f"Generation args: {gen_args}")
            start_time = time.time()
            if model_config.get('use_autocast', True):
                pretrained_args = model_config.get('from_pretrained_args', {})
                torch_dtype = pretrained_args.get('torch_dtype', torch.bfloat16)
                with torch.autocast(self.device, torch_dtype, cache_enabled=False): 
                    gen_output = self.t2vis_model(
                        prompt=truncated_prompt,
                        **gen_args
                    )
            else:
                gen_output = self.t2vis_model(
                    prompt=truncated_prompt,
                    **gen_args
                )
            gen_time = time.time() - start_time

        except Exception as e:
            if generate_at_target_size:
                bt.logging.error(
                    f"Attempt with custom dimensions failed, falling back to "
                    f"default dimensions. Error: {e}"
                )
                try:
                    gen_output = self.t2vis_model(prompt=truncated_prompt)
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
            'model_name': self.t2vis_model_name,
            'gen_time': gen_time
        }

    def load_t2vis_model(self, model_name: Optional[str] = None, modality: Optional[str] = None) -> None:
        """Load a Hugging Face text-to-image or text-to-video model to a specific GPU."""
        if model_name is not None:
            self.t2vis_model_name = model_name
        elif self.use_random_t2vis_model or model_name == 'random':
            model_name = select_random_t2vis_model(modality)
            self.t2vis_model_name = model_name

        bt.logging.info(f"Loading {self.t2vis_model_name}")
        
        pipeline_cls = T2VIS_MODELS[model_name]['pipeline_cls']
        pipeline_args = T2VIS_MODELS[model_name]['from_pretrained_args']

        self.t2vis_model = pipeline_cls.from_pretrained(
            pipeline_args.get('base', model_name),
            cache_dir=HUGGINGFACE_CACHE_DIR,
            **pipeline_args,
            add_watermarker=False
        )

        self.t2vis_model.set_progress_bar_config(disable=True)

        # Load scheduler if specified
        if 'scheduler' in T2VIS_MODELS[model_name]:
            sched_cls = T2VIS_MODELS[model_name]['scheduler']['cls']
            sched_args = T2VIS_MODELS[model_name]['scheduler']['from_config_args']
            self.t2vis_model.scheduler = sched_cls.from_config(
                self.t2vis_model.scheduler.config,
                **sched_args
            )

        # Configure model optimizations
        model_config = T2VIS_MODELS[model_name]
        if model_config.get('enable_model_cpu_offload', False):
            bt.logging.info(f"Enabling cpu offload for {model_name}")
            self.t2vis_model.enable_model_cpu_offload()
        if model_config.get('enable_sequential_cpu_offload', False):
            bt.logging.info(f"Enabling sequential cpu offload for {model_name}")
            self.t2vis_model.enable_sequential_cpu_offload()
        if model_config.get('vae_enable_slicing', False):
            bt.logging.info(f"Enabling vae slicing for {model_name}")
            try:
                self.t2vis_model.vae.enable_slicing()
            except Exception:
                try:
                    self.t2vis_model.enable_vae_slicing()
                except Exception:
                    bt.logging.warning(f"Could not enable vae slicing for {self.t2vis_model}")
        if model_config.get('vae_enable_tiling', False):
            bt.logging.info(f"Enabling vae tiling for {model_name}")
            try:
                self.t2vis_model.vae.enable_tiling()
            except Exception:
                try:
                    self.t2vis_model.enable_vae_tiling()
                except Exception:
                    bt.logging.warning(f"Could not enable vae tiling for {self.t2vis_model}")

        self.t2vis_model.to(self.device)
        bt.logging.info(f"Loaded {model_name} using {pipeline_cls.__name__}.")

    def clear_gpu(self) -> None:
        """Clear GPU memory by deleting models and running garbage collection."""
        if self.t2vis_model is not None:
            bt.logging.info(
                "Deleting previous text-to-image or text-to-video model, "
                "freeing memory"
            )
            del self.t2vis_model
            self.t2vis_model = None
            gc.collect()
            torch.cuda.empty_cache()

