import gc
import os
import re
import time
import warnings
from typing import Dict, List, Optional, Any, Union

import bittensor as bt
import numpy as np
import torch
from transformers import pipeline, set_seed
from PIL import Image

from bitmind.validator.config import (
    TEXT_MODERATION_MODEL,
    IMAGE_ANNOTATION_MODEL,
    T2I_MODELS,
    T2V_MODELS,
    T2VIS_MODELS,
    T2VIS_MODEL_NAMES,
    select_random_t2vis_model,
    TARGET_IMAGE_SIZE
)
from bitmind.synthetic_data_generation.prompt_utils import truncate_prompt_if_too_long
from bitmind.synthetic_data_generation.image_annotation_generator import ImageAnnotationGenerator
from bitmind.constants import HUGGINGFACE_CACHE_DIR

future_warning_modules_to_ignore = [
    'diffusers',
    'transformers.tokenization_utils_base'
]

for module in future_warning_modules_to_ignore:
    warnings.filterwarnings("ignore", category=FutureWarning, module=module)

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
        image_cache_dir: Directory to cache generated images.
    """

    def __init__(
        self,
        t2vis_model_name: Optional[str] = None,
        use_random_t2vis_model: bool = True,
        prompt_type: str = 'annotation',
        image_cache_dir: Optional[str] = None,
        gpu_id: int = 0
    ) -> None:
        """
        Initialize the SyntheticDataGenerator.

        Args:
            t2vis_model_name: Name of the text-to-video or text-to-image model.
            use_random_t2vis_model: Whether to randomly select models for generation.
            prompt_type: The type of prompt generation strategy.
            image_cache_dir: Directory to cache generated images.
            gpu_id: GPU device identifier.

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

        self.gpu_id = gpu_id
        self.image_cache_dir = image_cache_dir
        if image_cache_dir is not None:
            os.makedirs(self.image_cache_dir, exist_ok=True)

    def generate(
        self,
        real_image: Optional[Image.Image] = None,
        modality: str = 'image'
    ) -> Dict[str, Any]:
        """
        Generate synthetic data based on input parameters.

        Args:
            real_image: Input image for annotation-based generation.
            modality: Type of media to generate ('image' or 'video').

        Returns:
            Dictionary containing generated data information.

        Raises:
            ValueError: If real_image is None when using annotation prompt type.
            NotImplementedError: If prompt type is not supported.
        """
        bt.logging.info("Generating prompts...")
        if self.prompt_type == 'annotation':
            if real_image is None:
                raise ValueError(
                    "real_image can't be None if self.prompt_type is 'annotation'"
                )
            self.image_annotation_generator.load_models()
            prompt = self.image_annotation_generator.generate(real_image)
            self.image_annotation_generator.clear_gpu()
        else:
            raise NotImplementedError(f"Unsupported prompt type: {self.prompt_type}")

        if self.use_random_t2vis_model:
            self.t2vis_model_name = select_random_t2vis_model(modality)

        bt.logging.info(f"Loading {self.t2vis_model_name}")
        self.load_t2vis_model()

        bt.logging.info("Generating synthetic data...")
        gen_data = self.run_t2vis(prompt)
        self.clear_gpu()

        if self.image_cache_dir is not None:
            path = os.path.join(self.image_cache_dir, str(gen_data['id']))
            gen_data['gen_output'].save(path)  # TODO update for video

        return gen_data

    def run_t2vis(
        self,
        prompt: str,
        generate_at_target_size: bool = False
    ) -> Dict[str, Any]:
        """
        Generate synthetic data based on a text prompt.

        Args:
            prompt: The text prompt used to inspire the generation.
            generate_at_target_size: If True, generate at TARGET_IMAGE_SIZE dimensions.

        Returns:
            Dictionary containing generated data and metadata.

        Raises:
            RuntimeError: If generation fails.
        """
        gen_args = T2VIS_MODELS[self.t2vis_model_name].get(
            'generate_args', {}).copy()
        
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

            start_time = time.time()
            gen_output = self.t2vis_model(
                prompt=truncated_prompt,
                **gen_args
            )
            gen_time = time.time() - start_time

        except Exception as e:
            if generate_at_target_size:
                bt.logging.error(
                    "Attempt with custom dimensions failed, falling back to "
                    f"default dimensions. Error: {e}"
                )
                try:
                    gen_output = self.t2vis_model(prompt=truncated_prompt)
                    gen_time = time.time() - start_time
                except Exception as fallback_error:
                    bt.logging.error(
                        "Failed to generate image with default dimensions after "
                        f"initial failure: {fallback_error}"
                    )
                    raise RuntimeError(
                        f"Both attempts to generate image failed: {fallback_error}"
                    )
            else:
                bt.logging.error(f"Image generation error: {e}")
                raise RuntimeError(f"Failed to generate image: {e}")

        return {
            'prompt': truncated_prompt,
            'prompt_long': prompt,
            'gen_output': gen_output,  # image or video
            'id': time.time(),
            'gen_time': gen_time
        }

    def load_t2vis_model(self) -> None:
        """
        Load a Hugging Face text-to-image or text-to-video model to a specific GPU.
        """
        pipeline_cls = T2VIS_MODELS[self.t2vis_model_name]['pipeline_cls']
        pipeline_args = T2VIS_MODELS[self.t2vis_model_name]['from_pretrained_args']
        self.t2vis_model = pipeline_cls.from_pretrained(
            pipeline_args.get('base', self.t2vis_model_name),
            cache_dir=HUGGINGFACE_CACHE_DIR,
            **pipeline_args,
            add_watermarker=False
        )

        self.t2vis_model.set_progress_bar_config(disable=True)

        # load scheduler if specified
        if 'scheduler' in T2VIS_MODELS[self.t2vis_model_name]:
            sched_cls = T2VIS_MODELS[self.t2vis_model_name]['scheduler']['cls']
            sched_args = T2VIS_MODELS[self.t2vis_model_name]['scheduler']['from_config_args']
            self.t2vis_model.scheduler = sched_cls.from_config(
                self.t2vis_model.scheduler.config,
                **sched_args
            )

        if T2VIS_MODELS[self.t2vis_model_name].get('cpu_offload_enabled', False):
            self.t2vis_model.enable_model_cpu_offload()
        elif not self.gpu_id:
            self.t2vis_model.to("cuda")
        elif self.gpu_id:
            self.t2vis_model.to(f"cuda:{self.gpu_id}")

        bt.logging.info(
            f"Loaded {self.t2vis_model_name} using {pipeline_cls.__name__}."
        )

    def clear_gpu(self) -> None:
        """
        Clear GPU memory by deleting the loaded text-to-image or text-to-video
        model and performing garbage collection.
        """
        if self.t2vis_model is not None:
            bt.logging.debug(
                "Deleting previous text-to-image or text-to-video model, "
                "freeing memory"
            )
            del self.t2vis_model
            gc.collect()
            torch.cuda.empty_cache()
            self.t2vis_model = None