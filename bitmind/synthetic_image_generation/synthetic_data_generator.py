from transformers import set_seed
import bittensor as bt
import numpy as np
import torch
import random
import time
import re
import gc
import os
import warnings

from bitmind.constants import (
    PROMPT_GENERATOR_NAMES,
    PROMPT_GENERATOR_ARGS,
    TEXT_MODERATION_MODEL,
    T2I_MODEL_NAMES,
    T2V_MODEL_NAMES,
    MODEL_NAMES,
    MODEL_INIT_ARGS,
    MODEL_PIPELINE,
    MODEL_CPU_OFFLOAD_ENABLED,
    MODEL_VAE_ENABLE_TILING,
    GENERATE_ARGS,
    PROMPT_TYPES,
    IMAGE_ANNOTATION_MODEL,
    TARGET_IMAGE_SIZE
)

future_warning_modules_to_ignore = [
    'diffusers',
    'transformers.tokenization_utils_base'
]

for module in future_warning_modules_to_ignore:
    warnings.filterwarnings("ignore", category=FutureWarning, module=module)
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import pipeline, set_seed
import bittensor as bt

from bitmind.synthetic_image_generation.image_annotation_generator import ImageAnnotationGenerator
from bitmind.constants import HUGGINGFACE_CACHE_DIR


class SyntheticDataGenerator:
    """
    A class for generating synthetic images and videos based on text prompts. Supports different prompt generation strategies
    and can utilize various tesxt-to-video (t2v) and text-to-image (t2i) models.

    Attributes:
        use_random_t2vis_model (bool): Whether to randomly select a t2v or t2i for each generation task.
        prompt_type (str): The type of prompt generation strategy ('random', 'annotation').
        prompt_generator_name (str): Name of the prompt generation model.
        t2vis_model_name (str): Name of the t2v or t2i model
        image_annotation_generator (ImageAnnotationGenerator): The generator object for annotating images if required.
        image_cache_dir (str): Directory to cache generated images.
    """
    def __init__(
        self,
        t2vis_model_name=None,
        use_random_t2vis_model=True,
        prompt_type='random',
        prompt_generator_name=PROMPT_GENERATOR_NAMES[0],
        image_cache_dir=None,
        gpu_id=0
    ):
        if prompt_generator_name not in PROMPT_GENERATOR_NAMES:
            raise ValueError(f"Invalid prompt generator name '{prompt_generator_name}'. Options are {PROMPT_GENERATOR_NAMES}")
        if not use_random_t2vis_model and t2vis_model_name not in MODEL_NAMES:
            raise ValueError(f"Invalid text-to-image or text-to-video model name '{t2vis_model_name}'. Options are {MODEL_NAMES}")

        self.use_random_t2vis_model = use_random_t2vis_model
        self.prompt_type = prompt_type
        self.prompt_generator_name = prompt_generator_name

        self.t2vis_model_name = None
        if self.use_random_t2vis_model and t2vis_model_name is not None:
            bt.logging.warning("Warning: t2vis_model_name will be ignored (use_random_t2vis_model=True)")
            self.t2vis_model_name = None
        else:
            self.t2vis_model_name = use_random_t2vis_model

        self.image_annotation_generator = None
        if self.prompt_type == 'annotation':
            self.image_annotation_generator = ImageAnnotationGenerator(model_name=IMAGE_ANNOTATION_MODEL,
                                                                      text_moderation_model_name=TEXT_MODERATION_MODEL)
        else:
            raise NotImplementedError("Unsupported prompt_type: {self.prompt_type}")

        self.gpu_id = gpu_id
        self.image_cache_dir = image_cache_dir
        if image_cache_dir is not None:
            os.makedirs(self.image_cache_dir, exist_ok=True)

    def generate(self, k: int = 1, real_images=None, modality='image') -> list:
        """
        Generates k synthetic images. If self.prompt_type is 'annotation', a BLIP2 captioning pipeline is used
        to produce prompts by captioning real images. If self.prompt_type is 'random', an LLM is used to generate
        prompts.

        Args:
            k (int): Number of images to generate.

        Returns:
            list: List of dictionaries containing 'prompt', 'image', and 'id'.
        """
        bt.logging.info("Generating prompts...")
        if self.prompt_type == 'annotation':
            if real_images is None:
                raise ValueError(f"real_images can't be None if self.prompt_type is 'annotation'")
            prompts = [
                self.generate_image_caption(real_images[i])
                for i in range(k)
            ]
        elif self.prompt_type == 'random':
            prompts = [
                self.generate_random_prompt(retry_attempts=10)
                for _ in range(k)
            ]
        else:
            raise NotImplementedError

        if self.use_random_t2vis_model:
            self.load_t2vis_model('random', modality)
        else:
            self.load_t2vis_model(self.t2vis_model_name)

        bt.logging.info("Generating images...")
        gen_data = []
        for prompt in prompts:
            image_data = self.run_t2vis(prompt, modality)
            if self.image_cache_dir is not None:
                path = os.path.join(self.image_cache_dir, image_data['id'])
                image_data['image'].save(path)
            gen_data.append(image_data)
        self.clear_gpu()  # remove t2vis_model from gpu

        return gen_data

    def clear_gpu(self):
        """
        Clears GPU memory by deleting the loaded text-to-image or text-to-video model and performing 
        garbage collection.
        """
        if self.t2vis_model is not None:
            bt.logging.debug(f"Deleting previous text-to-image or text-to-video model, freeing memory")
            del self.t2vis_model
            gc.collect()
            torch.cuda.empty_cache()
            self.t2vis_model = None

    def load_t2vis_model(self, t2vis_model_name, modality=None) -> None:
        """
        Loads a Hugging Face text-to-image or text-to-video model model to a specific GPU.
        
        Parameters:
        t2vis_model_name (str): Name of the text-to-image or text-to-video model to load.
        gpu_index (int): Index of the GPU to use. Defaults to 0.
        """
        if t2vis_model_name == 'random':
            if modality is None or modality == 'random':
                modality = np.random.choice(['image', 'video'], 1)[0]
            if modality == 'image':
                t2vis_model_name = np.random.choice(T2I_MODEL_NAMES, 1)[0]
            elif modality == 'video':
                t2vis_model_name = np.random.choice(T2V_MODEL_NAMES, 1)[0]
            else:
                raise NotImplementedError(f"Unsupported modality: {modality}")
        
        bt.logging.info(f"Loading image generation model ({t2vis_model_name})...")
        self.t2vis_model_name = t2vis_model_name
        pipeline_class = globals()[MODEL_PIPELINE[t2vis_model_name]]
        self.t2vis_model = pipeline_class.from_pretrained(
            t2vis_model_name,
            cache_dir=HUGGINGFACE_CACHE_DIR,
            **MODEL_INIT_ARGS[t2vis_model_name],
            add_watermarker=False)
        self.t2vis_model.set_progress_bar_config(disable=True)
        if MODEL_CPU_OFFLOAD_ENABLED[t2vis_model_name]:
            self.t2vis_model.enable_model_cpu_offload()
        elif not self.gpu_id:
            self.t2vis_model.to("cuda")
        elif self.gpu_id:
            self.t2vis_model.to(f"cuda:{self.gpu_id}")
        if MODEL_VAE_ENABLE_TILING:
            self.t2vis_model.vae.enable_tiling()
            
        bt.logging.info(f"Loaded {t2vis_model_name} using {pipeline_class.__name__}.")

    def generate_image_caption(self, image_sample) -> str:
        """
        Generates a descriptive caption for a given image sample.
    
        This function takes an image sample as input, processes the image using a pre-trained
        model, and returns a generated caption describing the content of the image.
    
        Args:
            image_sample (dict): A dictionary containing information about the image to be processed.
                It includes:
                    - 'source' (str): The dataset or source name of the image.
                    - 'id' (int/str): The unique identifier of the image.
    
        Returns:
            str: A descriptive caption generated for the input image.
        """
        self.image_annotation_generator.load_models()
        annotation = self.image_annotation_generator.process_image(
            image_info=image_sample,
            dataset_name=image_sample['source'],
            image_index=image_sample['id'],
            resize=False,
            verbose=0
        )[0]
        self.image_annotation_generator.clear_gpu()
        return annotation['description']

    def get_tokenizer_with_min_len(self):
        """
        Returns the tokenizer with the smallest maximum token length from the 't2vis_model` object.
    
        If a second tokenizer exists, it compares both and returns the one with the smaller 
        maximum token length. Otherwise, it returns the available tokenizer.
        
        Returns:
            tuple: A tuple containing the tokenizer and its maximum token length.
        """
        # Check if a second tokenizer is available in the t2vis_model
        if self.t2vis_model.tokenizer_2:
            if self.t2vis_model.tokenizer.model_max_length > self.t2vis_model.tokenizer_2.model_max_length:
                return self.t2vis_model.tokenizer_2, self.t2vis_model.tokenizer_2.model_max_length
        return self.t2vis_model.tokenizer, self.t2vis_model.tokenizer.model_max_length

    def truncate_prompt_if_too_long(self, prompt: str):
        """
        Truncates the input string if it exceeds the maximum token length when tokenized.
    
        Args:
            prompt (str): The text prompt that may need to be truncated.
    
        Returns:
            str: The original prompt if within the token limit; otherwise, a truncated version of the prompt.
        """
        tokenizer, max_token_len = self.get_tokenizer_with_min_len()
        tokens = tokenizer(prompt, verbose=False) # Suppress token max exceeded warnings
        if len(tokens['input_ids']) < max_token_len:
            return prompt
        # Truncate tokens if they exceed the maximum token length, decode the tokens back to a string
        truncated_prompt = tokenizer.decode(token_ids=tokens['input_ids'][:max_token_len-1],
                                            skip_special_tokens=True)
        tokens = tokenizer(truncated_prompt)
        bt.logging.info("Truncated prompt to abide by token limit.")
        return truncated_prompt
    
    def run_t2vis(self, prompt, generate_at_target_size = False) -> list:
        """
        Generates a synthetic image based on a text prompt. This function can optionally adjust the generation args of the 
        diffusion model, such as dimensions and the number of inference steps.
        
        Args:
            prompt (str): The text prompt used to inspire the image generation.
            name (str, optional): An optional identifier for the generated image. If not provided, a timestamp-based
                identifier is used.
            generate_at_target_size (bool, optional): If True, the image is generated at the dimensions specified by the
                TARGET_IMAGE_SIZE constant. Otherwise, dimensions are selected based on the t2v or t2i's default or random settings.
        
        Returns:
            dict: A dictionary containing:
                - 'prompt': The possibly truncated version of the input prompt.
                - 'image': The generated image object.
                - 'id': The identifier of the generated image.
                - 'gen_time': The time taken to generate the image, measured from the start of the process.
        """
        # Check if the prompt is too long
        truncated_prompt = self.truncate_prompt_if_too_long(prompt)
        gen_args = {}

        # Load generation arguments based on t2vis_model settings
        if self.t2vis_model_name in GENERATE_ARGS:
            gen_args = GENERATE_ARGS[self.t2vis_model_name].copy()
            for k, v in gen_args:
                if isinstance(v, dict):
                    gen_args[k] = np.random.randint(
                        gen_args[k]['min'],
                        gen_args[k]['max'])

            for dim in ('height', 'width'):
                if isinstance(gen_args.get(dim), list):
                    gen_args[dim] = np.random.choice(gen_args[dim])

        try:
            if generate_at_target_size:
                gen_args['height'] = TARGET_IMAGE_SIZE[0]
                gen_args['width'] = TARGET_IMAGE_SIZE[1]

            start_time = time.time()
            gen_output = self.t2vis_model(prompt=truncated_prompt, **gen_args)
            gen_time = time.time() - start_time

        except Exception as e:
            if generate_at_target_size:
                bt.logging.error(f"Attempt with custom dimensions failed, falling back to default dimensions. Error: {e}")
                try:
                    gen_output = self.t2vis_model(prompt=truncated_prompt)
                    gen_time = time.time() - start_time
                except Exception as fallback_error:
                    bt.logging.error(f"Failed to generate image with default dimensions after initial failure: {fallback_error}")
                    raise RuntimeError(f"Both attempts to generate image failed: {fallback_error}")
            else:
                bt.logging.error(f"Image generation error: {e}")
                raise RuntimeError(f"Failed to generate image: {e}")
            
        return {
            'prompt': truncated_prompt,
            'gen_output': gen_output,  # image or video
            'id': time.time(),
            'gen_time': gen_time
        }
