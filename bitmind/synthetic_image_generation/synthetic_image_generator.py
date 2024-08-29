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
    DIFFUSER_NAMES,
    DIFFUSER_ARGS,
    DIFFUSER_PIPELINE,
    PROMPT_TYPES,
    IMAGE_ANNOTATION_MODEL,
    TARGET_IMAGE_SIZE
)

warnings.filterwarnings("ignore", category=FutureWarning, module='diffusers')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import pipeline, set_seed
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, DiffusionPipeline
import bittensor as bt
from bitmind.synthetic_image_generation.image_annotation_generator import ImageAnnotationGenerator


class SyntheticImageGenerator:

    def __init__(
        self,
        prompt_type='random',
        prompt_generator_name=PROMPT_GENERATOR_NAMES[0],
        diffuser_name=DIFFUSER_NAMES[0],
        use_random_diffuser=False,
        image_cache_dir=None
    ):
        if prompt_type not in PROMPT_TYPES:
            raise ValueError(f"Invalid prompt type '{prompt_type}'. Options are {PROMPT_TYPES}")
        if prompt_generator_name not in PROMPT_GENERATOR_NAMES:
            raise ValueError(f"Invalid prompt generator name '{prompt_generator_name}'. Options are {PROMPT_GENERATOR_NAMES}")
        if not use_random_diffuser and diffuser_name not in DIFFUSER_NAMES:
            raise ValueError(f"Invalid diffuser name '{diffuser_name}'. Options are {DIFFUSER_NAMES}")

        self.use_random_diffuser = use_random_diffuser
        self.prompt_type = prompt_type
        self.prompt_generator_name = prompt_generator_name

        if self.use_random_diffuser and diffuser_name is not None:
            bt.logging.warning("Warning: diffuser_name will be ignored (use_random_diffuser=True)")
            self.diffuser_name = None
        else:
            self.diffuser_name = diffuser_name

        self.image_annotation_generator = None
        if self.prompt_type == 'annotation':
            bt.logging.info(f"Loading image captioning model ({IMAGE_ANNOTATION_MODEL})...")
            self.image_annotation_generator = ImageAnnotationGenerator(model_name=IMAGE_ANNOTATION_MODEL)
        else:
            bt.logging.info(f"Loading prompt generation model ({prompt_generator_name})...")
            self.prompt_generator = pipeline(
                'text-generation', **PROMPT_GENERATOR_ARGS[prompt_generator_name])

        self.image_cache_dir = image_cache_dir
        if image_cache_dir is not None:
            os.makedirs(self.image_cache_dir, exist_ok=True)

    def generate(self, k: int = 1, real_images=None) -> list:
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
        
        if self.use_random_diffuser:
            self.load_diffuser('random')
        else:
            self.load_diffuser(self.diffuser_name)
        
        bt.logging.info("Generating images...")
        gen_data = []
        for prompt in prompts:
            image_data = self.generate_image(prompt)
            if self.image_cache_dir is not None:
                path = os.path.join(self.image_cache_dir, image_data['id'])
                image_data['image'].save(path)
            gen_data.append(image_data)
            
        self.clear_gpu()  # remove diffuser from gpu

        return gen_data

    def clear_gpu(self):
        if self.diffuser is not None:
            bt.logging.debug(f"Deleting previous diffuser, freeing memory")
            del self.diffuser
            gc.collect()
            torch.cuda.empty_cache()
            self.diffuser = None

    def load_diffuser(self, diffuser_name) -> None:
        """
        loads a huggingface diffuser model.
        """
        if diffuser_name == 'random':
            diffuser_name = np.random.choice(DIFFUSER_NAMES, 1)[0]
        
        bt.logging.info(f"Loading image generation model ({diffuser_name})...")
        self.diffuser_name = diffuser_name
        pipeline_class = globals()[DIFFUSER_PIPELINE[diffuser_name]]
        self.diffuser = pipeline_class.from_pretrained(diffuser_name,
                                                       torch_dtype=torch.float16,
                                                       **DIFFUSER_ARGS[diffuser_name],
                                                       add_watermarker=False)
        self.diffuser.set_progress_bar_config(disable=True)
        self.diffuser.to("cuda")
        print(f"Loaded {diffuser_name} using {pipeline_class.__name__}.")
        bt.logging.info(f"Loaded {diffuser_name} using {pipeline_class.__name__}.")

    def generate_image_caption(self, image_sample) -> str:
        """

        """
        self.image_annotation_generator.load_model()
        annotation = self.image_annotation_generator.process_image(
            image_info=image_sample,
            dataset_name=image_sample['source'],
            image_index=image_sample['id'],
            resize=False,
            verbose=0
        )[0]
        self.image_annotation_generator.clear_gpu()
        return annotation['description']

    def generate_random_prompt(self, retry_attempts: int = 10) -> str:
        """
        Generates a prompt for image generation.

        Args:
            retry_attempts (int): Number of attempts to generate a valid prompt.

        Returns:
            str: Generated prompt.
        """
        seed = random.randint(100, 1000000)
        set_seed(seed)

        starters = [
            'A photorealistic portrait',
            'A photorealistic image of a person',
            'A photorealistic landscape',
            'A photorealistic scene'
        ]
        quality = [
            'RAW photo', 'subject', '8k uhd',  'soft lighting', 'high quality', 'film grain'
        ]
        device = [
            'Fujifilm XT3', 'iphone', 'canon EOS r8' , 'dslr',
        ]

        for _ in range(retry_attempts):
            starting_text = np.random.choice(starters, 1)[0]
            response = self.prompt_generator(
                starting_text, max_length=(77 - len(starting_text)), num_return_sequences=1, truncation=True)

            prompt = response[0]['generated_text'].strip()
            prompt = re.sub('[^ ]+\.[^ ]+','', prompt)
            prompt = prompt.replace("<", "").replace(">", "")

            # temporary removal of extra context (like "featured on artstation") until we've trained our own prompt generator
            prompt = re.split('[,;]', prompt)[0] + ', '
            prompt += ', '.join(np.random.choice(quality, np.random.randint(len(quality)//2, len(quality))))
            prompt += ', ' + np.random.choice(device, 1)[0]
            if prompt != "":
                return prompt

    def get_tokenizer_with_min_len(self):
        """
        Returns the tokenizer with the smallest maximum token length from the 'diffuser` object.
    
        If a second tokenizer exists, it compares both and returns the one with the smaller 
        maximum token length. Otherwise, it returns the available tokenizer.
        
        Returns:
            tuple: A tuple containing the tokenizer and its maximum token length.
        """
        # Check if a second tokenizer is available in the diffuser
        if self.diffuser.tokenizer_2:
            if self.diffuser.tokenizer.model_max_length > self.diffuser.tokenizer_2.model_max_length:
                return self.diffuser.tokenizer_2, self.diffuser.tokenizer_2.model_max_length
        return self.diffuser.tokenizer, self.diffuser.tokenizer.model_max_length

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
    
    def generate_image(self, prompt, name = None, generate_at_target_size = False) -> list:
        """
        Generates an image based on a text prompt, optionally at a specified target size.
    
        Args:
            prompt (str): The text prompt used for generating the image.
            name (str, optional): The id associated with the generated image. Defaults to None.
            generate_at_target_size (bool, optional): If True, generates the image at a specified target size.
    
        Returns:
            list: A dictionary containing the prompt (truncated if over max token length), generated image, and image ID.
        """
        # Generate a unique image name based on current time if not provided
        image_name = name if name else f"{time.time():.0f}.jpg"
        # Check if the prompt is too long
        truncated_prompt = self.truncate_prompt_if_too_long(prompt)
        try:
            if generate_at_target_size:
                #Attempt to generate an image with specified dimensions
                gen_image = self.diffuser(prompt=truncated_prompt, height=TARGET_IMAGE_SIZE[0],
                                      width=TARGET_IMAGE_SIZE[1]).images[0]
            else:
                #Generate an image using default dimensions supported by the pipeline
                gen_image = self.diffuser(prompt=truncated_prompt).images[0]
        except Exception as e:
            if generate_at_target_size:
                bt.logging.warning(f"Attempt with custom dimensions failed, falling back to default dimensions. Error: {e}")
                try:
                    # Fallback to generating an image without specifying dimensions
                    gen_image = self.diffuser(prompt=truncated_prompt).images[0]
                except Exception as fallback_error:
                    bt.logging.error(f"Failed to generate image with default dimensions after initial failure: {fallback_error}")
                    raise RuntimeError(f"Both attempts to generate image failed: {fallback_error}")
            else:
                bt.logging.warning(f"Image generation error: {e}")
            
        image_data = {
            'prompt': truncated_prompt,
            'image': gen_image,
            'id': image_name
        }
        return image_data
