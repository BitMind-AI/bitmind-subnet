from transformers import pipeline
from diffusers import DiffusionPipeline
from transformers import set_seed
from datasets import load_dataset
import bittensor as bt
import numpy as np
import torch
import random
import time
import re
import gc
import os

from bitmind.constants import (
    PROMPT_GENERATOR_NAMES,
    PROMPT_GENERATOR_ARGS,
    DIFFUSER_NAMES,
    DIFFUSER_ARGS
)


class RandomImageGenerator:

    def __init__(
        self,
        prompt_generator_name=PROMPT_GENERATOR_NAMES[0],
        diffuser_name=DIFFUSER_NAMES[0],
        use_random_diffuser=False,
        image_cache_dir=None
    ):

        assert prompt_generator_name in PROMPT_GENERATOR_NAMES, 'invalid prompt generator name'
        assert use_random_diffuser or diffuser_name in DIFFUSER_NAMES, 'invalid diffuser name'

        if use_random_diffuser and diffuser_name is not None:
            bt.logging.warning("Warning: diffuser_name will be ignored (use_random_diffuser=True)")
            self.diffuser_name = None
        else:
            self.diffuser_name = diffuser_name

        self.use_random_diffuser = use_random_diffuser
        self.prompt_generator_name = prompt_generator_name

        self.image_cache_dir = image_cache_dir
        if image_cache_dir is not None:
            os.makedirs(self.image_cache_dir, exist_ok=True)

        bt.logging.info(f"Loading prompt generation model ({prompt_generator_name})...")
        self.prompt_generator = pipeline(
            'text-generation', **PROMPT_GENERATOR_ARGS[prompt_generator_name])

        if diffuser_name is not None:
            bt.logging.info(f"Loading image generation model ({diffuser_name})...")
            self.diffuser = DiffusionPipeline.from_pretrained(
                diffuser_name, torch_dtype=torch.float16, **DIFFUSER_ARGS[diffuser_name])
            self.diffuser.to("cuda")
        else:
            bt.logging.info("A random image generation model will be loaded on each generation step.")
            self.diffuser = None

    def generate(self, k: int = 1, annotation: dict = None) -> list:
        """
        If no annotation is provided, generates k prompts using self.prompt_generator, then passes those to self.diffuser to generate k images.
        If an annotation is provided, it uses the description field as the prompt.

        Args:
            k (int): Number of images to generate.

        Returns:
            list: List of dictionaries containing 'prompt', 'image', and 'id'.
        """
        if self.use_random_diffuser:
            self.load_random_diffuser()

        prompts = []
        if annotation and 'description' in annotation:
            bt.logging.info("Using provided annotation as prompt...")
            prompts = [annotation['description']] * k
        else:
            bt.logging.info("Generating prompts...")
            prompts = [
                self.generate_prompt()
                for _ in range(k)
            ]   

        bt.logging.info("Generating images...")
        gen_data = []
        for prompt in prompts:
            image_name = f"{time.time()}.jpg"
            gen_image = self.diffuser(prompt=prompt).images[0]
            gen_data.append({
                'prompt': prompt,
                'image': gen_image,
                'id': image_name
            })
            if self.image_cache_dir is not None:
                path = os.path.join(self.image_cache_dir, image_name)
                gen_image.save(path)

        return gen_data

    def load_random_diffuser(self) -> None:
        """
        Clears GPU memory, then loads a random diffuser model.
        """
        if self.diffuser is not None:
            bt.logging.info(f"Deleting previous diffuser, freeing memory")
            self.diffuser.to('cpu')
            del self.diffuser
            gc.collect()
            torch.cuda.empty_cache()

        diffuser_name = np.random.choice(DIFFUSER_NAMES, 1)[0]
        bt.logging.info(f"Loading image generation model ({diffuser_name})...")
        self.diffuser_name = diffuser_name
        self.diffuser = DiffusionPipeline.from_pretrained(
            diffuser_name, torch_dtype=torch.float16, **DIFFUSER_ARGS[diffuser_name])
        self.diffuser.to("cuda")

    def generate_prompt(self, retry_attempts: int = 10) -> str:
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