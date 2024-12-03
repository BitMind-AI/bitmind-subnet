import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    pipeline,
    logging as transformers_logging,
)
from transformers.utils.logging import disable_progress_bar

import bittensor as bt
from bitmind.validator.config import HUGGINGFACE_CACHE_DIR

disable_progress_bar()


class ImageAnnotationGenerator:
    """
    A class for generating and moderating image annotations using transformer models.

    This class provides functionality to generate descriptive captions for images
    using BLIP2 models and optionally moderate the generated text using a separate
    language model.
    """

    def __init__(
        self,
        model_name: str,
        text_moderation_model_name: str,
        device: str = 'cuda',
        apply_moderation: bool = True
    ) -> None:
        """
        Initialize the ImageAnnotationGenerator with specific models and device settings.

        Args:
            model_name: The name of the BLIP model for generating image captions.
            text_moderation_model_name: The name of the model used for moderating
                text descriptions.
            device: The device to use.
            apply_moderation: Flag to determine whether text moderation should be
                applied to captions.
        """
        self.model_name = model_name
        self.processor = Blip2Processor.from_pretrained(
            self.model_name,
            cache_dir=HUGGINGFACE_CACHE_DIR
        )

        self.apply_moderation = apply_moderation
        self.text_moderation_model_name = text_moderation_model_name
        self.text_moderation_pipeline = None
        self.model = None
        self.device = device

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def load_models(self) -> None:
        """
        Load the necessary models for image annotation and text moderation onto
        the specified device.
        """
        if self.is_model_loaded():
            bt.logging.warning(
                f"Image annotation model {self.model_name} is already loaded"
            )
            return

        bt.logging.info(f"Loading image annotation model {self.model_name}")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            cache_dir=HUGGINGFACE_CACHE_DIR
        )
        self.model.to(self.device)
        bt.logging.info(f"Loaded image annotation model {self.model_name}")
        bt.logging.info(
            f"Loading annotation moderation model {self.text_moderation_model_name}..."
        )
        if self.apply_moderation:
            model = AutoModelForCausalLM.from_pretrained(
                self.text_moderation_model_name,
                torch_dtype=torch.bfloat16,
                cache_dir=HUGGINGFACE_CACHE_DIR
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.text_moderation_model_name,
                cache_dir=HUGGINGFACE_CACHE_DIR
            )
            model = model.to(self.device)
            self.text_moderation_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer
            )
        bt.logging.info(
            f"Loaded annotation moderation model {self.text_moderation_model_name}."
        )

    def clear_gpu(self) -> None:
        """
        Clear GPU memory by moving models back to CPU and deleting them,
        followed by collecting garbage.
        """
        bt.logging.info("Clearing GPU memory after generating image annotation")
        self.model.to('cpu')
        del self.model
        self.model = None
        if self.text_moderation_pipeline:
            self.text_moderation_pipeline.model.to('cpu')
            del self.text_moderation_pipeline
            self.text_moderation_pipeline = None
        gc.collect()
        torch.cuda.empty_cache()

    def moderate(self, description: str, max_new_tokens: int = 80) -> str:
        """
        Use the text moderation pipeline to make the description more concise
        and neutral.

        Args:
            description: The text description to be moderated.
            max_new_tokens: Maximum number of new tokens to generate in the
                moderated text.

        Returns:
            The moderated description text, or the original description if
            moderation fails.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "[INST]You always concisely rephrase given descriptions, "
                    "eliminate redundancy, and remove all specific references to "
                    "individuals by name. You do not respond with anything other "
                    "than the revised description.[/INST]"
                )
            },
            {
                "role": "user",
                "content": description
            }
        ]
        try:
            moderated_text = self.text_moderation_pipeline(
                messages,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.text_moderation_pipeline.tokenizer.eos_token_id,
                return_full_text=False
            )

            if isinstance(moderated_text, list):
                return moderated_text[0]['generated_text']

            bt.logging.error("Moderated text did not return a list.")
            return description

        except Exception as e:
            bt.logging.error(f"An error occurred during moderation: {e}", exc_info=True)
            return description

    def generate(
        self,
        image: Image.Image,
        max_new_tokens: int = 20,
        verbose: bool = False
    ) -> str:
        """
        Generate a string description for a given image using prompt-based
        captioning and building conversational context.

        Args:
            image: The image for which the description is to be generated.
            max_new_tokens: The maximum number of tokens to generate for each
                prompt.
            verbose: If True, additional logging information is printed.

        Returns:
            A generated description of the image.
        """
        if not verbose:
            transformers_logging.set_verbosity_error()

        description = ""
        prompts = [
            "An image of",
            "The setting is",
            "The background is",
            "The image type/style is"
        ]

        for i, prompt in enumerate(prompts):
            description += prompt + ' '
            inputs = self.processor(
                image,
                text=description,
                return_tensors="pt"
            ).to(self.device, torch.float16)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )
            answer = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

            if verbose:
                bt.logging.info(f"{i}. Prompt: {prompt}")
                bt.logging.info(f"{i}. Answer: {answer}")

            if answer:
                answer = answer.rstrip(" ,;!?")
                if not answer.endswith('.'):
                    answer += '.'
                description += answer + ' '
            else:
                description = description[:-len(prompt) - 1]

        if not verbose:
            transformers_logging.set_verbosity_info()

        if description.startswith(prompts[0]):
            description = description[len(prompts[0]):]

        description = description.strip()
        if not description.endswith('.'):
            description += '.'

        if self.apply_moderation:
            moderated_description = self.moderate(description)
            return moderated_description

        return description
