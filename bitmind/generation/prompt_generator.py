import re
import gc
import bittensor as bt
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    pipeline,
)


class PromptGenerator:
    """
    A class for generating and moderating image annotations using transformer models.

    This class provides functionality to generate descriptive captions for images
    using BLIP2 models and optionally moderate the generated text using a separate
    language model.
    """

    def __init__(
        self,
        vlm_name: str,
        llm_name: str,
        device: str = "cuda",
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
        self.vlm_name = vlm_name
        self.llm_name = llm_name
        self.vlm_processor = None
        self.vlm = None
        self.llm = None
        self.device = device

    def load_vlm(self) -> None:
        """
        Load the vision-language model for image annotation.
        """
        bt.logging.debug(f"Loading caption generation model {self.vlm_name}")
        self.vlm_processor = Blip2Processor.from_pretrained(
            self.vlm_name, torch_dtype=torch.float32
        )
        self.vlm = Blip2ForConditionalGeneration.from_pretrained(
            self.vlm_name, torch_dtype=torch.float32
        )
        self.vlm.to(self.device)
        bt.logging.info(f"Loaded image annotation model {self.vlm_name}")

    def load_llm(self) -> None:
        """
        Load the language model for text moderation.
        """
        bt.logging.debug(f"Loading caption moderation model {self.llm_name}")
        m = re.match(r"cuda:(\d+)", self.device)
        gpu_id = int(m.group(1)) if m else 0
        llm = AutoModelForCausalLM.from_pretrained(
            self.llm_name, 
            torch_dtype=torch.bfloat16,
            device_map={"": gpu_id}
        )
        tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        self.llm = pipeline("text-generation", model=llm, tokenizer=tokenizer)
        bt.logging.info(f"Loaded caption moderation model {self.llm_name}")

    def load_models(self) -> None:
        """
        Load the necessary models for image annotation and text moderation onto
        the specified device.
        """
        if self.vlm is None:
            self.load_vlm()
        else:
            bt.logging.warning(f"vlm already loaded")

        if self.llm is None:
            self.load_llm()
        else:
            bt.logging.warning(f"llm already loaded")

    def clear_gpu(self) -> None:
        """
        Clear GPU memory by moving models back to CPU and deleting them,
        followed by collecting garbage.
        """
        bt.logging.debug("Clearing GPU memory after prompt generation")
        if self.vlm:
            del self.vlm
            self.vlm = None

        if self.llm:
            del self.llm
            self.llm = None

        gc.collect()
        torch.cuda.empty_cache()

    def generate(
        self, image: Image.Image, downstream_task: str = None, max_new_tokens: int = 20
    ) -> str:
        """
        Generate a string description for a given image using prompt-based
        captioning and building conversational context.

        Args:
            image: The image for which the description is to be generated.
            task: The generation task ('t2i', 't2v', 'i2i', 'i2v'). If video task,
                motion descriptions will be added.
            max_new_tokens: The maximum number of tokens to generate for each
                prompt.

        Returns:
            A generated description of the image.
        """
        if self.vlm is None or self.vlm_processor is None:
            self.load_vlm()

        description = ""
        prompts = [
            "An image of",
            "The setting is",
            "The background is",
            "The image type/style is",
        ]

        for i, prompt in enumerate(prompts):
            description += prompt + " "
            inputs = self.vlm_processor(
                image, text=description, return_tensors="pt"
            ).to(self.device, torch.float32)

            generated_ids = self.vlm.generate(**inputs, max_new_tokens=max_new_tokens)
            answer = self.vlm_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            bt.logging.trace(f"{i}. Prompt: {prompt}")
            bt.logging.trace(f"{i}. Answer: {answer}")

            if answer:
                answer = answer.rstrip(" ,;!?")
                if not answer.endswith("."):
                    answer += "."
                description += answer + " "
            else:
                description = description[: -len(prompt) - 1]

        if description.startswith(prompts[0]):
            description = description[len(prompts[0]) :]

        description = description.strip()
        if not description.endswith("."):
            description += "."

        moderated_description = self.moderate(description)

        if downstream_task in ["t2v", "i2v"]:
            return self.enhance(moderated_description)
        return moderated_description

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
        if self.llm is None:
            self.load_llm()

        messages = [
            {
                "role": "system",
                "content": (
                    "[INST]You always concisely rephrase given descriptions, "
                    "eliminate redundancy, and remove all specific references to "
                    "individuals by name. You do not respond with anything other "
                    "than the revised description.[/INST]"
                ),
            },
            {"role": "user", "content": description},
        ]
        try:
            moderated_text = self.llm(
                messages,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llm.tokenizer.eos_token_id,
                return_full_text=False,
            )
            return moderated_text[0]["generated_text"]

        except Exception as e:
            bt.logging.error(f"An error occurred during moderation: {e}", exc_info=True)
            return description

    def enhance(self, description: str, max_new_tokens: int = 80) -> str:
        """
        Enhance a static image description to make it suitable for video generation
        by adding dynamic elements and motion.

        Args:
            description: The static image description to enhance.
            max_new_tokens: Maximum number of new tokens to generate in the enhanced text.

        Returns:
            An enhanced description suitable for video generation, or the original
            description if enhancement fails.
        """
        if self.llm is None:
            self.load_llm()

        messages = [
            {
                "role": "system",
                "content": (
                    "[INST]You are an expert at converting image descriptions into video prompts. "
                    "Analyze the existing motion in the scene and enhance it naturally:\n"
                    "1. If motion exists in the image (falling, throwing, running, etc.):\n"
                    "   - Maintain and emphasize that existing motion\n"
                    "   - Add smooth continuation of the movement\n"
                    "2. If the subject is static (sitting, standing, placed):\n"
                    "   - Keep it stable\n"
                    "   - Add minimal environmental motion if appropriate\n"
                    "3. Add ONE subtle camera motion that complements the scene\n"
                    "4. Keep the description concise and natural\n"
                    "Only respond with the enhanced description.[/INST]"
                ),
            },
            {"role": "user", "content": description},
        ]

        try:
            enhanced_text = self.llm(
                messages,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llm.tokenizer.eos_token_id,
                return_full_text=False,
            )
            return enhanced_text[0]["generated_text"]

        except Exception as e:
            bt.logging.error(f"An error occurred during motion enhancement: {e}")
            return description

    def sanitize(self, prompt: str, max_new_tokens: int = 80) -> str:
        """
        Use the LLM to make the prompt more SFW (less NSFW).
        """

        if self.llm is None:
            self.load_llm()

        messages = [
            {
                "role": "system",
                "content": (
                    "[INST]You are an expert at making prompts safe for work (SFW). "
                    "Rephrase the following prompt to remove or neutralize any NSFW, sexual, or explicit content. "
                    "Keep the prompt as close as possible to the original intent, but ensure it is SFW. "
                    "Only respond with the sanitized prompt.[/INST]"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            sanitized = self.llm(
                messages,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llm.tokenizer.eos_token_id,
                return_full_text=False,
            )
            return sanitized[0]["generated_text"]
        except Exception as e:
            bt.logging.error(f"An error occurred during prompt sanitization: {e}")
            return prompt
