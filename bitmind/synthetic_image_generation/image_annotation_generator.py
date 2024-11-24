# Transformer models
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Logging and progress handling
from transformers import logging as transformers_logging
from transformers.utils.logging import disable_progress_bar

from typing import Any, Dict, List, Tuple
import bittensor as bt
import PIL
import time
import torch
import gc

from bitmind.image_dataset import ImageDataset
from bitmind.synthetic_image_generation.utils import image_utils
from bitmind.constants import HUGGINGFACE_CACHE_DIR

disable_progress_bar()


class ImageAnnotationGenerator:
    """
    A class responsible for generating text annotations for images using a transformer-based image captioning model.
    It integrates text moderation to ensure the descriptions are concise and neutral.

    Attributes:
        device (torch.device): The device (CPU or GPU) on which the models are loaded.
        model_name (str): The name of the BLIP model for generating image captions.
        processor (Blip2Processor): The processor associated with the BLIP model.
        model (Blip2ForConditionalGeneration): The BLIP model used for generating image captions.
        apply_moderation (bool): Flag to determine whether text moderation should be applied to captions.
        text_moderation_model_name (str): The name of the model used for moderating text descriptions.
        text_moderation_pipeline (pipeline): A Hugging Face pipeline for text moderation.

    Methods:
        __init__(self, model_name: str, text_moderation_model_name: str, device: str = cuda, apply_moderation: bool = True):
            Initializes the ImageAnnotationGenerator with the specified model, device, and moderation settings.

        load_models(self):
            Loads the image annotation and text moderation models into memory.

        clear_gpu(self):
            Clears GPU memory to ensure that no residual data remains that could affect further operations.

        moderate_description(self, description: str, max_new_tokens: int = 80) -> str:
            Moderates the given description to make it more concise and neutral, using the text moderation model.

        generate_description(self, image: PIL.Image.Image, verbose: bool = False, max_new_tokens: int = 20) -> str:
            Generates a description for the provided image using the image captioning model.

        generate_annotation(self, image_id, dataset_name: str, image: PIL.Image.Image, original_dimensions: tuple, resize: bool, verbose: int) -> dict:
            Generates a text annotation for a given image, including handling image resizing and verbose logging.

        process_image(self, image_info: dict, dataset_name: str, image_index: int, resize: bool, verbose: int) -> Tuple[Any, float]:
            Processes a single image from a dataset to generate its annotation and measures the time taken.

        generate_annotations(self, real_image_datasets: List[ImageDataset], verbose: int = 0, max_images: int = None, resize_images: bool = False) -> Dict[str, Dict[str, Any]]:
            Generates text annotations for a batch of images from the specified datasets and calculates the average processing latency.
    """
    def __init__(
        self, model_name: str, text_moderation_model_name: str, device: str = "cuda",
        apply_moderation: bool = True
    ):
        """
        Initializes the ImageAnnotationGenerator with specific models and device settings.
        
        Args:
            model_name (str): The name of the BLIP model for generating image captions.
            text_moderation_model_name (str): The name of the model used for moderating text descriptions.
            device (str): Device to use for model inference. Defaults to "cuda".
            apply_moderation (bool): Flag to determine whether text moderation should be applied to captions.
        """
        self.device = device
        self.model_name = model_name
        self.processor = Blip2Processor.from_pretrained(
            self.model_name, cache_dir=HUGGINGFACE_CACHE_DIR
        )
        self.model = None
        
        self.apply_moderation = apply_moderation
        self.text_moderation_model_name = text_moderation_model_name
        self.text_moderation_pipeline = None
        
    def load_models(self):
        """
        Loads the necessary models for image annotation and text moderation onto the specified device.
        """
        bt.logging.info(f"Loading image annotation model {self.model_name}")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16, 
            cache_dir=HUGGINGFACE_CACHE_DIR
        )
        self.model.to(self.device)
        bt.logging.info(f"Loaded image annotation model {self.model_name}")
        bt.logging.info(f"Loading annotation moderation model {self.text_moderation_model_name}...")
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
        bt.logging.info(f"Loaded annotation moderation model {self.text_moderation_model_name}.")

    def clear_gpu(self):
        """
        Clears GPU memory by moving models back to CPU and deleting them, followed by collecting garbage.
        """
        bt.logging.debug(f"Clearing GPU memory after generating image annotation")
        self.model.to('cpu')
        del self.model
        self.model = None
        if self.text_moderation_pipeline:
            self.text_moderation_pipeline.model.to('cpu')
            del self.text_moderation_pipeline
            self.text_moderation_pipeline = None
        gc.collect()
        torch.cuda.empty_cache()

    def moderate_description(self, description: str, max_new_tokens: int = 80) -> str:
        """
        Uses the text moderation pipeline to make the description more concise and neutral.
        """
        messages = [
            {
                "role": "system",
                "content": ("[INST]You always concisely rephrase given descriptions, eliminate redundancy, "
                            "and remove all specific references to individuals by name. You do not respond with"
                            "anything other than the revised description.[/INST]")
            },
            {
                "role": "user",
                "content": description
            }
        ]
        try:
            moderated_text = self.text_moderation_pipeline(messages, max_new_tokens=max_new_tokens,
                                                           pad_token_id=self.text_moderation_pipeline.tokenizer.eos_token_id,
                                                           return_full_text=False)
            
            if isinstance(moderated_text, list):
                return moderated_text[0]['generated_text']
                bt.logging.error("Failed to return moderated text.")
            else:
                bt.logging.error("Moderated text did not return a list.")
            
            return description  # Fallback to the original description if no suitable entry is found
        except Exception as e:
            bt.logging.error(f"An error occurred during moderation: {e}", exc_info=True)
            return description  # Return the original description as a fallback

    def generate_description(self,
                             image: PIL.Image.Image,
                             verbose: bool = False,
                             max_new_tokens: int = 20) -> str:
        """
        Generates a string description for a given image by interfacing with a transformer
        model using prompt-based captioning and building conversational context.
    
        Args:
            image (PIL.Image.Image): The image for which the description is to be generated.
            verbose (bool, optional): If True, additional logging information is printed. Defaults to False.
            max_new_tokens (int, optional): The maximum number of tokens to generate for each prompt. Defaults to 20.
    
        Returns:
            str: A generated description of the image.
        """
        if not verbose:
            transformers_logging.set_verbosity_error()

        description = ""
        prompts = ["An image of", "The setting is", "The background is", "The image type/style is"]
        for i, prompt in enumerate(prompts):
            description += prompt + ' '
            inputs = self.processor(image, text=description, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens) #GPT2Tokenizer
            answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if verbose:
                bt.logging.info(f"{i}. Prompt: {prompt}")
                bt.logging.info(f"{i}. Answer: {answer}")
            
            if answer:
                # Remove any ending spaces or punctuation that is not a period
                answer = answer.rstrip(" ,;!?")
                # Add a period at the end if it's not already there
                if not answer.endswith('.'):
                    answer += '.'
                    
                description += answer + ' '
            else:
                description = description[:-len(prompt) - 1]

        if not verbose:
            transformers_logging.set_verbosity_info()

        if description.startswith(prompts[0]):
                    description = description[len(prompts[0]):]
        
        # Remove any trailing spaces and ensure the description ends with a period
        description = description.strip()
        if not description.endswith('.'):
            description += '.'
        if self.apply_moderation:
            moderated_description = self.moderate_description(description)
            return moderated_description
        return description

    def generate_annotation(
            self,
            image_id,
            dataset_name: str,
            image: PIL.Image.Image,
            original_dimensions: tuple,
            resize: bool,
            verbose: int) -> dict:
        """
        Generate a text annotation for a given image.
    
        Parameters:
        image_id (int or str): The identifier for the image within the dataset.
        dataset_name (str): The name of the dataset the image belongs to.
        image (PIL.Image.Image): The image object that requires annotation.
        original_dimensions (tuple): Original dimensions of the image as (width, height).
        resize (bool): Allow image downsizing to maximum dimensions of (1280, 1280).
        verbose (int): Verbosity level.
    
        Returns:
        dict: Dictionary containing the annotation data.
        """
        image_to_process = image.copy()
        if resize: # Downsize if dimension(s) are greater than 1280
            image_to_process = image_utils.resize_image(image_to_process, 1280, 1280)
            if verbose > 1 and image_to_process.size != image.size:
                bt.logging.info(f"Resized {image_id}: {image.size} to {image_to_process.size}")
        try:
            description = self.generate_description(image_to_process, verbose > 2)
            annotation = {
                'description': description,
                'original_dataset': dataset_name,
                'original_dimensions': f"{original_dimensions[0]}x{original_dimensions[1]}",
                'id': image_id
            }
            return annotation
        except Exception as e:
            if verbose > 1:
                bt.logging.error(f"Error processing image {image_id} in {dataset_name}: {e}")
            return None

    def process_image(
            self,
            image_info: dict,
            dataset_name: str,
            image_index: int,
            resize: bool,
            verbose: int) -> Tuple[Any, float]:
        """
        Processes an individual image for annotation, including resizing and verbosity controls, 
        and calculates the time taken to process the image.

        Args:
            image_info (dict): Dictionary containing image data and metadata.
            dataset_name (str): The name of the dataset containing the image.
            image_index (int): The index of the image within the dataset.
            resize (bool): Whether to resize the image before processing.
            verbose (int): Verbosity level for logging outputs.

        Returns:
            Tuple[Any, float]: A tuple containing the generated annotation (or None if failed) and the time taken to process.
        """

        if image_info['image'] is None:
            if verbose > 1:
                bt.logging.debug(f"Skipping image {image_index} in dataset {dataset_name} due to missing image data.")
            return None, 0

        original_dimensions = image_info['image'].size
        start_time = time.time()
        annotation = self.generate_annotation(image_index,
                                              dataset_name,
                                              image_info['image'],
                                              original_dimensions,
                                              resize,
                                              verbose)
        time_elapsed = time.time() - start_time

        if annotation is None:
            if verbose > 1:
                bt.logging.debug(f"Failed to generate annotation for image {image_index} in dataset {dataset_name}")
            return None, time_elapsed

        return annotation, time_elapsed
    
    def generate_annotations(
            self,
            real_image_datasets:
            List[ImageDataset],
            verbose: int = 0,
            max_images: int = None,
            resize_images: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Generates text annotations for images in the given datasets, saves them in a specified directory, 
        and computes the average per image latency. Returns a dictionary of new annotations and the average latency.

        Parameters:
            real_image_datasets (List[Any]): Datasets containing images.
            verbose (int): Verbosity level for process messages (Most verbose = 3).
            max_images (int): Maximum number of images to annotate.
            resize_images (bool) : Allow image downsizing before captioning.
                                Sets max dimensions to (1280, 1280), maintaining aspect ratio.
    
        Returns:
            Tuple[Dict[str, Dict[str, Any]], float]: A tuple containing the annotations dictionary and average latency.
        """
        annotations = {}
        total_time = 0
        total_processed_images = 0
        for dataset in real_image_datasets:
            dataset_name = dataset.huggingface_dataset_path
            processed_images = 0
            dataset_time = 0
            for j, image_info in enumerate(dataset):
                annotation, time_elapsed = self.process_image(image_info,
                                                              dataset_name,
                                                              j,
                                                              resize_images,
                                                              verbose)
                if annotation is not None:
                    annotations.setdefault(dataset_name, {})[image_info['id']] = annotation
                    total_time += time_elapsed
                    dataset_time += time_elapsed
                    processed_images += 1
                    if max_images is not None and len(annotations[dataset_name]) >= max_images:
                        break
            total_processed_images += processed_images
        overall_average_latency = total_time / total_processed_images if total_processed_images else 0
        return annotations, overall_average_latency
