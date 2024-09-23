# Transformer models
from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline

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
    def __init__(
        self, model_name: str, text_moderation_model_name: str, device: str = 'auto',
        apply_moderation: bool = True
    ):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu'
        )
        
        self.model_name = model_name
        self.processor = Blip2Processor.from_pretrained(
            self.model_name, cache_dir=HUGGINGFACE_CACHE_DIR
        )
        self.model = None
        
        self.apply_moderation = apply_moderation
        self.text_moderation_model_name = text_moderation_model_name
        self.text_moderation_pipeline = None
        
    def load_models(self):
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
            self.text_moderation_pipeline = pipeline(
                "text-generation",
                model=self.text_moderation_model_name,
                model_kwargs={"torch_dtype": torch.bfloat16, "cache_dir": HUGGINGFACE_CACHE_DIR}, 
                device_map="auto"
            )
        bt.logging.info(f"Loaded annotation moderation model {self.text_moderation_model_name}.")

    def clear_gpu(self):
        bt.logging.debug(f"Clearing GPU memory after generating image annotation")
        self.model.to('cpu')
        del self.model
        if self.text_moderation_pipeline:
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
