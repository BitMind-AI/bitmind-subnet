from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import logging as transformers_logging
from typing import Any, Dict, List, Tuple
import bittensor as bt
import PIL
import time
import torch
import gc

from bitmind.image_dataset import ImageDataset
from bitmind.synthetic_image_generation.utils import image_utils


class ImageAnnotationGenerator:
    def __init__(self, model_name: str, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = None
        self.load_model()

    def load_model(self):
        self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.float16)
        self.model.to(self.device)

    def clear_gpu(self):
        bt.logging.debug(f"Clearing GPU memory after generating image annotation")
        self.model.to('cpu')
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def generate_description(self, image: PIL.Image.Image, verbose: bool = False) -> str:
        if not verbose:
            transformers_logging.set_verbosity_error()

        description = ""
        prompts = ["A picture of", "The setting is", "The background is", "The image type/style is"]
        for i, prompt in enumerate(prompts):
            description += prompt + ' '
            inputs = self.processor(image, text=description, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs, max_new_tokens=20)
            answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            if verbose:
                bt.logging.info(f"{i}. Prompt: {prompt}")
                bt.logging.info(f"{i}. Answer: {answer}")

            if answer:
                description += answer
            else:
                description = description[:-len(prompt) - 1]

        if not verbose:
            transformers_logging.set_verbosity_info()

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
        if resize:
            image_to_process = image_utils.resize_image(image_to_process, 1280, 1280)
            if verbose > 1 and image_to_process.size != image.size:
                bt.logging.info(f"Resized {image_id}: {image.size} to {image_to_process.size}")
    
        description = self.generate_description(image_to_process, verbose > 2)
        annotation = {
            'description': description,
            'original_dataset': dataset_name,
            'original_dimensions': f"{original_dimensions[0]}x{original_dimensions[1]}",
            'index': image_id
        }
        return annotation

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
        annotation = self.generate_annotation(image_index, dataset_name, image_info['image'], original_dimensions, resize, verbose)
        time_elapsed = time.time() - start_time

        if annotation == -1:
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
                annotation, time_elapsed = self.process_image(image_info, dataset_name, j, resize_images, verbose)
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