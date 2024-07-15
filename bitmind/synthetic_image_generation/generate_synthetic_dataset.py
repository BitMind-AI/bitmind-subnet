%%time

# Standard library imports
import sys
import gc
import json
import logging
import os
import random
import time
from multiprocessing import Pool, cpu_count, current_process, get_context, Manager
from typing import Any, Dict, List, Tuple

# Third-party library imports
import numpy as np
import PIL
import torch
import warnings
from diffusers import DiffusionPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from IPython.display import display
from PIL import Image
from torchvision.transforms import ToPILImage
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipProcessor, logging as transformers_logging, pipeline

# Suppress TensorFlow logging before import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 1: filter out INFO, 2: filter out WARNING, 3: filter out ERROR
import tensorflow as tf

# Local/application-specific imports
import bittensor as bt
from bitmind.constants import DATASET_META, IMAGE_ANNOTATION_MODEL, PROMPT_GENERATOR_ARGS, PROMPT_GENERATOR_NAMES, DIFFUSER_ARGS, DIFFUSER_NAMES
from bitmind.image_dataset import ImageDataset
from utils import image_utils
from multiprocessing_tasks import generate_images_for_chunk, worker_initializer

%%time
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

#Default log settings
transformers_level = logging.getLogger("transformers").getEffectiveLevel()
huggingface_hub_level = logging.getLogger("huggingface_hub").getEffectiveLevel()

#Suppress logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
processor = AutoProcessor.from_pretrained(IMAGE_ANNOTATION_MODEL)
# by default `from_pretrained` loads the weights in float32
# we load in float16 instead to save memory
model = Blip2ForConditionalGeneration.from_pretrained(IMAGE_ANNOTATION_MODEL, torch_dtype=torch.float16) 
model.to(device)

#Restore log settings
logging.getLogger("transformers").setLevel(transformers_level)
logging.getLogger("huggingface_hub").setLevel(huggingface_hub_level)

### Real Image to Annotation

prompts = [
    "A picture of",
    "The setting is",
    "The background is",
    "The image type/style is"
    # "the background is",
    # "The color(s) are",
    # "The texture(s) are",
    # "The emotion/mood is",
    # "The image medium is",
    # "The image style is"
]

def generate_description(image: PIL.Image.Image, use_prompts: bool = True, verbose: bool = False) -> str:
    """
    Generates a description for a given image using a sequence of prompts.

    Parameters:
    image (PIL.Image.Image): The image for which to generate a text description (string).
    use_prompts (bool) : Determines whether to use prompt input for BLIP-2 image-to-text generation.
    verbose (bool): If True, prints the prompts and answers during processing. Defaults to False.

    Returns:
    str: The generated description for the image.
    """
    if not verbose: transformers_logging.set_verbosity_error() # Only display error messages (no warnings)
    description = ""
    if not use_prompts:
        inputs = processor(image, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        description += generated_text
    else:
        for i, prompt in enumerate(prompts):
            # Append prompt to description to build context history
            description += prompt + ' '
            inputs = processor(image, text=description, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if answer:
                # Append answer to description to build context history
                description += answer
            else:
                description = description[:-len(prompt) - 1]  # Remove the last prompt if no answer is generated
            if verbose:
                print(f"{i}. Prompt: {prompt}")
                print(f"{i}. Answer: {answer}")
    if not verbose: transformers_logging.set_verbosity_info() # Restore transformer warnings
    return description

# Helper functions

def set_logging_level(verbose: int):
    level = logging.WARNING if verbose == 0 else logging.INFO if verbose < 3 else logging.DEBUG
    logging.getLogger().setLevel(level)

def ensure_save_path(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def create_annotation_dataset_directory(base_path: str, dataset_name: str) -> str:
    safe_name = dataset_name.replace("/", "_")
    full_path = os.path.join(base_path, safe_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

def generate_annotation(image_id,
                        dataset_name: str,
                        image: PIL.Image.Image,
                        original_dimensions: tuple,
                        resize: bool,
                        resize_dim: int,
                        use_prompts: bool,
                        verbose: int):
    """
    Generate a text annotation for a given image.

    Parameters:
    image_id (int or str): The identifier for the image within the dataset.
    dataset_name (str): The name of the dataset the image belongs to.
    image (PIL.Image.Image): The image object that requires annotation.
    original_dimensions (tuple): Original dimensions of the image as (width, height).
    resize (bool): Allow image downsizing to maximum dimensions of (1280, 1280).
    verbose (int): Verbosity level. If greater than 1, prints image resize message.

    Returns:
    dict: Dictionary containing the annotation data.
    """
    image_to_process = image.copy()
    if resize:
        image_to_process = image_utils.resize_image(image_to_process, resize_dim, resize_dim)
        if verbose > 1 and image_to_process.size != image.size:
            print(f"Resized {image_id}: {image.size} to {image_to_process.size}")

    description = generate_description(image_to_process, use_prompts, verbose > 2)
    annotation = {
        'description': description,
        'original_dataset': dataset_name,
        'original_dimensions': f"{original_dimensions[0]}x{original_dimensions[1]}",
        'index': image_id
    }
    return annotation

def save_annotation(dataset_dir: str, image_id, annotation: dict, verbose: int):
    """
    Save a text annotation to a JSON file if it doesn't already exist.

    Parameters:
    dataset_dir (str): The directory where the annotation file will be saved.
    image_id (int or str): The identifier for the image within the dataset.
    annotation (dict): Annotation data to be saved.
    verbose (int): Verbosity level. If greater than 0, it prints messages during processing.

    Returns:
    int: Returns 0 if the annotation is successfully saved, -1 if the annotation file already exists.
    """
    file_path = os.path.join(dataset_dir, f"{image_id}.json")
    if os.path.exists(file_path):
        if verbose > 0: print(f"Annotation for {image_id} already exists - Skipping")
        return -1  # Skip this image as it already has an annotation
    
    with open(file_path, 'w') as f:
        json.dump(annotation, f, indent=4)
        if verbose > 0: print(f"Created {file_path}")

    return 0

def process_image(dataset_dir: str,
                  image_info: dict,
                  dataset_name: str,
                  image_index: int,
                  resize: bool,
                  resize_dim : int,
                  use_prompts: bool,
                  verbose: int) -> tuple:
    """
    Process an image from a dataset and generate annotations.

    This function handles the processing of an individual image from the given dataset.
    It generates annotations based on the provided parameters, saves the annotation
    in JSON format, and returns the annotation along with the time taken for processing.

    Parameters:
        dataset_dir (str): The directory where the annotation will be stored.
        image_info (dict): Dictionary containing Pillow image.
        dataset_name (str): The name of the dataset.
        image_index (int): The index of the image within the dataset.
        resize (bool): A flag indicating whether the image should be resized.
        resize_dim (int): The dimension to resize the image to if resizing is enabled.
        use_prompts (bool): A flag indicating whether prompts should be used in annotation generation.
        verbose (int): The verbosity level for logging (0 = no logs/messages, 3 = most messages)

    Returns:
        tuple: A tuple containing the generated annotation and the time elapsed during processing.
            If the image data is missing or annotation generation fails, returns (None, time_elapsed).
    """
    if image_info['image'] is None:
        if verbose > 1:
            logging.debug(f"Skipping image {image_index} in dataset {dataset_name} due to missing image data.")
        return None, 0

    original_dimensions = image_info['image'].size
    start_time = time.time()
    annotation = generate_annotation(image_index,
                                     dataset_name,
                                     image_info['image'],
                                     original_dimensions,
                                     resize,
                                     resize_dim,
                                     use_prompts,
                                     verbose)
    save_annotation(dataset_dir, image_index, annotation, verbose)
    time_elapsed = time.time() - start_time

    if annotation == -1:
        if verbose > 1:
            logging.debug(f"Failed to generate annotation for image {image_index} in dataset {dataset_name}")
        return None, time_elapsed
    
    return annotation, time_elapsed

def compute_annotation_latency(processed_images: int, dataset_time: float, dataset_name: str) -> float:
    """
    Compute the average annotation latency for a dataset.

    This function calculates the average time taken to annotate each image in a dataset
    based on the total time spent and the number of processed images. If no images were
    processed, it returns 0.0.

    Parameters:
        processed_images (int): The number of images that were successfully processed.
        dataset_time (float): The total time taken to process the dataset, in seconds.
        dataset_name (str): The name of the dataset.

    Returns:
        float: The average annotation latency per image in seconds. If no images were processed, returns 0.0.
    """
    if processed_images > 0:
        average_latency = dataset_time / processed_images
        logging.info(f'Average annotation latency for {dataset_name}: {average_latency:.4f} seconds')
        return average_latency
    return 0.0

def generate_annotation_dataset(real_image_datasets: List[Any],
                                save_path: str = 'annotations/',
                                verbose: int = 0,
                                max_images: int | None = None,
                                resize_images = False,
                                resize_dim = 512,
                                use_prompts : bool = True) -> Tuple[Dict[str, Dict[str, Any]], float]:
    """
    Generates text annotations for images in the given datasets, saves them in a specified directory, 
    and computes the average per image latency. Returns a dictionary of new annotations and the average latency.

    Parameters:
        real_image_datasets (List[Any]): Datasets containing images.
        save_path (str): Directory path for saving annotation files.
        verbose (int): Verbosity level between 0, 1, 2, 3 for process logs/messages (No messages = 0; Most messages = 3).
        max_images (int): Maximum number of images to annotate.
        resize_images (bool): Allow image downsizing before captioning.
                            Sets max dimensions to (resize_dim, resize_dim), maintaining aspect ratio.
        use_prompts (bool): Whether to enable prompt input for BLIP-2 for more descriptive annotations.
3
    Returns:
        Tuple[Dict[str, Dict[str, Any]], float]: A tuple containing the annotations dictionary and average latency.
    """
    set_logging_level(verbose)
    annotations_dir = ensure_save_path(save_path)
    annotations = {}
    total_time = 0
    total_processed_images = 0

    for i, dataset in enumerate(real_image_datasets):
        dataset_name = dataset.huggingface_dataset_path
        dataset_dir = create_annotation_dataset_directory(annotations_dir, dataset_name)
        processed_images = 0
        dataset_time = 0

        for j, image_info in enumerate(dataset):
            annotation, time_elapsed = \
            process_image(dataset_dir, image_info, dataset_name, j, resize_images, resize_dim, use_prompts, verbose)
            if annotation is not None:
                annotations.setdefault(dataset_name, {})[image_info['id']] = annotation
                total_time += time_elapsed
                dataset_time += time_elapsed
                processed_images += 1
                if max_images is not None and processed_images >= max_images:
                    break

        average_latency = compute_annotation_latency(processed_images, dataset_time, dataset_name)
        total_processed_images += processed_images

    overall_average_latency = total_time / total_processed_images if total_processed_images else 0
    return annotations, overall_average_latency

def list_datasets(base_dir):
    """List all subdirectories in the base directory."""
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

def load_annotations(base_dir, dataset):
    """Load annotations from JSON files within a specified directory."""
    annotations = []
    path = os.path.join(base_dir, dataset)
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), 'r') as file:
                data = json.load(file)
                annotations.append(data)
    return annotations

def load_diffuser(model_name):
    """Load a diffusion model by name, configured to provided arguments."""
    bt.logging.info(f"Loading image generation model ({model_name})...")
    model = DiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float32 if device == "cpu" else torch.float16, **DIFFUSER_ARGS[model_name]
    )
    model.to(device)
    return model

## GPU
def generate_images(annotations, diffuser, save_dir, num_images, batch_size, diffuser_name):
    """Generate images from annotations using a diffuser and save to the specified directory."""
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    generated_images = []
    start_time = time.time()

    with torch.no_grad():
        for i in range(min(num_images, len(annotations))):
            start_loop = time.time()
            annotation = annotations[i]
            prompt = annotation['description']
            index = annotation.get('index', f"missing_index")

            logging.info(f"Annotation {i}: {json.dumps(annotation, indent=2)}")

            generated_image = diffuser(prompt=prompt).images[0]
            logging.info(f"Type of generated image: {type(generated_image)}")

            if isinstance(generated_image, torch.Tensor):
                img = ToPILImage()(generated_image)
            else:
                img = generated_image

            safe_prompt = prompt[:50].replace(' ', '_').replace('/', '_').replace('\\', '_')
            img_filename = f"{save_dir}/{safe_prompt}-{index}.png"
            img.save(img_filename)
            generated_images.append(img_filename)
            loop_time = time.time() - start_loop
            logging.info(f"Image saved to {img_filename}")

    total_time = time.time() - start_time
    logging.info(f"Total processing time: {total_time:.2f} seconds")
    return generated_images


def load_and_initialize_diffuser(diffuser_name, previous_diffuser=None):
    """Load and initialize the diffuser, handling previous diffuser cleanup if needed."""
    if previous_diffuser is not None:
        logging.info("Deleting previous diffuser, freeing memory")
        # Move to float32 if it's float16, then move to CPU for deletion
        if previous_diffuser.dtype == torch.float16:
            previous_diffuser = previous_diffuser.to(dtype=torch.float32)
        previous_diffuser.to('cpu')
        del previous_diffuser
        gc.collect()
        torch.cuda.empty_cache()
        
    return load_diffuser(diffuser_name)

def run_diffuser_on_dataset(dataset, annotations, diffuser, output_dir, num_images, batch_size, diffuser_name):
    """Test a single diffuser on a given dataset."""
    dataset_name = dataset.rsplit('/', 1)[-1] if '/' in dataset else dataset
    diffuser_name = diffuser_name.rsplit('/', 1)[-1] if '/' in diffuser_name else diffuser_name
    save_dir = os.path.join(output_dir, dataset_name, diffuser_name)
    logging.info(f"Testing {diffuser_name} on annotation dataset {dataset} at {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        generate_images(annotations, diffuser, save_dir, num_images, batch_size, diffuser_name)
        logging.info("Images generated and saved successfully.")
    except Exception as e:
        logging.error(f"Failed to generate images with {diffuser_name}: {str(e)}")

def cleanup_diffuser(diffuser):
    """Clean up resources associated with a diffuser."""
    logging.info("Deleting diffuser, freeing memory")
    # Move to float32 if it's float16, then move to CPU for deletion
    if diffuser.dtype == torch.float16:
        diffuser = diffuser.to(dtype=torch.float32)
    diffuser.to('cpu')
    del diffuser
    gc.collect()
    torch.cuda.empty_cache()

def run_diffusers_on_datasets(annotations_dir, output_dir, num_images=sys.maxsize, batch_size=2):
    """Test all diffusers on datasets."""
    datasets = list_datasets(annotations_dir)
    for diffuser_name in DIFFUSER_NAMES:
        logging.info(f"Loading and initializing diffuser: {diffuser_name}")
        diffuser = load_and_initialize_diffuser(diffuser_name)
        for dataset in datasets:
            annotations = load_annotations(annotations_dir, dataset)
            run_diffuser_on_dataset(dataset, annotations, diffuser, output_dir, num_images, batch_size, diffuser_name)
        cleanup_diffuser(diffuser)

### Generate Annotations

# Change real_image_datasets to select which datasets to generate annotations and synthetics from.

# max_images=None for entire dataset

def main():
    # Configure logging to capture process information
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Suppress FutureWarnings from diffusers module
    warnings.filterwarnings("ignore", category=FutureWarning, module='diffusers')

    # Set the device for model operations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        raise RuntimeError("This script requires a GPU because it uses torch.float16.")

    # Load real image datasets from predefined metadata
    print("Loading real datasets")
    real_image_datasets = [
        ImageDataset(ds['path'], 'test', ds.get('name', None), ds['create_splits'])
        for ds in DATASET_META['real']
    ]

    # Path settings for annotations and synthetic image outputs
    ANNOTATIONS_DIR = "test_data/dataset/annotations/"
    OUTPUT_DIR = "test_data/dataset/synthetics_from_annotations/"

    # Generate text annotations for the real image datasets
    annotations_dict, average_latency = generate_annotation_dataset(
        real_image_datasets,
        save_path=ANNOTATIONS_DIR,
        verbose=2,
        max_images=None,
        use_prompts=True
    )

    # Move BLIP-2 to CPU and free up GPU memory after generating annotations
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic images from the annotations
    run_diffusers_on_datasets(ANNOTATIONS_DIR, OUTPUT_DIR, num_images=sys.maxsize, batch_size=8)

if __name__ == "__main__":
    main()
