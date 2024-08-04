import argparse
import logging
import json
import os
import torch
import time
from pathlib import Path
import pandas as pd
from math import ceil
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

from datasets import load_dataset

from synthetic_image_generator import SyntheticImageGenerator
from bitmind.image_dataset import ImageDataset
from bitmind.constants import TARGET_IMAGE_SIZE
from utils.hugging_face_utils import (
    dataset_exists_on_hf, load_and_sort_dataset, upload_to_huggingface, 
    slice_dataset, save_as_json
)
from utils.image_utils import resize_image, resize_images_in_directory

PROGRESS_INCREMENT = 10

def parse_arguments():
    """
    Parse command-line arguments for generating synthetic images and annotations
    from a single real dataset.

    Before running, authenticate with command line to upload to Hugging Face:
    huggingface-cli login
    
    Do not add token as Git credential.

    Example Usage:

    Generate the first 10 mirrors of celeb-a-hq with stabilityai/stable-diffusion-xl-base-1.0
    and existing annotations from Hugging Face, and upload images to Hugging Face.
    Replace YOUR_HF_TOKEN with your actual Hugging Face API token:

    pm2 start generate_synthetic_dataset.py --name "first_ten_celebahq" --no-autorestart \
    -- --hf_org 'bitmind' --real_image_dataset_name 'celeb-a-hq' \
    --diffusion_model 'stabilityai/stable-diffusion-xl-base-1.0' --upload_synthetic_images \
    --hf_token 'YOUR_HF_TOKEN' --start_index 0 --end_index 10

    Generate mirrors of the entire ffhq256 using stabilityai/stable-diffusion-xl-base-1.0
    and upload annotations and images to Hugging Face. Replace YOUR_HF_TOKEN with your
    actual Hugging Face API token:

    pm2 start generate_synthetic_dataset.py --name "ffhq256" --no-autorestart \
    -- --hf_org "bitmind" --real_image_dataset_name "ffhq256" \
    --diffusion_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --upload_annotations --upload_synthetic_images \ --hf_token "YOUR_HF_TOKEN""

    Arguments:
    --hf_org (str): Required. Hugging Face organization name.
    --real_image_dataset_name (str): Required. Name of the real image dataset.
    --diffusion_model (str): Required. Diffusion model to use for image generation.
    --upload_annotations (bool): Optional. Flag to upload annotations to Hugging Face.
    --generate_synthetic_images (bool): Optional. Flag to generate synthetic images.
    --upload_synthetic_images (bool): Optional. Flag to upload synthetic images to Hugging Face.
    --hf_token (str): Required for interfacing with Hugging Face.
    parser.add_argument('--start_index', type=int, default=0, help='Start index for processing the dataset.')
    parser.add_argument('--end_index', type=int, default=None, help='End index (exclusive) for processing the dataset.')
    """
    parser = argparse.ArgumentParser(description='Generate synthetic images and annotations from a real dataset.')
    parser.add_argument('--hf_org', type=str, required=True, help='Hugging Face org name.')
    parser.add_argument('--real_image_dataset_name', type=str, required=True, help='Real image dataset name.')
    parser.add_argument('--diffusion_model', type=str, required=True, 
                        help='Diffusion model to use for image generation.')
    parser.add_argument('--upload_annotations', action='store_true', default=False, 
                        help='Upload annotations to Hugging Face.')
    parser.add_argument('--generate_synthetic_images', action='store_true', default=False, 
                        help='Generate synthetic images.')
    parser.add_argument('--upload_synthetic_images', action='store_true', default=False, 
                        help='Upload synthetic images to Hugging Face.')
    parser.add_argument('--hf_token', type=str, default=None, help='Token for uploading to Hugging Face.')
    parser.add_argument('--start_index', type=int, default=0, required=True, help='Start index for processing the dataset. Default to the first index.')
    parser.add_argument('--end_index', type=int, default=None, required=True, help='End index for processing the dataset. Default to the last index.')
    parser.add_argument('--no-resize', action='store_false', dest='resize', help='Do not resize to target image size from BitMind constants.')
    parser.add_argument('--resize_existing', action='store_true', default=False, required=False, help='Resize existing image files.')
    return parser.parse_args()


def generate_and_save_annotations(dataset, dataset_name, synthetic_image_generator, annotations_dir, batch_size=16):
    annotations_batch = []
    image_count = 0
    start_time = time.time()
    # Update progress every PROGRESS_INCREMENT % of image chunk
    progress_interval = (batch_size * ceil(len(dataset) / (PROGRESS_INCREMENT * batch_size)))

    for index, real_image in enumerate(dataset):
        annotation = synthetic_image_generator.image_annotation_generator.process_image(
            real_image,
            dataset_name,
            index,
            resize=False,
            verbose=0
        )[0]
        annotations_batch.append((index, annotation))

        if len(annotations_batch) == batch_size or image_count == len(dataset) - 1:
            for image_id, annotation in annotations_batch:
                file_path = os.path.join(annotations_dir, f"{image_id}.json")
                with open(file_path, 'w') as f:
                    json.dump(annotation, f)
            annotations_batch = []

        image_count += 1

        if image_count % progress_interval == 0 or image_count == len(dataset):
            print(f"Progress: {image_count}/{len(dataset)} annotations generated.")

    synthetic_image_generator.image_annotation_generator.clear_gpu()
    duration = time.time() - start_time
    print(f"All {image_count} annotations generated and saved in {duration:.2f} seconds.")
    print(f"Mean annotation generation time: {duration/image_count:.2f} seconds if any.")


def save_json_files(json_filenames, annotations_dir, synthetic_image_generator, synthetic_images_dir, resize=True):
    total_images = 0
    for json_filename in json_filenames:
        json_path = os.path.join(annotations_dir, json_filename)
        with open(json_path, 'r') as file:
            annotation = json.load(file)
        prompt = annotation['description']
        name = annotation['id']
        synthetic_image = synthetic_image_generator.generate_image(prompt, name=name)
        filename = f"{name}.png"
        file_path = os.path.join(synthetic_images_dir, filename)
        if resize:
            synthetic_image = resize_image(synthetic_image['image'], TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1])
        synthetic_image['image'].save(file_path)
        total_images += 1
    return total_images


def generate_and_save_synthetic_images(annotations_dir, synthetic_image_generator, 
                                       synthetic_images_dir, start_index, end_index, 
                                       batch_size=16, resize=True):
    start_time = time.time()
    total_images = 0

    # Collect all valid annotation file paths first
    valid_files = []
    for json_filename in sorted(os.listdir(annotations_dir)):
        # Extract index from filename
        try:
            # Remove '.json' extension and converts to integer
            file_index = int(json_filename[:-5])  
        except ValueError:
            continue  # Skip files that don't match expected format

        if start_index <= file_index < end_index:
            valid_files.append(json_filename)

    total_valid_files = len(valid_files)
    # Update progress every PROGRESS_INCREMENT % of total annotations
    progress_interval = (batch_size * ceil(total_valid_files / (PROGRESS_INCREMENT * batch_size)))
    
    with torch.no_grad():  # Use no_grad to reduce memory usage during inference
        for i in range(0, total_valid_files, batch_size):
            batch_files = valid_files[i:i+batch_size]
            total_images += save_json_files(batch_files, annotations_dir, 
                                            synthetic_image_generator, synthetic_images_dir,
                                            resize)

            if i % progress_interval == 0 or total_images >= total_valid_files:
                print(f"Progress: {total_images}/{total_valid_files} images generated \
                ({(total_images / total_valid_files) * 100:.2f}%)")

    synthetic_image_generator.clear_gpu()
    duration = time.time() - start_time
    print(f"All {total_images} synthetic images generated in {duration:.2f} seconds.")
    print(f"Mean synthetic images generation time: {duration/max(total_images, 1):.2f} seconds.")

def main():
    args = parse_arguments()
    hf_dataset_name = f"{args.hf_org}/{args.real_image_dataset_name}"
    data_range = f"{args.start_index}-to-{args.end_index-1}"
    hf_annotations_name = f"{hf_dataset_name}___annotations"
    model_name = args.diffusion_model.split('/')[-1]
    hf_synthetic_images_name = f"{hf_dataset_name}___{data_range}___{model_name}"
    annotations_dir = f'test_data/annotations/{args.real_image_dataset_name}'
    real_image_samples_dir = f'test_data/real_images/{args.real_image_dataset_name}'
    synthetic_images_dir = f'test_data/synthetic_images/{args.real_image_dataset_name}'
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(synthetic_images_dir, exist_ok=True)

    synthetic_image_generator = SyntheticImageGenerator(
        prompt_type='annotation', use_random_diffuser=False, diffuser_name=args.diffusion_model)
                
    batch_size = 16

    # If annotations exist on Hugging Face, load them to disk.
    if dataset_exists_on_hf(hf_annotations_name, args.hf_token):
        print("Annotations exist on Hugging Face.")
        # Check if the annotations are already saved locally
        if not Path(annotations_dir).is_dir() or not any(Path(annotations_dir).iterdir()):
            print(f"Downloading annotations from {hf_annotations_name} and saving annotations to {annotations_dir}.")
            # Download annotations from Hugging Face
            all_annotations = load_dataset(hf_annotations_name, split='train', keep_in_memory=False)
            df_annotations = pd.DataFrame(all_annotations)
            # Ensure the index is of integer type and sort by it
            df_annotations['id'] = df_annotations['id'].astype(int)
            df_annotations.sort_values('id', inplace=True)
            # Slice specified chunk
            annotations_chunk = df_annotations.iloc[args.start_index:args.end_index]
            all_annotations = None
             # Save the chunk as JSON files on disk
            save_as_json(annotations_chunk, annotations_dir)
            annotations_chunk = None
        else:
            print("Annotations already saved to disk.")
    else:
        print("Generating new annotations.")
        all_images = ImageDataset(hf_dataset_name, 'train')
        images_chunk = slice_dataset(all_images.dataset, start_index=args.start_index, end_index=args.end_index)
        all_images = None
        generate_and_save_annotations(images_chunk, hf_dataset_name, synthetic_image_generator, annotations_dir, batch_size=batch_size)
        images_chunk = None # Free up memory

        # Upload to Hugging Face
        if args.upload_annotations and args.hf_token:
            start_time = time.time()
            print("Uploading annotations to HF.")
            print("Loading annotations dataset.")
            annotations_dataset = load_and_sort_dataset(annotations_dir, 'json')
            print("Uploading annotations of " + args.real_image_dataset_name + " to Hugging Face.")
            upload_to_huggingface(annotations_dataset, hf_annotations_name, args.hf_token)
            print(f"Annotations uploaded to Hugging Face in {time.time() - start_time:.2f} seconds.")

    if args.generate_synthetic_images:
        synthetic_image_generator.load_diffuser(diffuser_name=args.diffusion_model)
        generate_and_save_synthetic_images(annotations_dir, synthetic_image_generator,
                                           synthetic_images_dir, args.start_index, args.end_index,
                                           batch_size=batch_size, resize=args.resize)
    
        synthetic_image_generator.clear_gpu()
    
    if args.resize_existing:
        print(f"Resizing images in {synthetic_images_dir}.")
        resize_images_in_directory(synthetic_images_dir)
        hf_synthetic_images_name += f"___{TARGET_IMAGE_SIZE[0]}"
        print(f"Done resizing existing images.")

    if args.upload_synthetic_images and args.hf_token:
        start_time = time.time()
        print("Loading synthetic image dataset.")
        synthetic_image_dataset = load_and_sort_dataset(synthetic_images_dir, 'image')
        print("Uploading synthetic image mirrors of " + args.real_image_dataset_name + " to Hugging Face.")
        upload_to_huggingface(synthetic_image_dataset, hf_synthetic_images_name, args.hf_token)
        print(f"Synthetic images uploaded in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()