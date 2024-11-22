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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

from datasets import load_dataset

from bitmind.synthetic_data_generation.synthetic_data_generator import SyntheticDataGenerator
from bitmind.image_dataset import ImageDataset
from bitmind.constants import TARGET_IMAGE_SIZE
from bitmind.utils.hugging_face_utils import (
    dataset_exists_on_hf, load_and_sort_dataset, upload_to_huggingface, 
    slice_dataset, save_as_json
)
from bitmind.utils.image_transforms import resize_image, resize_images_in_directory

PROGRESS_INCREMENT = 10

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate synthetic images and annotations from a real dataset.')
    parser.add_argument('--hf_org', type=str, required=True, help='Hugging Face org name.')
    parser.add_argument('--real_image_dataset_name', type=str, required=True, help='Real image dataset name.')
    parser.add_argument('--diffusion_model', type=str, required=True, 
                        help='Diffusion model to use for image generation.')
    parser.add_argument('--upload_annotations', action='store_true', default=False, 
                        help='Upload annotations to Hugging Face.')
    parser.add_argument('--download_annotations', action='store_true', default=False, 
                        help='Download annotations from Hugging Face.')
    parser.add_argument('--skip_generate_annotations', action='store_true', default=False, 
                        help='Skip annotation generation and use existing annotations.')
    parser.add_argument('--generate_synthetic_images', action='store_true', default=False, 
                        help='Generate synthetic images.')
    parser.add_argument('--upload_synthetic_images', action='store_true', default=False, 
                        help='Upload synthetic images to Hugging Face.')
    parser.add_argument('--hf_token', type=str, default=None, help='Token for uploading to Hugging Face.')
    parser.add_argument('--start_index', type=int, default=0, required=True, 
                        help='Start index for processing the dataset.')
    parser.add_argument('--end_index', type=int, default=None, required=True, 
                        help='End index for processing the dataset.')
    parser.add_argument('--gpu_id', type=int, default=0, required=True, 
                        help='Which GPU to use (check nvidia-smi -L).')
    parser.add_argument('--no-resize', action='store_false', dest='resize', 
                        help='Do not resize to target image size from BitMind constants.')
    parser.add_argument('--resize_existing', action='store_true', default=False, 
                        help='Resize existing image files.')
    return parser.parse_args()

def generate_and_save_annotations(dataset,
                                start_index,
                                dataset_name,
                                synthetic_data_generator,
                                annotations_dir,
                                batch_size=16):
    annotations_batch = []
    image_count = 0
    start_time = time.time()
    progress_interval = (batch_size * ceil(len(dataset) / (PROGRESS_INCREMENT * batch_size)))

    for index, real_image in enumerate(dataset):
        adjusted_index = index + start_index
        # Use the image annotation generator directly
        annotation = synthetic_data_generator.image_annotation_generator.process_image(
            real_image,
            dataset_name,
            adjusted_index,
            resize=False,
            verbose=0
        )[0]
        annotations_batch.append((adjusted_index, annotation))

        if len(annotations_batch) == batch_size or image_count == len(dataset) - 1:
            for image_id, annotation in annotations_batch:
                file_path = os.path.join(annotations_dir, f"{image_id}.json")
                with open(file_path, 'w') as f:
                    json.dump(annotation, f)
            annotations_batch = []

        image_count += 1
        if image_count % progress_interval == 0 or image_count == len(dataset):
            print(f"Progress: {image_count}/{len(dataset)} annotations generated.")

    synthetic_data_generator.image_annotation_generator.clear_gpu()
    duration = time.time() - start_time
    print(f"All {image_count} annotations generated and saved in {duration:.2f} seconds.")

def save_json_files(batch_files, annotations_dir, synthetic_data_generator, synthetic_images_dir, resize=True):
    total_images = 0
    for json_filename in batch_files:
        with open(os.path.join(annotations_dir, json_filename), 'r') as f:
            annotation = json.load(f)
        
        try:
            # Generate image using the prompt from annotation
            gen_data = synthetic_data_generator.generate(
                modality='image',
                prompt=annotation['prompt']
            )
            
            # Save the generated image
            image_filename = json_filename.replace('.json', '.png')
            image_path = os.path.join(synthetic_images_dir, image_filename)
            
            gen_image = gen_data['gen_output'].images[0]
            if resize:
                gen_image = resize_image(gen_image, TARGET_IMAGE_SIZE)
            gen_image.save(image_path)
            
            total_images += 1
            
        except Exception as e:
            print(f"Error generating image for {json_filename}: {e}")
            continue
            
    return total_images

def generate_and_save_synthetic_images(annotations_dir, synthetic_data_generator,
                                     synthetic_images_dir, start_index, end_index,
                                     batch_size=16, resize=True):
    total_images = 0
    start_time = time.time()
    os.makedirs(synthetic_images_dir, exist_ok=True)

    valid_files = []
    for json_filename in sorted(os.listdir(annotations_dir)):
        try:
            file_index = int(json_filename[:-5])
        except ValueError:
            continue
        if start_index <= file_index <= end_index:
            valid_files.append(json_filename)

    total_valid_files = len(valid_files)
    progress_interval = (batch_size * ceil(total_valid_files / (PROGRESS_INCREMENT * batch_size)))
    
    with torch.no_grad():
        for i in range(0, total_valid_files, batch_size):
            batch_files = valid_files[i:i+batch_size]
            total_images += save_json_files(batch_files, annotations_dir, 
                                          synthetic_data_generator, synthetic_images_dir,
                                          resize)

            if i % progress_interval == 0 or total_images >= total_valid_files:
                print(f"Progress: {total_images}/{total_valid_files} images generated "
                      f"({(total_images / total_valid_files) * 100:.2f}%)")

    synthetic_data_generator.clear_gpu()
    duration = time.time() - start_time
    print(f"All {total_images} synthetic images generated in {duration:.2f} seconds.")
    print(f"Mean synthetic images generation time: {duration/max(total_images, 1):.2f} seconds.")

def main():
    args = parse_arguments()
    hf_dataset_name = f"{args.hf_org}/{args.real_image_dataset_name}"
    data_range = f"{args.start_index}-to-{args.end_index}"
    hf_annotations_name = f"{hf_dataset_name}___annotations"
    model_name = args.diffusion_model.split('/')[-1]
    hf_synthetic_images_name = f"{hf_dataset_name}___{data_range}___{model_name}"
    
    # Setup directories
    annotations_dir = f'test_data/annotations/{args.real_image_dataset_name}'
    annotations_chunk_dir = Path(f"{annotations_dir}/{args.start_index}_{args.end_index}/")
    synthetic_images_dir = f'test_data/synthetic_images/{args.real_image_dataset_name}'
    synthetic_images_chunk_dir = Path(f'{synthetic_images_dir}/{args.start_index}_{args.end_index}/')
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(annotations_chunk_dir, exist_ok=True)
    os.makedirs(synthetic_images_dir, exist_ok=True)
                
    batch_size = 16
    
    synthetic_data_generator = None
    
    # Handle annotations
    if args.download_annotations and dataset_exists_on_hf(hf_annotations_name, args.hf_token):
        synthetic_data_generator = SyntheticDataGenerator(
            t2vis_model_name=args.diffusion_model,
            use_random_t2vis_model=False,
            prompt_type='none',
            gpu_id=args.gpu_id
        )
        # Download and process annotations
        if not annotations_chunk_dir.is_dir() or not any(annotations_chunk_dir.iterdir()):
            print(f"Downloading annotations from {hf_annotations_name}")
            all_annotations = load_dataset(hf_annotations_name, split='train', keep_in_memory=False)
            df_annotations = pd.DataFrame(all_annotations)
            df_annotations['id'] = df_annotations['id'].astype(int)
            df_annotations.sort_values('id', inplace=True)
            annotations_chunk = df_annotations.iloc[args.start_index:args.end_index + 1]
            save_as_json(annotations_chunk, annotations_chunk_dir)
        else:
            print("Annotations already saved to disk.")
    elif not args.skip_generate_annotations:
        synthetic_data_generator = SyntheticDataGenerator(
            t2vis_model_name=args.diffusion_model,
            use_random_t2vis_model=False,
            prompt_type='annotation',
            gpu_id=args.gpu_id
        )
        print("Generating new annotations.")
        all_images = ImageDataset(hf_dataset_name, 'train')
        images_chunk = slice_dataset(all_images.dataset, start_index=args.start_index, end_index=args.end_index)
        generate_and_save_annotations(
            images_chunk,
            args.start_index,
            hf_dataset_name,
            synthetic_data_generator,
            annotations_chunk_dir,
            batch_size=batch_size
        )
    
    # Upload annotations if requested
    if args.upload_annotations and args.hf_token:
        start_time = time.time()
        print("Uploading annotations to HF.")
        annotations_dataset = load_and_sort_dataset(annotations_chunk_dir, 'json')
        upload_to_huggingface(annotations_dataset, hf_annotations_name, args.hf_token)
        print(f"Annotations uploaded in {time.time() - start_time:.2f} seconds.")

    # Generate synthetic images
    if args.generate_synthetic_images:
        if synthetic_data_generator is None:
            synthetic_data_generator = SyntheticDataGenerator(
                t2vis_model_name=args.diffusion_model,
                use_random_t2vis_model=False,
                prompt_type='none',
                gpu_id=args.gpu_id
            )
        synthetic_images_chunk_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating and saving images to {synthetic_images_chunk_dir}")
        generate_and_save_synthetic_images(
            annotations_chunk_dir,
            synthetic_data_generator,
            synthetic_images_chunk_dir,
            args.start_index,
            args.end_index,
            batch_size=batch_size,
            resize=args.resize
        )
    
    if args.resize_existing:
        print(f"Resizing images in {synthetic_images_chunk_dir}")
        resize_images_in_directory(synthetic_images_chunk_dir)
        hf_synthetic_images_name += f"___{TARGET_IMAGE_SIZE[0]}"
        print("Done resizing existing images.")

    if args.upload_synthetic_images and args.hf_token:
        start_time = time.time()
        print("Uploading synthetic images to HF")
        synthetic_image_dataset = load_and_sort_dataset(synthetic_images_chunk_dir, 'image')
        upload_to_huggingface(synthetic_image_dataset, hf_synthetic_images_name, args.hf_token)
        print(f"Synthetic images uploaded in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main() 