import argparse
import logging
import json
import os
import sys
import torch
import time
from pathlib import Path
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

from datasets import load_dataset
from huggingface_hub import HfApi

from synthetic_image_generator import SyntheticImageGenerator
from bitmind.image_dataset import ImageDataset
from bitmind.utils.data import load_huggingface_dataset

def parse_arguments():
    """
    Parse command-line arguments for generating synthetic images and annotations
    from a single real dataset.

    Before running, authenticate with command line to upload to Hugging Face:
    huggingface-cli login

    Example Usage:

    Generate the first 10 mirrors of celeb-a-hq using stabilityai/stable-diffusion-xl-base-1.0
    using existing annotations from Hugging Face, and upload images to Hugging Face.
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
    parser.add_argument('--upload_synthetic_images', action='store_true', default=False, 
                        help='Upload synthetic images to Hugging Face.')
    parser.add_argument('--hf_token', type=str, default=None, help='Token for uploading to Hugging Face.')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for processing the dataset. Default to the first index.')
    parser.add_argument('--end_index', type=int, default=None, help='End index for processing the dataset. Default to the last index.')

    return parser.parse_args()

def dataset_exists_on_hf(hf_dataset_name, token):
    """Check if the dataset exists on Hugging Face."""
    api = HfApi()
    try:
        dataset_info = api.dataset_info(hf_dataset_name, token=token)
        return True
    except Exception as e:
        return False

def upload_to_huggingface(dataset, repo_name, token):
    """Uploads the dataset dictionary to Hugging Face."""
    api = HfApi()
    api.create_repo(repo_name, repo_type="dataset", private=False, token=token)
    dataset.push_to_hub(repo_name)

def slice_dataset(dataset, start_index, end_index=None):
    """
    Slice the dataset according to provided start and end indices.

    Parameters:
    dataset (Dataset): The dataset to be sliced.
    start_index (int): The index of the first element to include in the slice.
    end_index (int, optional): The index of the last element to include in the slice. If None, slices to the end of the dataset.

    Returns:
    Dataset: The sliced dataset.
    """
    if end_index is not None and end_index < len(dataset):
        return dataset.select(range(start_index, end_index))
    else:
        return dataset.select(range(start_index, len(dataset)))

def save_as_json(dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    # Iterate through each record in the dataset
    for i, record in enumerate(dataset):
        file_path = os.path.join(output_dir, f"{i}.json")
        # Write the record as a JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=4)

def generate_and_save_annotations(dataset, dataset_name, synthetic_image_generator, annotations_dir, batch_size=8):
    annotations_batch = []
    image_count = 0
    start_time = time.time()
    progress_interval = max(1, len(dataset) // 10)  # Update progress every 10% of image chunk

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

def save_images_to_disk(image_dataset, start_index, num_images, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for i in range(start_index, start_index + num_images):
        try:
            image_data = image_dataset[i]  # Retrieve image using the __getitem__ method
            image = image_data['image']  # Extract the image
            image_id = image_data['id']  # Extract the image ID
            file_path = os.path.join(save_directory, f"{image_id}.jpg")  # Construct file path
            image.save(file_path, 'JPEG')  # Save the image
            print(f"Saved: {file_path}")
        except Exception as e:
            print(f"Failed to save image {i}: {e}")

def generate_and_save_synthetic_images(annotations_dir, synthetic_image_generator, 
                                       synthetic_images_dir, start_index, end_index, batch_size=8):
    start_time = time.time()
    annotation_files = sorted(Path(annotations_dir).glob('*.json'))
    total_images = 0
    annotations_batch = []

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
    progress_interval = max(1, total_valid_files // 10)  # Update progress every 10% of total annotations
    
    with torch.no_grad():  # Use no_grad to reduce memory usage during inference
        for i in range(0, len(valid_files), batch_size):
            batch_files = valid_files[i:i+batch_size]
            for json_filename in batch_files:
                json_path = os.path.join(annotations_dir, json_filename)
                with open(json_path, 'r') as file:
                    annotation = json.load(file)
                prompt = annotation['description']
                name = annotation['id']
                synthetic_image = synthetic_image_generator.generate_image(prompt, name=name)
                filename = f"{name}.png"
                file_path = os.path.join(synthetic_images_dir, filename)
                synthetic_image['image'].save(file_path)
                total_images += 1

            if i % progress_interval == 0 or end >= total:
                print(f"Progress: {total_images}/{total_valid_files} images generated \
                ({(total_images / total_valid_files) * 100:.2f}%)")

    synthetic_image_generator.clear_gpu()
    duration = time.time() - start_time
    print(f"All {total_images} synthetic images generated in {duration:.2f} seconds.")
    print(f"Mean synthetic images generation time: {duration/max(total_images, 1):.2f} seconds.")

def main():
    args = parse_arguments()
    hf_dataset_name = f"{args.hf_org}/{args.real_image_dataset_name}"
    hf_annotations_name = f"{hf_dataset_name}-annotations"
    model_name = args.diffusion_model.split('/')[-1]
    hf_synthetic_images_name = f"{hf_dataset_name}_{model_name}"
    annotations_dir = f'test_data/annotations/{args.real_image_dataset_name}'
    real_image_samples_dir = f'test_data/real_images/{args.real_image_dataset_name}'
    synthetic_images_dir = f'test_data/synthetic_images/{args.real_image_dataset_name}'
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(synthetic_images_dir, exist_ok=True)

    synthetic_image_generator = SyntheticImageGenerator(
        prompt_type='annotation', use_random_diffuser=False, diffuser_name=args.diffusion_model)
                
    batch_size = 8
    image_count = 0

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
            df_annotations['index'] = df_annotations['index'].astype(int)
            df_annotations.sort_values('index', inplace=True)
            print(df_annotations)
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
        #save_images_to_disk(all_images, 0, 10, real_image_samples_dir)
        images_chunk = slice_dataset(all_images.dataset, start_index=args.start_index, end_index=args.end_index)
        all_images = None
        generate_and_save_annotations(images_chunk, hf_dataset_name, synthetic_image_generator, annotations_dir, batch_size=batch_size)
        images_chunk = None # Free up memory

        # Upload to Hugging Face
        if args.upload_annotations and args.hf_token:
            start_time = time.time()
            print("Uploading annotations to HF.")
            print("Loading annotations dataset.")
            annotations_dataset = load_dataset('json', data_files=os.path.join(annotations_dir, '*.json'))
            print("Uploading annotations of" + args.real_image_dataset_name + " to Hugging Face.")
            upload_to_huggingface(annotations_dataset, hf_annotations_name, args.hf_token)
            print(f"Annotations uploaded to Hugging Face in {time.time() - start_time:.2f} seconds.")

    synthetic_image_generator.load_diffuser(diffuser_name=args.diffusion_model)
    generate_and_save_synthetic_images(annotations_dir, synthetic_image_generator,
                                       synthetic_images_dir, args.start_index, args.end_index,
                                       batch_size=batch_size)
    
    synthetic_image_generator.clear_gpu()

    if args.upload_synthetic_images and args.hf_token:
        start_time = time.time()
        print("Loading synthetic image dataset.")
        synthetic_image_dataset = load_dataset("imagefolder", data_dir=synthetic_images_dir)
        print("Uploading synthetic image mirrors of " + args.real_image_dataset_name + " to Hugging Face.")
        upload_to_huggingface(synthetic_image_dataset, hf_synthetic_images_name, args.hf_token)
        print(f"Synthetic images uploaded in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
