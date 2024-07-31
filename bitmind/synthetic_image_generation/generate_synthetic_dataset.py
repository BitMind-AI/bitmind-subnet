import argparse
import logging
import json
import os
import sys
import torch
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi

from synthetic_image_generator import SyntheticImageGenerator
from bitmind.image_dataset import ImageDataset
from bitmind.constants import HUGGINGFACE_CACHE_DIR
from bitmind.utils.data import load_huggingface_dataset

def parse_arguments():
    """
    Parse command-line arguments for generating synthetic images and annotations
    from a single real dataset.

    Example Usage:

    Generate 10 mirrors of celeb-a-hq using stabilityai/stable-diffusion-xl-base-1.0
    and upload annotations and images to Hugging Face. Replace YOUR_HF_TOKEN with your
    actual Hugging Face API token:

    python generate_synthetic_dataset.py --hf_org "bitmind" --real_image_dataset_name "celeb-a-hq" \
    --generate_annotations --diffusion_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --upload_annotations --upload_synthetic_images --hf_token YOUR_HF_TOKEN --n 10

    Generate mirrors of the entire ffhq256 using stabilityai/stable-diffusion-xl-base-1.0
    and upload annotations and images to Hugging Face. Replace YOUR_HF_TOKEN with your
    actual Hugging Face API token:

    python generate_synthetic_dataset.py --hf_org "bitmind" --real_image_dataset_name "ffhq256" \
    --generate_annotations --diffusion_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --upload_annotations --upload_synthetic_images --hf_token YOUR_HF_TOKEN

    Arguments:
    --hf_org (str): Required. Hugging Face organization name.
    --real_image_dataset_name (str): Required. Name of the real image dataset.
    --generate_annotations (bool): Flag to generate annotations; skips if false.
    --diffusion_model (str): Required. Diffusion model to use for image generation.
    --upload_annotations (bool): Optional. Flag to upload annotations to Hugging Face.
    --upload_synthetic_images (bool): Optional. Flag to upload synthetic images to Hugging Face.
    --hf_token (str): Optional. Token for uploading to Hugging Face.
    --n (int): Optional. Maximum number of annotations and images to generate.
    """
    parser = argparse.ArgumentParser(description='Generate synthetic images and annotations from a real dataset.')
    parser.add_argument('--hf_org', type=str, required=True, help='Hugging Face prg name.')
    parser.add_argument('--real_image_dataset_name', type=str, required=True, help='Real image dataset name.')
    parser.add_argument('--generate_annotations', action='store_true', 
                        help='Generate annotations; skip if annotations already exist.')
    parser.add_argument('--diffusion_model', type=str, required=True, 
                        help='Diffusion model to use for image generation.')
    parser.add_argument('--upload_annotations', action='store_true', default=True, 
                        help='Upload annotations to Hugging Face. hf_token arg required.')
    parser.add_argument('--upload_synthetic_images', action='store_true', default=True, 
                        help='Upload synthetic images to Hugging Face. hf_token arg required.')
    parser.add_argument('--hf_token', type=str, default=None, help='Token for uploading to Hugging Face.')
    parser.add_argument('--n', type=int, default=None, help='Maximum number of annotations and images to generate.')

    return parser.parse_args()

def upload_to_huggingface(dataset, repo_name, token):
    """Uploads the dataset dictionary to Hugging Face."""
    api = HfApi()
    api.create_repo(repo_name, repo_type="dataset", private=False, token=token)
    dataset.push_to_hub(repo_name)

def main():
    args = parse_arguments()
    hf_dataset_name = f"{args.hf_org}/{args.real_image_dataset_name}"
    annotations_dir = f'test_data/annotations/{args.real_image_dataset_name}'
    synthetic_images_dir = f'test_data/synthetic_images/{args.real_image_dataset_name}'
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(synthetic_images_dir, exist_ok=True)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

    synthetic_image_generator = SyntheticImageGenerator(
        prompt_type='annotation', use_random_diffuser=False, diffuser_name=args.diffusion_model)
    
    start_time = time.time()
    # Load the dataset based on command-line args
    dataset = ImageDataset(hf_dataset_name, 'train')
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")
            
    annotations = []
    image_count = 0

    # Processing loop: Generate annotations and synthetic images
    if args.generate_annotations: 
        start_time = time.time()
        for real_image in dataset:
            if args.n is not None and image_count >= args.n:
                break
            annotation = synthetic_image_generator.image_annotation_generator.process_image(
                image_info=real_image,
                dataset_name=real_image['source'],
                image_index=real_image['id'],
                resize=False,
                verbose=0
            )[0]

            # Save to local dir to using up memory
            file_path = os.path.join(annotations_dir, f"{real_image['id']}.json")
            with open(file_path, 'w') as f:
                json.dump(annotation, f)

            image_count += 1

        synthetic_image_generator.image_annotation_generator.clear_gpu()
        print(f"{args.n} annotations generated and saved in {time.time() - start_time:.2f} seconds.")
        print(f"Mean annotation generation time: {args.n/(time.time() - start_time):.2f} seconds.")

        # Upload to Hugging Face
        if args.upload_annotations and args.hf_token:
            print("Loading annotations dataset.")
            start_time = time.time()
            annotations_dataset = load_dataset('json', data_files=os.path.join(annotations_dir, '*.json'))
            print("Uploading annotations of" + args.real_image_dataset_name + " to Hugging Face.")
            upload_to_huggingface(annotations_dataset, hf_dataset_name+"_annotations", args.hf_token)
            print(f"Annotations uploaded to Hugging Face in {time.time() - start_time:.2f} seconds.")

    else:
        # Load annotations from Hugging Face
        annotations = load_huggingface_dataset(args.hf_dataset_path,
                                    cache_dir=HUGGINGFACE_CACHE_DIR,
                                    download_mode="reuse_cache_if_exists",
                                    trust_remote_code=True)
    
    image_count = 0
    start_time = time.time()
    for annotation in annotations:
        if args.n is not None and image_count >= args.n:
            break
        synthetic_image = synthetic_image_generator.generate_image(annotation)
        filename = f"{annotation['index']}.png"
        file_path = os.path.join(synthetic_images_dir, filename)
        synthetic_image.save(synthetic_images_dir)
        image_count += 1
    print(f"Synthetic images generated in {time.time() - start_time:.2f} seconds.")
    print(f"Mean synthetic images generation time: {image_count/(time.time() - start_time):.2f} seconds.")

    if args.upload_synthetic_images and args.hf_token:
        start_time = time.time()
        print("Loading synthetic image dataset.")
        synthetic_image_dataset = load_dataset("imagefolder", data_dir=synthetic_images_dir)
        print("Uploading synthetic image mirrors of " + args.real_image_dataset_name + " to Hugging Face.")
        upload_to_huggingface(synthetic_image_dataset, hf_dataset_name+"_"+args.diffusion_model, args.hf_token)
        print(f"Synthetic images uploaded in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
