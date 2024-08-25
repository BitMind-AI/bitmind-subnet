# Run this script in the bitmind-subnet/base_miner/UCF/preprocessing/ directory
# to interface with TrainingDatasetProcessor from training_dataset_processor.py

# Example usage:
# !) Generate and upload preprocessed training dataset of transformed images
#    from BitMind HuggingFace repos filtered for faces only, cropped and aligned:
# 
#    pm2 start generate_training_dataset.py --no-autorestart -- --faces-only --hf-token [HuggingFace Write Token]
# 2) Generate and upload preprocessed training dataset of transformed images
#    from BitMind HuggingFace repos (unfiltered)
# 
#    pm2 start generate_training_dataset.py --no-autorestart -- --hf-token [HuggingFace Write Token]
#
# 3) Same as 1 but with splits generated for individual datasets uploaded to HF
#    NOTE: Not a default function because splitting is typically handled after
#          loading all datasets in bitmind-subnet/base_miner/UCF/train_ucf.py
#
#    pm2 start generate_training_dataset.py --no-autorestart -- --faces-only --splits --hf-token [HuggingFace Write Token]

import argparse
from training_dataset_processor import TrainingDatasetProcessor
from bitmind.image_transforms import random_aug_transforms, base_transforms
from bitmind.constants import DATASET_META

FACE_DATASET_META = {
    "real": [
        {"path": "bitmind/ffhq-256", "create_splits": False},
        {"path": "bitmind/celeb-a-hq", "create_splits": False}
    ],
    "fake": [
        {"path": "bitmind/celeb-a-hq___stable-diffusion-xl-base-1.0___256", "create_splits": False},
        {"path": "bitmind/ffhq-256___stable-diffusion-xl-base-1.0", "create_splits": False}
    ]
}

def main(args):
    dataset_processor = TrainingDatasetProcessor(
        dataset_meta=FACE_DATASET_META if args.faces_only else DATASET_META,
        faces_only=args.faces_only,
        transforms={"random_aug_transforms": random_aug_transforms,
                    "base_transforms": base_transforms},
        hf_token=args.hf_token,
        split=args.splits
    )
    
    dataset_processor.process_and_upload_all_datasets(save_locally=args.save_locally, hf_root=args.hf_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and upload training dataset using TrainingDatasetProcessor.")
    
    parser.add_argument("--faces-only", action='store_true', help="Process faces only. If flag is present, faces_only=True.")
    parser.add_argument("--splits", action='store_true', help="Create splits for the dataset. If flag is present, splits=True.")
    parser.add_argument("--save-locally", action='store_true', help="Save preprocessed dicts locally as pickle. If flag is present, save_locally=True.")
    parser.add_argument("--hf-token", type=str, required=True, help="Hugging Face token for dataset upload.")
    parser.add_argument("--hf-root", type=str, required=False, help="Hugging Face repo root (user or company) to upload to.")
    
    args = parser.parse_args()
        
    main(args)
