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
        transforms={"random_aug_transforms": random_aug_transforms, "base_transforms": base_transforms},
        hf_token=args.hf_token,
        split=args.splits
    )
    dataset_processor.process_and_upload_all_datasets()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and upload training dataset using TrainingDatasetProcessor.")
    
    parser.add_argument("--faces-only", action='store_true', help="Process faces only. If flag is present, faces_only=True.")
    parser.add_argument("--splits", action='store_true', help="Create splits for the dataset. If flag is present, splits=True.")
    parser.add_argument("--hf-token", type=str, required=True, help="Hugging Face token for dataset upload.")
    
    args = parser.parse_args()
    
    main(args)
