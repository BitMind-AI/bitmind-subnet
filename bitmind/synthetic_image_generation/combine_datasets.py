import argparse
from datasets import load_dataset, concatenate_datasets
from utils.hugging_face_utils import dataset_exists_on_hf, upload_to_huggingface


def calculate_splits(total_items, num_splits):
    """Calculate and return list of tuples with start and end indices for each split."""
    chunk_size = total_items // num_splits
    return [(i * chunk_size, (i + 1) * chunk_size - 1) if i < num_splits - 1 
            else (i * chunk_size, total_items - 1) for i in range(num_splits)]
    

def load_and_combine_datasets(hf_org, dataset_name, model, px, splits):
    dataset_parts = []
    
    # Iterate over each split and load the corresponding dataset
    for start_index, end_index in splits:
        dataset_id = f"{hf_org}/{dataset_name}___{start_index}-to-{end_index}___{model}"
        if px:
           dataset_id += f" ___{px}"
        try:
            dataset = load_dataset(dataset_id)
            dataset_parts.append(dataset['train'])
            print(f"Loaded dataset: {dataset_id}")
        except Exception as e:
            print(f"Failed to load dataset {dataset_id}: {str(e)}")
    
    # Concatenate all datasets in the correct order
    combined_dataset = concatenate_datasets(dataset_parts, split='train')
    print("All datasets combined successfully.")
    return combined_dataset
    

def parse_arguments():
    """
    Before running, authenticate with command line to upload to Hugging Face:
    huggingface-cli login
    
    Do not add token as Git credential.

    Example usage:
    pm2 start combine_datasets.py --name "combine celeb-a-hq datasets" --no-autorestart \
    -- "bitmind" "celeb-a-hq" "stable-diffusion-xl-base-1.0" 30000 4 \
        "YOUR_HF_TOKEN" --px 256
    """
    parser = argparse.ArgumentParser(description='Load and combine Hugging Face datasets.')
    parser.add_argument('hf_org', type=str, help='Hugging Face organization name')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('model_name', type=str, help='Name of the diffusion model')
    parser.add_argument('total_items', type=int, help='Total number of items in the dataset')
    parser.add_argument('num_splits', type=int, help='Number of splits')
    parser.add_argument('hf_token', type=str, help='Token for uploading to Hugging Face.')
    parser.add_argument('--px', type=int, default=None, help='Dimensions (ex. 256) of images.')

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    combined_dataset_name = f"{args.hf_org}/{args.dataset_name}___{args.model_name}"
    if args.px:
        combined_dataset_name += f"___{args.px}"
    # Calculate splits based on total items and number of splits
    splits = calculate_splits(args.total_items, args.num_splits)
    print(splits)
    
    combined_dataset = load_and_combine_datasets(args.hf_org, args.dataset_name,
                                                 args.model_name, args.px, splits)
    
    print("Dataset ready for further processing or uploading.")
    if dataset_exists_on_hf(combined_dataset_name, args.hf_token):
        print("Combined dataset exist on Hugging Face.")
    else:
        print(f"Uploading {combined_dataset_name} to Hugging Face.")
        upload_to_huggingface(combined_dataset, combined_dataset_name, args.hf_token)
        print(f"Finished uploading {combined_dataset_name} to Hugging Face.")
        

if __name__ == "__main__":
    main()