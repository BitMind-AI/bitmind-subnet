import os
import json
from datasets import load_dataset
from huggingface_hub import HfApi

def dataset_exists_on_hf(hf_dataset_name, token):
    """Check if the dataset exists on Hugging Face."""
    api = HfApi()
    try:
        dataset_info = api.dataset_info(hf_dataset_name, token=token)
        return True
    except Exception as e:
        return False

def numerical_sort(value):
    return int(os.path.splitext(os.path.basename(value))[0])

def load_and_sort_dataset(data_dir, file_type):
    # Get list of filenames in the directory with the given extension
    try:
        if file_type == 'image':
            # List image filenames with common image extensions
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
            filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                         if f.lower().endswith(valid_extensions)]
        elif file_type == 'json':
            # List json filenames
            filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                         if f.lower().endswith('.json')]
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if not filenames:
            raise FileNotFoundError(f"No files with the extension '{file_type}' \
                                    found in directory '{data_dir}'")
    
        # Sort filenames numerically (0, 1, 2, 3, 4). Necessary because
        # HF datasets are ordered by string (0, 1, 10, 11, 12). 
        sorted_filenames = sorted(filenames, key=numerical_sort)
        
        # Load the dataset with sorted filenames
        if file_type == 'image':
            return load_dataset("imagefolder", data_files=sorted_filenames)
        elif file_type == 'json':
            return load_dataset("json", data_files=sorted_filenames)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
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

def save_as_json(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    # Iterate through rows in dataframe
    for index, row in df.iterrows():
        file_path = os.path.join(output_dir, f"{row['id']}.json")
        # Convert the row to a dictionary and save it as JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(row.to_dict(), f, ensure_ascii=False, indent=4)
