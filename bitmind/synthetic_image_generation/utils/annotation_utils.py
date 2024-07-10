import json
import logging
import os

def set_logging_level(verbose: int):
    level = logging.WARNING if verbose == 0 else logging.INFO if verbose < 3 else logging.DEBUG
    logging.getLogger().setLevel(level)

def ensure_save_path(path: str) -> str:
    """Ensure that a directory exists; if it does not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def create_annotation_dataset_directory(base_path: str, dataset_name: str) -> str:
    """Create a directory for a dataset with a safe name, replacing any invalid characters."""
    safe_name = dataset_name.replace("/", "_")
    full_path = os.path.join(base_path, safe_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

def save_annotation(dataset_dir: str, image_id, annotation: dict, verbose: int):
    """Save a text annotation to a JSON file if it doesn't already exist."""
    file_path = os.path.join(dataset_dir, f"{image_id}.json")
    if os.path.exists(file_path):
        if verbose > 0: logging.info(f"Annotation for {image_id} already exists - Skipping")
        return -1  # Skip this image as it already has an annotation
    
    with open(file_path, 'w') as f:
        json.dump(annotation, f, indent=4)
        if verbose > 0: logging.info(f"Created {file_path}")

    return 0

def compute_annotation_latency(self, processed_images: int, dataset_time: float, dataset_name: str) -> float:
    if processed_images > 0:
        average_latency = dataset_time / processed_images
        logging.info(f'Average annotation latency for {dataset_name}: {average_latency:.4f} seconds')
        return average_latency
    return 0.0

def list_datasets(base_dir: str) -> list[str]:
    """List all subdirectories in the base directory."""
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

def load_annotations(base_dir: str, dataset: str) -> list[dict]:
    """Load annotations from JSON files within a specified directory."""
    annotations = []
    path = os.path.join(base_dir, dataset)
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), 'r') as file:
                data = json.load(file)
                annotations.append(data)
    return annotations
