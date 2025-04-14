import os
from bitmind.synthetic_data_generation import SyntheticDataGenerator
from bitmind.validator.config import MODEL_NAMES, IMAGE_ANNOTATION_MODEL, TEXT_MODERATION_MODEL
import bittensor as bt


def is_model_cached(model_name):
    """
    Check if the specified model is cached by looking for its directory in the Hugging Face cache.

    Args:
        model_name (str): The name of the model to check.

    Returns:
        bool: True if the model is cached, False otherwise.
    """
    cache_dir = os.path.expanduser('~/.cache/huggingface/')
    # Format the directory name correctly by replacing each slash with double dashes
    model_dir = f"models--{model_name.replace('/', '--')}"

    # Construct the full path to where the model directory should be
    model_path = os.path.join(cache_dir, model_dir)

    # Check if the model directory exists
    if os.path.isdir(model_path):
        print(f"{model_name} is in HF cache. Skipping....")
        return True
    else:
        print(f"{model_name} is not cached. Downloading....")
        return False


def main():
    """
    Main function to verify and download validator models.

    This function checks if the required models are cached and downloads them if necessary.
    It also initializes and loads diffusers for uncached models.
    """
    bt.logging.info("Verifying validator model downloads....")
    synthetic_image_generator = SyntheticDataGenerator(
        prompt_type='annotation',
        image_cache='test',
        use_random_model=True
    )

    # Check and load annotation and moderation models if not cached
    if not is_model_cached(IMAGE_ANNOTATION_MODEL) or not is_model_cached(TEXT_MODERATION_MODEL):
        synthetic_image_generator.prompt_generator.load_models()
        synthetic_image_generator.prompt_generator.clear_gpu()

    # Initialize and load diffusers if not cached
    for model_name in MODEL_NAMES:
        if not is_model_cached(model_name):
            synthetic_image_generator = SyntheticDataGenerator(
                prompt_type=None,
                use_random_model=False,
                model_name=model_name
            )
            synthetic_image_generator.load_model(model_name)
            synthetic_image_generator.clear_gpu()


if __name__ == "__main__":
    main()
