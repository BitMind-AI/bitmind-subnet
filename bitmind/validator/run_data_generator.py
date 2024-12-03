import time
import yaml

import wandb
import bittensor as bt

import bitmind
from bitmind.synthetic_data_generation import SyntheticDataGenerator
from bitmind.validator.cache import ImageCache
from bitmind.validator.config import (
    REAL_IMAGE_CACHE_DIR,
    SYNTH_CACHE_DIR,
    WANDB_ENTITY,
    TESTNET_WANDB_PROJECT,
    MAINNET_WANDB_PROJECT,
    MAINNET_UID,
    VALIDATOR_INFO_PATH
)

def load_validator_info():
    try:
        with open(VALIDATOR_INFO_PATH, 'r') as f:
            validator_info = yaml.safe_load(f)
        bt.logging.info(f"Loaded validator info from {VALIDATOR_INFO_PATH}")
    except FileNotFoundError:
        bt.logging.error(f"Could not find validator info at {VALIDATOR_INFO_PATH}")
        validator_info = {
            'uid': 'NotFound', 
            'hotkey': 'NotFound', 
            'full_path': 'NotFound', 
            'netuid': TESTNET_WANDB_PROJECT
        }
    except yaml.YAMLError:
        bt.logging.error(f"Could not parse validator info at {VALIDATOR_INFO_PATH}")
        validator_info = {
            'uid': 'ParseError', 
            'hotkey': 'ParseError',
            'full_path': 'ParseError', 
            'netuid': TESTNET_WANDB_PROJECT
        }
    return validator_info

 
def init_wandb_run(uid: str, hotkey: str, netuid: int, full_path: str) -> None:
    """
    Initialize a Weights & Biases run for tracking the validator.

    Args:
        vali_uid: The validator's uid
        vali_hotkey: The validator's hotkey address
        netuid: The network ID (mainnet or testnet)
        vali_full_path: Validator's bittensor directory 

    Returns:
        None
    """
    run_name = f'data-generator-{uid}-{bitmind.__version__}'
    
    config = {
        'run_name': run_name,
        'uid': uid,
        'hotkey': hotkey,
        'version': bitmind.__version__
    }
    
    wandb_project = TESTNET_WANDB_PROJECT
    if netuid == MAINNET_UID:
        wandb_project = MAINNET_WANDB_PROJECT

    # Initialize the wandb run for the single project
    bt.logging.info(f"Initializing W&B run for '{WANDB_ENTITY}/{wandb_project}'")
    try:
        run = wandb.init(
            name=run_name,
            project=wandb_project,
            entity=WANDB_ENTITY,
            config=config,
            dir=full_path,
            reinit=True
        )
    except wandb.UsageError as e:
        bt.logging.warning(e)
        bt.logging.warning("Did you run wandb login?")
        return

if __name__ == '__main__':

    init_wandb_run(**load_validator_info())

    image_cache = ImageCache(REAL_IMAGE_CACHE_DIR, datasets=None, run_updater=False)
    while True:
        if image_cache._extracted_cache_empty():
            bt.logging.info("SyntheticDataGenerator waiting for real image cache to populate")
            time.sleep(5)
            continue
        bt.logging.info("Image cache was populated! Proceeding to data generation")
        break

    sgd = SyntheticDataGenerator(
        prompt_type='annotation',
        use_random_t2vis_model=True,
        device='cuda',
        image_cache=image_cache,
        output_dir=SYNTH_CACHE_DIR)

    bt.logging.info("Starting standalone data generator service")
    while True:
        try:
            sgd.batch_generate(batch_size=1)
            time.sleep(1)
        except Exception as e:
            bt.logging.error(f"Error in batch generation: {str(e)}")
            time.sleep(5)
