import yaml

import wandb
import bittensor as bt

import bitmind
from bitmind.validator.config import (    
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

 
def init_wandb_run(run_base_name: str, uid: str, hotkey: str, netuid: int, full_path: str) -> None:
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
    run_name = f'{run_base_name}-{uid}-{bitmind.__version__}'
    
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
        return wandb.init(
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