import time

import bittensor as bt

from bitmind.validator.scripts.util import load_validator_info, init_wandb_run
from bitmind.synthetic_data_generation import SyntheticDataGenerator
from bitmind.validator.cache import ImageCache
from bitmind.validator.config import (
    REAL_IMAGE_CACHE_DIR,
    SYNTH_CACHE_DIR
)


if __name__ == '__main__':

    init_wandb_run(run_base_name='data-generator', **load_validator_info())

    image_cache = ImageCache(REAL_IMAGE_CACHE_DIR)
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
