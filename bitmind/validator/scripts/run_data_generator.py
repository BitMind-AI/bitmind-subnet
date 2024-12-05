import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-cache-dir', type=str, default=REAL_IMAGE_CACHE_DIR,
                      help='Directory containing real images to use as reference')
    parser.add_argument('--output-dir', type=str, default=SYNTH_CACHE_DIR,
                      help='Directory to save generated synthetic data')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to run generation on (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=3,
                      help='Number of images to generate per batch')
    args = parser.parse_args()

    init_wandb_run(run_base_name='data-generator', **load_validator_info())

    image_cache = ImageCache(args.image_cache_dir)
    while True:
        if image_cache._extracted_cache_empty():
            bt.logging.info("SyntheticDataGenerator waiting for real image cache to populate")
            time.sleep(5)
            continue
        bt.logging.info("Image cache was populated! Proceeding to data generation")
        break

    sdg = SyntheticDataGenerator(
        prompt_type='annotation',
        use_random_t2vis_model=True,
        device=args.device,
        image_cache=image_cache,
        output_dir=args.output_dir)

    bt.logging.info("Starting standalone data generator service")
    sdg.batch_generate(batch_size=1)
    while True:
        try:
            sdg.batch_generate(batch_size=args.batch_size)
        except Exception as e:
            bt.logging.error(f"Error in batch generation: {str(e)}")
            time.sleep(5)
