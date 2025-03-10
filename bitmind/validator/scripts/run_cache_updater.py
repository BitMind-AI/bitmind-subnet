import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import bittensor as bt
import asyncio
import argparse

from bitmind.validator.cache.image_cache import ImageCache
from bitmind.validator.cache.video_cache import VideoCache
from bitmind.validator.scripts.util import load_validator_info, init_wandb_run
from bitmind.validator.config import (
    IMAGE_DATASETS, 
    VIDEO_DATASETS,
    IMAGE_CACHE_UPDATE_INTERVAL,
    VIDEO_CACHE_UPDATE_INTERVAL,
    IMAGE_PARQUET_CACHE_UPDATE_INTERVAL,
    VIDEO_ZIP_CACHE_UPDATE_INTERVAL,
    REAL_VIDEO_CACHE_DIR,
    REAL_IMAGE_CACHE_DIR,
    SEMISYNTH_VIDEO_CACHE_DIR,
    SEMISYNTH_IMAGE_CACHE_DIR,
    MAX_COMPRESSED_GB,
    MAX_EXTRACTED_GB
)


async def main(args):

    if args.modality in ['all', 'image']:
        bt.logging.info("Starting real image cache updater")
        real_image_cache = ImageCache(
            cache_dir=REAL_IMAGE_CACHE_DIR,
            datasets=IMAGE_DATASETS['real'],
            parquet_update_interval=args.image_parquet_interval,
            image_update_interval=args.image_interval,
            num_parquets_per_dataset=5,
            num_images_per_source=100,
            max_extracted_size_gb=MAX_EXTRACTED_GB,
            max_compressed_size_gb=MAX_COMPRESSED_GB
        )
        real_image_cache.start_updater()

        bt.logging.info("Starting semisynthetic image cache updater")
        semisynth_image_cache = ImageCache(
            cache_dir=SEMISYNTH_IMAGE_CACHE_DIR,
            datasets=IMAGE_DATASETS['semisynthetic'],
            parquet_update_interval=args.image_parquet_interval,
            image_update_interval=args.image_interval,
            num_parquets_per_dataset=5,
            num_images_per_source=100,
            max_extracted_size_gb=MAX_EXTRACTED_GB,
            max_compressed_size_gb=MAX_COMPRESSED_GB
        )
        semisynth_image_cache.start_updater()
    
    if args.modality in ['all', 'video']:
        bt.logging.info("Starting semisynthetic video cache updater")
        semisynth_video_cache = VideoCache(
            cache_dir=SEMISYNTH_VIDEO_CACHE_DIR,
            datasets=VIDEO_DATASETS['semisynthetic'],
            video_update_interval=args.video_interval,
            zip_update_interval=args.video_zip_interval,
            num_zips_per_dataset=2,
            num_videos_per_zip=100,
            max_extracted_size_gb=MAX_EXTRACTED_GB,
            max_compressed_size_gb=MAX_COMPRESSED_GB
        )
        semisynth_video_cache.start_updater()

        bt.logging.info("Starting real video cache updater")
        real_video_cache = VideoCache(
            cache_dir=REAL_VIDEO_CACHE_DIR,
            datasets=VIDEO_DATASETS['real'],
            video_update_interval=args.video_interval,
            zip_update_interval=args.video_zip_interval,
            num_zips_per_dataset=2,
            num_videos_per_zip=100,
            max_extracted_size_gb=MAX_EXTRACTED_GB,
            max_compressed_size_gb=100,
        )
        real_video_cache.start_updater()

    while True:
        bt.logging.info(f"Running cache updaters for: {args.modality}")
        await asyncio.sleep(600)  # Status update every 10 minutes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--modality', type=str, default='all', choices=['all', 'video', 'image'],
        help='Which cache updater(s) to run')
    parser.add_argument(
        '--image-interval', type=int, default=IMAGE_CACHE_UPDATE_INTERVAL,
        help='Update interval for images in hours')
    parser.add_argument(
        '--image-parquet-interval', type=int, default=IMAGE_PARQUET_CACHE_UPDATE_INTERVAL,
        help='Update interval for image parquet files in hours')
    parser.add_argument(
        '--video-interval', type=int, default=VIDEO_CACHE_UPDATE_INTERVAL,
        help='Update interval for videos in hours')
    parser.add_argument(
        '--video-zip-interval', type=int, default=VIDEO_ZIP_CACHE_UPDATE_INTERVAL,
        help='Update interval for video zip files in hours')
    args = parser.parse_args()

    bt.logging.set_info()
    init_wandb_run(run_base_name='cache-updater', **load_validator_info())

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        bt.logging.info("Shutting down cache updaters...")
