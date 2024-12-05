import asyncio
import argparse
import bittensor as bt
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
    MAX_COMPRESSED_GB,
    MAX_EXTRACTED_GB
)


async def main(args):

    image_cache = ImageCache(
        cache_dir=args.image_cache_dir,
        datasets=IMAGE_DATASETS['real'],
        parquet_update_interval=args.image_parquet_interval,
        image_update_interval=args.image_interval,
        num_parquets_per_dataset=5,
        num_images_per_source=100,
        max_extracted_size_gb=MAX_EXTRACTED_GB,
        max_compressed_size_gb=MAX_COMPRESSED_GB
    )
    image_cache.start_updater()
    
    video_cache = VideoCache(
        cache_dir=args.video_cache_dir,
        datasets=VIDEO_DATASETS['real'],
        video_update_interval=args.video_interval,
        zip_update_interval=args.video_zip_interval,
        num_zips_per_dataset=2,
        num_videos_per_zip=50,
        max_extracted_size_gb=MAX_EXTRACTED_GB,
        max_compressed_size_gb=MAX_COMPRESSED_GB
    )
    video_cache.start_updater()
    
    while True:
        bt.logging.info("Caches running...")
        await asyncio.sleep(600)  # Status update every 10 minutes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-cache-dir', type=str, default=REAL_VIDEO_CACHE_DIR,
                        help='Directory to cache video data')
    parser.add_argument('--image-cache-dir', type=str, default=REAL_IMAGE_CACHE_DIR,
                        help='Directory to cache image data')
    parser.add_argument('--image-interval', type=int, default=IMAGE_CACHE_UPDATE_INTERVAL,
                        help='Update interval for images in hours')
    parser.add_argument('--image-parquet-interval', type=int, default=IMAGE_PARQUET_CACHE_UPDATE_INTERVAL,
                        help='Update interval for image parquet files in hours')
    parser.add_argument('--video-interval', type=int, default=VIDEO_CACHE_UPDATE_INTERVAL,
                        help='Update interval for videos in hours')
    parser.add_argument('--video-zip-interval', type=int, default=VIDEO_ZIP_CACHE_UPDATE_INTERVAL,
                        help='Update interval for video zip files in hours')
    args = parser.parse_args()

    init_wandb_run(run_base_name='cache-updater', **load_validator_info())

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        bt.logging.info("Shutting down cache updaters...")
