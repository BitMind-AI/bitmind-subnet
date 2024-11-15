from typing import Optional, List, Tuple
from io import BytesIO
from pathlib import Path
from PIL import Image
from datetime import datetime
import bittensor as bt
import multiprocessing as mp
from multiprocessing import Manager
from zipfile import ZipFile
import signal
import sys
import time
import json
import shutil
import ffmpeg
import numpy as np
import random
import asyncio

from bitmind.validator.video_datasets import download_openvid1m_zips
from bitmind.constants import VIDEO_CACHE_DIR, COMPRESSED_VIDEO_CACHE_DIR
from bitmind.validator.video_utils import (
    search_and_download_youtube_videos, 
    VideoMetadata, 
    clip_video, 
    get_video_duration
)


class VideoCache:
    def __init__(
        self, 
        cache_dir=VIDEO_CACHE_DIR, 
        compressed_dir=COMPRESSED_VIDEO_CACHE_DIR,
        update_interval: int = 2 * 60 * 60,
        num_videos_per_source: int = 10, 
        use_youtube: bool = False
    ):   
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.compressed_dir = Path(compressed_dir)
        self.compressed_dir.mkdir(exist_ok=True)

        self.update_interval = update_interval
        self.num_videos_per_source = num_videos_per_source
        self.use_youtube = use_youtube

        self.compressed_source_dirs = list(self.compressed_dir.glob('*/'))
        bt.logging.info(f"Setting up video cache")
        bt.logging.info(f"Cache dir:{cache_dir}")
        bt.logging.info(f"Video source dir: {compressed_dir}")

        self.metadata_path = self.cache_dir / 'video_metadata.json'
        self.metadata = {}
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)
        self.video_files = list(self.metadata.keys()) 

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.get_event_loop()

        self._refresh_cache()  # initial cache population
        self.updater_task = self.loop.create_task(self._run_cache_updater())

    def sample_random_video_frames(self, num_seconds: int = 6) -> dict:
        """Sample random frames from a random video in the cache"""
        if not self.video_files:
            bt.logging.warning("No videos available in cache")
            return None
        
        video_path = random.choice(self.video_files)
        if not Path(video_path).exists():
            bt.logging.error(f"Selected video {video_path} not found")
            return None
            
        try:
            duration = get_video_duration(str(video_path))
                
            start_time = random.uniform(0, np.clip(duration - num_seconds, 0, None))
            temp_file = clip_video(video_path, str(start_time), num_seconds)
            frames = []
            
            for second in range(num_seconds):
                out_bytes, _ = (
                    ffmpeg
                    .input(temp_file.name, ss=str(second))
                    .filter('select', 'gte(n,0)')
                    .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
                    .run(capture_stdout=True, quiet=True)
                )
                frames.append(Image.open(BytesIO(out_bytes)))
                    
            return {
                'video': frames,
                'path': video_path,
                **self.metadata[video_path]
            }
        except Exception as e:
            bt.logging.error(f"Error sampling frames from {video_path}: {e}")
            return None

    async def _run_cache_updater(self):
        """
        Asynchronously updates the cache at a regular interval
        """
        while True:
            try:
                bt.logging.info("Running cache refresh...")
                self._refresh_cache()
                bt.logging.info("Cache refresh complete. Sleeping for 2 hours...")
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                bt.logging.error(f"Error in cache update: {e}")
                await asyncio.sleep(60)

    def _refresh_cache(self):
        bt.logging.info(f"Starting video cache refresh")
        try:
            if len(list(self.compressed_dir.iterdir())) == 0:
                bt.logging.info("No video data sources found. Downloading..")
                download_openvid1m_zips(self.compressed_dir, download_all=False, num_zips=1)
    
            prior_cache_files = list(self.cache_dir.glob('*.mp4'))

            new_cache_files = self._extract_random_videos(self.compressed_dir)

            if self.use_youtube:
                youtube_metas = search_and_download_youtube_videos(
                    'dummy query', num_videos_per_source)
                if youtube_metas:
                    new_cache_files.extend([Path(meta.video_path) for meta in youtube_metas])
                    bt.logging.info(f"Added {len(youtube_metas)} YouTube videos")
  
            if new_cache_files:
                self._save_metadata(new_cache_files)
                bt.logging.info(f"{len(new_cache_files)} new videos added to cache")
                bt.logging.info(f"Removing {len(prior_cache_files)} previously cached videos")
                for file in prior_cache_files:
                    try:
                        file.unlink()
                    except Exception as e:
                        bt.logging.error(f"Error removing file {file}: {e}")
            else:
                bt.logging.error("No videos were added to cache")

        except Exception as e:
            bt.logging.error(f"Error during video refresh: {e}")
            raise

    def _extract_random_videos(self, source_dir) -> List[Path]:
        """Extract random videos from zip files in compressed directory"""
        extracted_files = []
        zip_files = list(source_dir.glob('*.zip'))
        if not zip_files:
            bt.logging.warning(f"No zip files found in {self.compressed_dir}")
            return extracted_files

        zip_path = random.choice(zip_files)
        try:
            with ZipFile(zip_path) as zip_file:
                video_files = [
                    f for f in zip_file.namelist() 
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
                ]
                
                if not video_files:
                    bt.logging.warning(f"No video files found in {zip_path}")
                    return extracted_files
                    
                selected_videos = random.sample(
                    video_files, 
                    min(self.num_videos_per_source, len(video_files))
                )
                bt.logging.debug(f'{len(selected_videos)} videos sampled')
                bt.logging.debug(selected_videos)
                
                for video in selected_videos:
                    try:
                        filename = Path(video).name
                        temp_path = zip_file.extract(video, path=self.cache_dir)
                        desired_path = Path(self.cache_dir) / filename
                        Path(temp_path).rename(desired_path)
                        extracted_files.append(desired_path)
                        bt.logging.info(f"Extracted {filename} from {zip_path}")
                    except Exception as e:
                        bt.logging.error(f"Error extracting {video}: {e}")
        except Exception as e:
            bt.logging.error(f"Error processing zip file {zip_path}: {e}")
            
        return extracted_files

    def _save_metadata(self, video_paths: List[Path]):
        """Save video metadata as JSON"""
        new_metadata = {}
        
        for path in video_paths:
            try:
                duration = get_video_duration(str(path))
                new_metadata[str(path)] = {
                    'video_id': path.stem,
                    'length': duration,
                    'start_time': 0,
                    'end_time': duration,
                    'refreshed_at': datetime.now().isoformat(),
                }
            except Exception as e:
                bt.logging.error(f"Error creating metadata for {path}: {e}")
                continue
        
        self.metadata = new_metadata
        self.video_files = list(self.metadata.keys())

        # atomic write
        temp_path = self.metadata_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        temp_path.replace(self.metadata_path)

