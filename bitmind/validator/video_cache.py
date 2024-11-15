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
    clip_video,
    get_video_duration,
    seconds_to_str
)


class VideoCache:
    """A class to manage video caching and frame extraction operations.

    This class handles downloading, extracting, and managing video files from various
    sources including compressed archives and YouTube. It provides functionality to
    sample random frames from cached videos.

    Attributes:
        cache_dir (Path): Directory to store extracted video files.
        compressed_dir (Path): Directory to store compressed video archives.
        update_interval (int): Time in seconds between cache updates.
        num_videos_per_source (int): Number of videos to extract per source.
        use_youtube (bool): Whether to include YouTube videos in the cache.
    """

    def __init__(
        self,
        cache_dir=VIDEO_CACHE_DIR,
        compressed_dir=COMPRESSED_VIDEO_CACHE_DIR,
        update_interval: int = 2 * 60 * 60,
        num_videos_per_source: int = 10,
        use_youtube: bool = False
    ):
        """Initialize the VideoCache with specified parameters.

        Args:
            cache_dir: Path to store extracted video files.
            compressed_dir: Path to store compressed video archives.
            update_interval: Time in seconds between cache updates.
            num_videos_per_source: Number of videos to extract per source.
            use_youtube: Whether to include YouTube videos in the cache.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.compressed_dir = Path(compressed_dir)
        self.compressed_dir.mkdir(exist_ok=True)

        self.update_interval = update_interval
        self.num_videos_per_source = num_videos_per_source
        self.use_youtube = use_youtube

        bt.logging.info(f"Setting up video cache {cache_dir}")

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.get_event_loop()

        self._refresh_cache()  # blocking to ensure data are available
        self.updater_task = self.loop.create_task(self._run_cache_updater())

    async def _run_cache_updater(self):
        """Asynchronously update the cache at specified intervals."""
        while True:
            try:
                bt.logging.info("Running cache refresh...")
                self._refresh_cache()
                bt.logging.info("Video cache refresh complete.")
                bt.logging.info(f"Sleeping for {seconds_to_str(self.update_interval)}")
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                bt.logging.error(f"Error in cache update: {e}")
                await asyncio.sleep(60)

    def sample_random_video_frames(self, num_seconds: int = 6) -> Optional[dict]:
        """Sample random frames from a random video in the cache.

        Args:
            num_seconds: Number of consecutive frames to sample.

        Returns:
            dict: Contains video frames, path, dataset info, and duration details.
                 Returns None if no videos are available or extraction fails.
        """
        video_files = list(self.cache_dir.glob('*.mp4'))
        if not video_files:
            bt.logging.warning("No videos available in cache")
            return None

        video_path = random.choice(video_files)
        if not Path(video_path).exists():
            bt.logging.error(f"Selected video {video_path} not found")
            return None

        duration = get_video_duration(str(video_path))
        start_time = random.uniform(0, np.clip(duration - num_seconds, 0, None))
        frames = []

        try:
            for second in range(num_seconds):
                out_bytes, _ = (
                    ffmpeg
                    .input(str(video_path), ss=str(start_time + second))
                    .filter('select', 'eq(n,0)')
                    .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
                    .run(capture_stdout=True, capture_stderr=True)
                )

                try:
                    frame = Image.open(BytesIO(out_bytes))
                    frames.append(frame)
                except Exception as e:
                    bt.logging.warning(
                        f"Failed to decode frame at second {second}, stopping extraction"
                    )
                    break

        except ffmpeg.Error as e:
            bt.logging.warning(f"FFmpeg error at second {second}, stopping extraction")
            bt.logging.warning(e.stderr.decode())

        return {
            'video': frames,
            'path': str(video_path),
            'dataset': str(Path(video_path).name.split('_')[0]),
            'total_duration': duration,
            'sampled_length': num_seconds
        }

    def _refresh_cache(self):
        """Refresh the video cache with new random selections."""
        bt.logging.info("Starting video cache refresh")
        try:
            if len(list(self.compressed_dir.iterdir())) == 0:
                bt.logging.info("No video data sources found. Downloading..")
                download_openvid1m_zips(
                    self.compressed_dir,
                    download_all=False,
                    num_zips=1
                )

            prior_cache_files = list(self.cache_dir.glob('*.mp4'))
            new_cache_files = self._extract_random_videos(self.compressed_dir)

            if self.use_youtube:
                youtube_metas = search_and_download_youtube_videos(
                    'dummy query',
                    num_videos_per_source
                )
                if youtube_metas:
                    new_cache_files.extend([Path(meta.video_path) for meta in youtube_metas])
                    bt.logging.info(f"Cached {len(youtube_metas)} YouTube videos")

            if new_cache_files:
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
        """Extract random videos from zip files in compressed directory.

        Args:
            source_dir: Directory containing compressed video archives.

        Returns:
            List[Path]: Paths to extracted video files.
        """
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
                        target_path = Path(self.cache_dir) / '_'.join(
                            [zip_path.name.split('.zip')[0], filename]
                        )
                        Path(temp_path).rename(target_path)
                        extracted_files.append(target_path)
                        bt.logging.info(f"Extracted {filename} from {zip_path}")
                    except Exception as e:
                        bt.logging.error(f"Error extracting {video}: {e}")
        except Exception as e:
            bt.logging.error(f"Error processing zip file {zip_path}: {e}")

        return extracted_files