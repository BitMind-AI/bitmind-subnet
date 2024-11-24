import asyncio
from datetime import datetime
from io import BytesIO
import json
import multiprocessing as mp
from multiprocessing import Manager
from pathlib import Path
import random
import shutil
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import bittensor as bt
import ffmpeg
import numpy as np
from PIL import Image

from bitmind.constants import (
    VIDEO_CACHE_DIR,
    COMPRESSED_VIDEO_CACHE_DIR,
    VIDEO_DATASET_META
)
from bitmind.validator.video_datasets import download_zips
from bitmind.validator.video_utils import (
    search_and_download_youtube_videos,
    clip_video,
    get_video_duration,
    seconds_to_str,
    get_most_recent_update_time,
    is_zip_complete
)


class VideoCache:
    """
    A class to manage video caching and processing operations.

    This class handles the caching, updating, and sampling of video files from 
    various sources, including compressed archives and optionally YouTube. It 
    maintains both a compressed cache of source files and an extracted cache
    of video files ready for processing.

    Attributes:
        cache_dir: Directory for storing extracted video files.
        compressed_dir: Directory for storing compressed source files.
        video_update_interval: Time in hours between video cache updates.
        zip_update_interval: Time in hours between zip cache updates.
        num_videos_per_source: Number of videos to extract from each source.
        use_youtube: Whether to include YouTube videos in the cache.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = VIDEO_CACHE_DIR,
        compressed_dir: Union[str, Path] = COMPRESSED_VIDEO_CACHE_DIR,
        video_update_interval: int = 2,
        zip_update_interval: int = 24,
        num_videos_per_source: int = 10,
        use_youtube: bool = False
    ) -> None:
        """
        Initialize the VideoCache with specified parameters.

        Args:
            cache_dir: Path to store extracted video files.
            compressed_dir: Path to store compressed source files.
            video_update_interval: Hours between video cache updates.
            zip_update_interval: Hours between zip cache updates.
            num_videos_per_source: Number of videos to extract per source.
            use_youtube: Whether to include YouTube videos.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.compressed_dir = Path(compressed_dir)
        self.compressed_dir.mkdir(exist_ok=True)
        self.video_update_interval = video_update_interval * 60 * 60
        self.zip_update_interval = zip_update_interval * 60 * 60
        self.num_videos_per_source = num_videos_per_source
        self.use_youtube = use_youtube

        bt.logging.info(f"Setting up video cache {cache_dir}")
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.get_event_loop()

        # blocking calls before starting async to ensure data are available
        bt.logging.info("Initializing zip cache")
        self._clear_incomplete_zips()
        if self._zip_cache_is_empty():
            self._refresh_zip_cache()

        bt.logging.info("Initializing video cache")
        if self._video_cache_is_empty():
            self._refresh_video_cache()

        self._video_updater_task = self.loop.create_task(
            self._run_video_cache_updater()
        )
        self._zip_updater_task = self.loop.create_task(
            self._run_zip_cache_updater()
        )

    async def _run_video_cache_updater(self) -> None:
        """
        Asynchronously refresh cached video files according to update interval.
        """
        while True:
            try:
                last_update = get_most_recent_update_time(self.cache_dir)
                time_elapsed = time.time() - last_update

                if time_elapsed >= self.video_update_interval:
                    bt.logging.info("Running video cache refresh...")
                    self._refresh_video_cache()
                    bt.logging.info("Video cache refresh complete.")

                sleep_time = max(0, self.video_update_interval - time_elapsed)
                bt.logging.info(f"Sleeping for {seconds_to_str(sleep_time)}")
                await asyncio.sleep(sleep_time)
            except Exception as e:
                bt.logging.error(f"Error in video cache update: {e}")
                await asyncio.sleep(60)

    async def _run_zip_cache_updater(self) -> None:
        """
        Asynchronously refresh cached zip files according to update interval.
        """
        while True:
            try:
                last_update = get_most_recent_update_time(self.compressed_dir)
                time_elapsed = time.time() - last_update

                if time_elapsed >= self.zip_update_interval:
                    bt.logging.info("Running zip cache refresh...")
                    self._refresh_zip_cache()
                    bt.logging.info("Zip cache refresh complete.")

                sleep_time = max(0, self.zip_update_interval - time_elapsed)
                bt.logging.info(f"Sleeping for {seconds_to_str(sleep_time)}")
                await asyncio.sleep(sleep_time)
            except Exception as e:
                bt.logging.error(f"Error in zip cache update: {e}")
                await asyncio.sleep(60)

    def _get_video_files(self) -> List[Path]:
        """Get list of all video files in cache directory."""
        return list(self.cache_dir.iterdir())

    def _video_cache_is_empty(self) -> bool:
        """Check if video cache directory is empty."""
        return len(self._get_video_files()) == 0

    def _get_zip_files(self) -> List[Path]:
        """Get list of all zip files in compressed directory."""
        return list(self.compressed_dir.iterdir())

    def _zip_cache_is_empty(self) -> bool:
        """Check if zip cache directory is empty."""
        return len(self._get_zip_files()) == 0

    def _clear_incomplete_zips(self) -> None:
        """Remove any incomplete or corrupted zip files from cache."""
        for path in self._get_zip_files():
            if path.suffix != '.txt' and not is_zip_complete(path):
                try:
                    path.unlink()
                    bt.logging.warning(f"Removed incomplete or corrupt zip file {path}")
                except Exception as e:
                    bt.logging.error(
                        f"Error removing incomplete or corrupt zip file {path}: {e}"
                    )

    def _refresh_video_cache(self) -> None:
        """
        Refresh the video cache with new random selections.
        
        Clears existing cached videos and extracts new ones from the compressed
        sources. Optionally includes YouTube videos if enabled.
        """
        bt.logging.info("Starting video cache refresh")
        try:
            prior_cache_files = list(self.cache_dir.glob('*.mp4'))
            new_cache_files = self._extract_random_videos(self.compressed_dir)

            if new_cache_files:
                bt.logging.info(f"{len(new_cache_files)} new videos added to cache")
                bt.logging.info(
                    f"Removing {len(prior_cache_files)} previously cached videos"
                )
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

    def _refresh_zip_cache(self) -> None:
        """
        Refresh the compressed file cache with new downloads.

        Downloads new zip files from configured sources and removes old ones.
        """
        bt.logging.info("Starting video cache refresh")
        try:
            prior_cache_files = list(self.compressed_dir.glob('*.zip'))
            bt.logging.info("Downloading video data")

            new_cache_files: List[Path] = []
            for meta in VIDEO_DATASET_META['real']:
                new_cache_files += await download_zips(
                    meta["base_zip_url"],
                    self.compressed_dir,
                    meta["max_zip_id"],
                    meta.get("min_zip_id", 0),
                    num_zips=1,
                    download_all=False,
                    err_handler_fn=meta.get("err_handler", None)
                )

            if new_cache_files:
                bt.logging.info(f"{len(new_cache_files)} new video zips added to cache")
                bt.logging.info(
                    f"Removing {len(prior_cache_files)} previously cached video zips"
                )
                for file in prior_cache_files:
                    try:
                        file.unlink()
                    except Exception as e:
                        bt.logging.error(f"Error removing file {file}: {e}")
            else:
                bt.logging.error("No videos were added to cache")

        except Exception as e:
            bt.logging.error(f"Error during video zip refresh: {e}")
            raise

    def _extract_random_videos(self, source_dir: Path) -> List[Path]:
        """
        Extract random videos from zip files in compressed directory.

        Args:
            source_dir: Directory containing compressed video archives.

        Returns:
            List of paths to extracted video files.
        """
        extracted_files: List[Path] = []
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

    def sample_random_video_frames(
        self,
        num_seconds: int = 6
    ) -> Optional[Dict[str, Union[List[Image.Image], str, float]]]:
        """
        Sample random frames from a random video in the cache.

        Args:
            num_seconds: Number of consecutive frames to sample.

        Returns:
            Dictionary containing:
                - video: List of sampled video frames as PIL Images
                - path: Path to source video file
                - dataset: Name of source dataset
                - total_duration: Total video duration in seconds
                - sampled_length: Number of seconds sampled
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
        frames: List[Image.Image] = []
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