from datetime import datetime
from io import BytesIO
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import bittensor as bt
import huggingface_hub as hf_hub
import numpy as np
import ffmpeg
from PIL import Image

from .base_cache import BaseCache
from .download import download_files, list_hf_files
from .extract import extract_videos_from_zip
from .util import is_zip_complete
from bitmind.validator.video_utils import get_video_duration


class VideoCache(BaseCache):
    """
    A class to manage video caching and processing operations.
    
    This class handles the caching, updating, and sampling of video files from 
    compressed archives and optionally YouTube. It maintains both a compressed
    cache of source files and an extracted cache of video files ready for processing.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        datasets: dict,
        video_update_interval: int = 2,
        zip_update_interval: int = 24,
        num_videos_per_source: int = 10,
        run_updater: bool = True
    ) -> None:
        """
        Initialize the VideoCache.
        
        Args:
            cache_dir: Path to store extracted video files
            video_update_interval: Hours between video cache updates
            zip_update_interval: Hours between zip cache updates
            num_videos_per_source: Number of videos to extract per source
            use_youtube: Whether to include YouTube videos
        """
        super().__init__(
            cache_dir=cache_dir,
            datasets=datasets,
            extracted_update_interval=video_update_interval,
            compressed_update_interval=zip_update_interval,
            num_samples_per_source=num_videos_per_source,
            file_extensions=['.mp4', '.avi', '.mov', '.mkv'],
            run_updater=run_updater
        )

    def _clear_incomplete_sources(self) -> None:
        """Remove any incomplete or corrupted zip files from cache."""
        for path in self._get_compressed_files():
            if path.suffix == '.zip' and not is_zip_complete(path):
                try:
                    path.unlink()
                    bt.logging.warning(f"Removed incomplete zip file {path}")
                except Exception as e:
                    bt.logging.error(f"Error removing incomplete zip {path}: {e}")

    def _refresh_compressed_cache(self, n_zips_per_source=5) -> None:
        """
        Refresh the compressed file cache with new downloads.
        """
        try:
            prior_files = list(self.compressed_dir.glob('*.zip'))

            new_files: List[Path] = []
            for source in self.datasets:
                zip_files = list_hf_files(repo_id=source['path'], extension='zip')
                remote_zip_paths = [
                    f"https://huggingface.co/datasets/{source['path']}/resolve/main/{f}"
                    for f in zip_files
                ]
                bt.logging.info(f"Downloading {n_zips_per_source} from {source['path']}")
                new_files += download_files(
                    urls=np.random.choice(remote_zip_paths, n_zips_per_source),
                    output_dir=self.compressed_dir)

            if new_files:
                bt.logging.info(f"{len(new_files)} new files added to cache")
            else:
                bt.logging.error("No new files were added to cache")

        except Exception as e:
            bt.logging.error(f"Error during compressed refresh: {e}")
            raise

    def _refresh_extracted_cache(self) -> None:
        """
        Refresh the video cache with new random selections.
        
        Clears existing cached videos and extracts new ones from the compressed
        sources.
        """
        prior_cache_files = self._get_cached_files()
        new_cache_files = self._extract_random_videos()
        if new_cache_files:
            bt.logging.info(f"{len(new_cache_files)} new videos added to cache")
        else:
            bt.logging.error("No videos were added to cache")

    def _extract_random_videos(self) -> List[Path]:
        """
        Extract random videos from zip files in compressed directory.
        
        Returns:
            List of paths to extracted video files.
        """
        zip_files = list(self.compressed_dir.glob('*.zip'))
        extracted_files = []
        if not zip_files:
            bt.logging.warning(f"No zip files found in {self.compressed_dir}")
            return extracted_files

        for zip_file in zip_files:
            try:
                extracted_files += extract_videos_from_zip(
                    zip_file,
                    self.cache_dir, 
                    self.num_samples_per_source)
            except Exception as e:
                bt.logging.error(f"Error processing zip file {zip_file}: {e}")

        return extracted_files

    def sample(
        self,
        num_seconds: int = 6
    ) -> Optional[Dict[str, Union[List[Image.Image], str, float]]]:
        """
        Sample random frames from a random video in the cache.

        Args:
            num_seconds: Number of consecutive frames to sample

        Returns:
            Dictionary containing:
                - video: List of sampled video frames as PIL Images
                - path: Path to source video file
                - dataset: Name of source dataset
                - total_duration: Total video duration in seconds
                - sampled_length: Number of seconds sampled
            Returns None if no videos are available or extraction fails.
        """
        video_files = self._get_cached_files()
        if not video_files:
            bt.logging.warning("No videos available in cache")
            return None

        video_path = random.choice(video_files)
        if not Path(video_path).exists():
            bt.logging.error(f"Selected video {video_path} not found")
            return None

        duration = get_video_duration(str(video_path))
        start_time = random.uniform(0, max(0, duration - num_seconds))
        frames: List[Image.Image] = []

        start_time = random.uniform(0, max(0, duration - num_seconds))
        bt.logging.info(f'Extracting frames starting atq {start_time:.2f}s')

        for second in range(num_seconds):
            timestamp = start_time + second
            
            try:
                # extract frames
                out_bytes, err = (
                    ffmpeg
                    .input(str(video_path), ss=str(timestamp))
                    .filter('select', 'eq(n,0)')
                    .output('pipe:', 
                           vframes=1,
                           format='image2',
                           vcodec='mjpeg',
                           loglevel='error',  # silence ffmpeg output
                           **{'qscale:v': 2}  # Better quality JPEG
                    )
                    .run(capture_stdout=True, capture_stderr=True)
                )

                if not out_bytes:
                    bt.logging.error(f'No data received for frame at {timestamp}s')
                    continue

                try:
                    frame = Image.open(BytesIO(out_bytes))
                    frame.load()  # Verify image can be loaded
                    frames.append(frame)
                    bt.logging.debug(f'Successfully extracted frame at {timestamp}s')
                except Exception as e:
                    bt.logging.error(f'Failed to process frame at {timestamp}s: {e}')
                    continue

            except ffmpeg.Error as e:
                bt.logging.error(f'FFmpeg error at {timestamp}s: {e.stderr.decode()}')
                continue

        return {
            'video': frames,
            'path': str(video_path),
            'dataset': str(Path(video_path).name.split('_')[0]),
            'total_duration': duration,
            'sampled_length': num_seconds
        }