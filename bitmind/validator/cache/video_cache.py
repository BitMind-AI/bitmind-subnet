import random
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import bittensor as bt
import ffmpeg
from PIL import Image

from .base_cache import BaseCache
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
        datasets: Optional[dict] = None,
        video_update_interval: int = 2,
        zip_update_interval: int = 24,
        num_videos_per_source: int = 10
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
            compressed_file_extension='.zip'
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

    def _extract_random_items(self) -> List[Path]:
        """
        Extract random videos from zip files in compressed directory.
        
        Returns:
            List of paths to extracted video files.
        """
        extracted_files = []
        zip_files = self._get_compressed_files()
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
        num_frames: int = 6,
        fps: Optional[float] = None,
        min_fps: Optional[float] = None,
        max_fps: Optional[float] = None
    ) -> Optional[Dict[str, Union[List[Image.Image], str, float]]]:
        """
        Sample random frames from a random video in the cache.

        Args:
            num_frames: Number of consecutive frames to sample
            fps: Fixed frames per second to sample. Mutually exclusive with min_fps/max_fps.
            min_fps: Minimum frames per second when auto-calculating fps. Must be used with max_fps.
            max_fps: Maximum frames per second when auto-calculating fps. Must be used with min_fps.

        Returns:
            Dictionary containing:
                - video: List of sampled video frames as PIL Images
                - path: Path to source video file
                - dataset: Name of source dataset
                - total_duration: Total video duration in seconds
                - sampled_length: Number of seconds sampled
            Returns None if no videos are available or extraction fails.
        """
        if fps is not None and (min_fps is not None or max_fps is not None):
            raise ValueError("Cannot specify both fps and min_fps/max_fps")
        if (min_fps is None) != (max_fps is None):
            raise ValueError("min_fps and max_fps must be specified together")

        video_files = self._get_cached_files()
        if not video_files:
            bt.logging.warning("No videos available in cache")
            return None

        video_path = random.choice(video_files)
        if not Path(video_path).exists():
            bt.logging.error(f"Selected video {video_path} not found")
            return None

        duration = get_video_duration(str(video_path))

        # Use fixed fps if provided, otherwise calculate from range
        frame_rate = fps
        if frame_rate is None:
            # For very short videos (< 1 second), use max_fps to capture detail
            if duration <= 1.0:
                frame_rate = max_fps
            else:
                # For longer videos, scale fps inversely with duration
                # This ensures we don't span too much of longer videos
                # while still capturing enough detail in shorter ones
                target_duration = min(2.0, duration * 0.2)  # Cap at 2 seconds or 20% of duration
                frame_rate = (num_frames - 1) / target_duration
                frame_rate = max(min_fps, min(frame_rate, max_fps))

        sample_duration = (num_frames - 1) / frame_rate
        start_time = random.uniform(0, max(0, duration - sample_duration))
        frames: List[Image.Image] = []

        bt.logging.info(f'Extracting {num_frames} frames at {frame_rate}fps starting at {start_time:.2f}s')

        for i in range(num_frames):
            timestamp = start_time + (i / frame_rate)
            
            try:
                # extract frames
                out_bytes, err = (
                    ffmpeg
                    .input(str(video_path), ss=str(timestamp))
                    .filter('select', 'eq(n,0)')
                    .output(
                        'pipe:',
                        vframes=1,
                        format='image2',
                        vcodec='png',
                        loglevel='error'  # silence ffmpeg output
                    )
                    .run(capture_stdout=True, capture_stderr=True)
                )

                if not out_bytes:
                    bt.logging.error(f'No data received for frame at {timestamp}s; Error: {err}')
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
 
        bt.logging.success(f"Sampled {len(frames)} frames at {frame_rate}fps")
        return {
            'video': frames,
            'fps': frame_rate,
            'num_frames': num_frames,
            'path': str(video_path),
            'dataset': str(Path(video_path).name.split('_')[0]),
            'total_duration': duration,
            'sampled_length': sample_duration
        }
