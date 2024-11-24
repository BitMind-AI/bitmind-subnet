from datetime import datetime
from io import BytesIO
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import bittensor as bt
import ffmpeg
from PIL import Image

from .base_cache import BaseCache
from bitmind.validator.video_utils import (
    search_and_download_youtube_videos,
    get_video_duration,
    is_zip_complete,
    clip_video,
    download_zips,
)
from bitmind.validator.config import VIDEO_DATASET_META


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
        compressed_dir: Union[str, Path],
        video_update_interval: int = 2,
        zip_update_interval: int = 24,
        num_videos_per_source: int = 10,
        use_youtube: bool = False
    ) -> None:
        """
        Initialize the VideoCache.
        
        Args:
            cache_dir: Path to store extracted video files
            compressed_dir: Path to store compressed source files
            video_update_interval: Hours between video cache updates
            zip_update_interval: Hours between zip cache updates
            num_videos_per_source: Number of videos to extract per source
            use_youtube: Whether to include YouTube videos
        """
        super().__init__(
            cache_dir=cache_dir,
            compressed_dir=compressed_dir,
            extracted_update_interval=video_update_interval,
            compressed_update_interval=zip_update_interval,
            num_samples_per_source=num_videos_per_source,
            file_extensions=['.mp4', '.avi', '.mov', '.mkv']
        )
        self.use_youtube = use_youtube

    def _clear_incomplete_sources(self) -> None:
        """Remove any incomplete or corrupted zip files from cache."""
        for path in self._get_compressed_files():
            if path.suffix == '.zip' and not is_zip_complete(path):
                try:
                    path.unlink()
                    bt.logging.warning(f"Removed incomplete zip file {path}")
                except Exception as e:
                    bt.logging.error(f"Error removing incomplete zip {path}: {e}")

    def _refresh_compressed_cache(self) -> None:
        """
        Refresh the compressed file cache with new downloads.
        
        Downloads new zip files from configured sources and removes old ones.
        Optionally includes YouTube videos if enabled.
        """
        try:
            prior_files = list(self.compressed_dir.glob('*.zip'))
            
            new_files: List[Path] = []
            for meta in VIDEO_DATASET_META['real']:
                new_files += download_zips(
                    meta["base_zip_url"],
                    self.compressed_dir,
                    meta["max_zip_id"],
                    meta.get("min_zip_id", 0),
                    num_zips=1,
                    download_all=False,
                    err_handler_fn=meta.get("err_handler", None)
                )
                
            if self.use_youtube:
                youtube_files = search_and_download_youtube_videos(
                    self.compressed_dir,
                    num_videos=self.num_samples_per_source
                )
                new_files.extend(youtube_files)

            if new_files:
                bt.logging.info(f"{len(new_files)} new files added to cache")
                bt.logging.info(f"Removing {len(prior_files)} previous files")
                for file in prior_files:
                    try:
                        file.unlink()
                    except Exception as e:
                        bt.logging.error(f"Error removing file {file}: {e}")
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
        try:
            prior_cache_files = self._get_cached_files()
            new_cache_files = self._extract_random_videos()

            if new_cache_files:
                bt.logging.info(f"{len(new_cache_files)} new videos added to cache")
                bt.logging.info(f"Removing {len(prior_cache_files)} previous videos")
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

    def _extract_random_videos(self) -> List[Path]:
        """
        Extract random videos from zip files in compressed directory.
        
        Returns:
            List of paths to extracted video files.
        """
        extracted_files: List[Path] = []
        zip_files = list(self.compressed_dir.glob('*.zip'))
        if not zip_files:
            bt.logging.warning(f"No zip files found in {self.compressed_dir}")
            return extracted_files

        zip_path = random.choice(zip_files)
        try:
            with ZipFile(zip_path) as zip_file:
                video_files = [
                    f for f in zip_file.namelist()
                    if any(f.lower().endswith(ext) for ext in self.file_extensions)
                ]

                if not video_files:
                    bt.logging.warning(f"No video files found in {zip_path}")
                    return extracted_files

                selected_videos = random.sample(
                    video_files,
                    min(self.num_samples_per_source, len(video_files))
                )

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

    def sample_random_items(
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