import json
import os
import math
import random
import tempfile

from pathlib import Path
from typing import Dict, List, Any, Optional
from io import BytesIO

import ffmpeg
import numpy as np
from PIL import Image

from bitmind.cache.sampler.base import BaseSampler
from bitmind.cache.cache_fs import CacheConfig
from bitmind.cache.util.video import get_video_metadata


class VideoSampler(BaseSampler):
    """
    Sampler for cached video data.

    This class provides access to videos in the media cache,
    allowing sampling of video segments as binary data.
    """

    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config)

    @property
    def media_file_extensions(self) -> List[str]:
        """List of file extensions supported by this sampler"""
        return [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"]

    async def sample(
        self,
        count: int = 1,
        remove_from_cache: bool = False,
        min_duration: float = 1.0,
        max_duration: float = 6.0,
        max_frames: int = 144,
    ) -> Dict[str, Any]:
        """
        Sample random video segments from the cache.

        Args:
            count: Number of videos to sample
            remove_from_cache: Whether to remove sampled videos from cache

        Returns:
            Dictionary containing:
                - count: Number of videos successfully sampled
                - items: List of dictionaries containing video binary data and metadata
        """
        cached_files = self.cache_fs.get_files(
            cache_type="media",
            file_extensions=self.media_file_extensions,
            group_by_source=True,
        )

        if not cached_files:
            self.cache_fs._log_warning("No videos available in cache")
            return {"count": 0, "items": []}

        sampled_items = []
        for _ in range(count):
            if not cached_files:
                break

            video_result = await self._sample_frames(
                files=cached_files,
                min_duration=min_duration,
                max_duration=max_duration,
                max_frames=max_frames,
                remove_from_cache=remove_from_cache,
            )

            if video_result:
                sampled_items.append(video_result)

        return {"count": len(sampled_items), "items": sampled_items}

    async def _sample_frames(
        self,
        files,
        min_duration: float = 1.0,
        max_duration: float = 6.0,
        max_fps: float = 30.0,
        max_frames: int = 144,
        remove_from_cache: bool = False,
        as_float32: bool = False,
        channels_first: bool = False,
        as_rgb: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Sample a random video segment and return it as a numpy array.

        Args:
            files: Dict mapping source names to lists of video file paths
            min_duration: Minimum duration of video segment to extract in seconds
            max_duration: Maximum duration of video segment to extract in seconds
            max_fps: Maximum frame rate to use when sampling frames
            max_frames: Maximum number of frames to extract
            remove_from_cache: Whether to remove the source video from cache
            as_float32: Whether to return frames as float32 (0-1) instead of uint8 (0-255)
            channels_first: Whether to return frames with channels first (TCHW) instead of channels last (THWC)
            as_rgb: Whether to return frames in RGB format (True) or BGR format (False)

        Returns:
            Dictionary containing:
                - frames: Video frames as numpy array with shape (T,H,W,C)
                - metadata: Video metadata
                - source: Source information
                - segment: Information about the extracted segment
            Or None if sampling fails
        """
        for _ in range(5):
            if not files:
                self.cache_fs._log_warning("No more videos available to try")
                return None

            source = random.choice(list(files.keys()))
            if not files[source]:
                del files[source]
                continue

            video_path = random.choice(files[source])

            try:
                if not video_path.exists():
                    files[source].remove(video_path)
                    continue

                try:
                    video_info = get_video_metadata(str(video_path))
                    total_duration = video_info.get("duration", 0)
                    width = int(video_info.get("width", 256))
                    height = int(video_info.get("height", 256))
                    reported_fps = float(video_info.get("fps", max_fps))
                except Exception as e:
                    self.cache_fs._log_error(
                        f"Unable to extract video metadata from {str(video_path)}: {e}"
                    )
                    files[source].remove(video_path)
                    continue

                if (
                    reported_fps > max_fps
                    or reported_fps <= 0
                    or not math.isfinite(reported_fps)
                ):
                    self.cache_fs._log_warning(
                        f"Unreasonable FPS ({reported_fps}) detected in {video_path}, capping at {max_fps}"
                    )
                    frame_rate = max_fps
                else:
                    frame_rate = reported_fps

                target_duration = random.uniform(min_duration, max_duration)
                target_duration = min(target_duration, total_duration)

                num_frames = int(target_duration * frame_rate) + 1
                num_frames = min(num_frames, max_frames)

                actual_duration = (num_frames - 1) / frame_rate

                max_start = max(0, total_duration - actual_duration)
                start_time = random.uniform(0, max_start)

                frames = []
                no_data = []

                for i in range(num_frames):
                    timestamp = start_time + (i / frame_rate)

                    try:
                        out_bytes, err = (
                            ffmpeg.input(str(video_path), ss=str(timestamp))
                            .filter("select", "eq(n,0)")
                            .output(
                                "pipe:",
                                vframes=1,
                                format="image2",
                                vcodec="png",
                                loglevel="error",
                            )
                            .run(capture_stdout=True, capture_stderr=True)
                        )

                        if not out_bytes:
                            no_data.append(timestamp)
                            continue

                        try:
                            frame = Image.open(BytesIO(out_bytes))
                            frame.load()  # Verify image can be loaded
                            frames.append(np.array(frame))
                        except Exception as e:
                            self.cache_fs._log_error(
                                f"Failed to process frame at {timestamp}s: {e}"
                            )
                            continue

                    except ffmpeg.Error as e:
                        self.cache_fs._log_error(
                            f"FFmpeg error at {timestamp}s: {e.stderr.decode()}"
                        )
                        continue

                if len(no_data) > 0:
                    tmin, tmax = min(no_data), max(no_data)
                    self.cache_fs._log_warning(
                        f"No data received for {len(no_data)} frames between {tmin} and {tmax}"
                    )

                if not frames:
                    self.cache_fs._log_warning(
                        f"No frames successfully extracted from {video_path}"
                    )
                    files[source].remove(video_path)
                    continue

                frames = np.stack(frames, axis=0)

                if as_float32:
                    frames = frames.astype(np.float32) / 255.0

                if not as_rgb:
                    frames = frames[:, :, :, [2, 1, 0]]  # RGB to BGR

                if channels_first:
                    frames = np.transpose(frames, (0, 3, 1, 2))

                metadata = {}
                metadata_path = video_path.with_suffix(".json")
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                    except Exception as e:
                        self.cache_fs._log_warning(
                            f"Error loading metadata for {video_path}: {e}"
                        )

                result = {
                    "video": frames,
                    "path": str(video_path),
                    "metadata_path": str(metadata_path),
                    "metadata": metadata,
                    "segment": {
                        "start_time": start_time,
                        "duration": actual_duration,
                        "fps": frame_rate,
                        "width": width,
                        "height": height,
                        "num_frames": len(frames),
                    },
                }

                if remove_from_cache:
                    try:
                        video_path.unlink(missing_ok=True)
                        metadata_path.unlink(missing_ok=True)
                        files[source].remove(video_path)
                    except Exception as e:
                        self.cache_fs._log_warning(
                            f"Failed to remove {video_path}: {e}"
                        )

                self.cache_fs._log_debug(
                    f"Successfully sampled {actual_duration}s segment ({len(frames)} frames)"
                )
                return result

            except Exception as e:
                self.cache_fs._log_error(f"Error sampling from {video_path}: {e}")
                files[source].remove(video_path)

        self.cache_fs._log_error("Failed to sample any video after multiple attempts")
        return None
