import zipfile
from pathlib import Path
from typing import List

from bitmind.types import CacheUpdaterConfig, CacheConfig
from bitmind.cache.updater import BaseUpdater
from bitmind.cache.datasets import DatasetRegistry


class VideoUpdater(BaseUpdater):
    """
    Updater for video data from zip files.

    This class handles downloading zip files from Hugging Face datasets
    and extracting videos from them into the media cache.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        updater_config: CacheUpdaterConfig,
        data_manager: DatasetRegistry,
    ):
        super().__init__(
            cache_config=cache_config,
            updater_config=updater_config,
            data_manager=data_manager,
        )

    @property
    def media_file_extensions(self) -> List[str]:
        """List of file extensions supported by this updater"""
        return [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"]

    @property
    def compressed_file_extension(self) -> str:
        """File extension for compressed source files"""
        return ".zip"

    async def _extract_items_from_source(
        self, source_path: Path, count: int
    ) -> List[Path]:
        """
        Extract videos from a zip file.

        Args:
            source_path: Path to the zip file
            count: Number of videos to extract

        Returns:
            List of paths to extracted video files
        """
        self.cache_fs._log_trace(f"Extracting up to {count} videos from {source_path}")

        dataset_name = source_path.parent.name
        if not dataset_name:
            dataset_name = source_path.stem

        dest_dir = self.cache_fs.cache_dir / dataset_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            from ..util import extract_videos_from_zip

            extracted_pairs = extract_videos_from_zip(
                zip_path=source_path,
                dest_dir=dest_dir,
                num_videos=count,
                file_extensions=set(self.media_file_extensions),
            )

            # extract_videos_from_zip returns pairs of (video_path, metadata_path)
            # We just need the video paths for our return value
            video_paths = [Path(pair[0]) for pair in extracted_pairs]

            self.cache_fs._log_trace(
                f"Extracted {len(video_paths)} videos from {source_path}"
            )
            return video_paths

        except Exception as e:
            self.cache_fs._log_trace(f"Error extracting videos from {source_path}: {e}")
            return []
