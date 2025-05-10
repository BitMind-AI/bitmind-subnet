from pathlib import Path
from typing import List
import traceback

from bitmind.cache.updater import BaseUpdater
from bitmind.cache.datasets import DatasetRegistry
from bitmind.cache.util.filesystem import is_parquet_complete
from bitmind.types import CacheUpdaterConfig, CacheConfig


class ImageUpdater(BaseUpdater):
    """
    Updater for image data from parquet files.

    This class handles downloading parquet files from Hugging Face datasets
    and extracting images from them into the media cache.
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
        return [".jpg", ".jpeg", ".png", ".webp"]

    @property
    def compressed_file_extension(self) -> str:
        """File extension for compressed source files"""
        return ".parquet"

    async def _extract_items_from_source(
        self, source_path: Path, count: int
    ) -> List[Path]:
        """
        Extract images from a parquet file.

        Args:
            source_path: Path to the parquet file
            count: Number of images to extract

        Returns:
            List of paths to extracted image files
        """
        self.cache_fs._log_trace(f"Extracting up to {count} images from {source_path}")

        dataset_name = source_path.parent.name
        if not dataset_name:
            dataset_name = source_path.stem

        dest_dir = self.cache_fs.cache_dir / dataset_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            from ..util import extract_images_from_parquet

            saved_files = extract_images_from_parquet(
                parquet_path=source_path, dest_dir=dest_dir, num_images=count
            )

            self.cache_fs._log_trace(
                f"Extracted {len(saved_files)} images from {source_path}"
            )
            return [Path(f) for f in saved_files]

        except Exception as e:
            self.cache_fs._log_error(f"Error extracting images from {source_path}: {e}")
            self.cache_fs._log_error(traceback.format_exc())
            return []
