import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import bittensor as bt
from PIL import Image

from .base_cache import BaseCache
from .extract import extract_images_from_parquet
from .util import is_parquet_complete


class ImageCache(BaseCache):
    """
    A class to manage image caching from parquet files.
    
    This class handles the caching, updating, and sampling of images stored
    in parquet files. It maintains both a compressed cache of parquet files
    and an extracted cache of images ready for processing.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        datasets: Optional[dict] = None,
        parquet_update_interval: int = 24,
        image_update_interval: int = 2,
        num_images_per_source: int = 100,
    ) -> None:
        """        
        Args:
            cache_dir: Path to store extracted images
            parquet_update_interval: Hours between parquet cache updates
            image_update_interval: Hours between image cache updates
            num_images_per_source: Number of images to extract per parquet
        """
        super().__init__(
            cache_dir=cache_dir,
            datasets=datasets,
            extracted_update_interval=image_update_interval,
            compressed_update_interval=parquet_update_interval,
            num_samples_per_source=num_images_per_source,
            file_extensions=['.jpg', '.jpeg', '.png'],
            compressed_file_extension='.parquet'
        )  
                
    def _clear_incomplete_sources(self) -> None:
        """Remove any incomplete or corrupted parquet files."""
        for path in self._get_compressed_files():
            if path.suffix == '.parquet' and not is_parquet_complete(path):
                try:
                    path.unlink()
                    bt.logging.warning(f"Removed incomplete parquet file {path}")
                except Exception as e:
                    bt.logging.error(f"Error removing incomplete parquet {path}: {e}")
    
    def _extract_random_items(self) -> List[Path]:
        """
        Extract random videos from zip files in compressed directory.
        
        Returns:
            List of paths to extracted video files.
        """
        extracted_files = []
        parquet_files = self._get_compressed_files()
        if not parquet_files:
            bt.logging.warning(f"No parquet files found in {self.compressed_dir}")
            return extracted_files

        for parquet_file in parquet_files:
            try:
                extracted_files += extract_images_from_parquet(
                    parquet_file,
                    self.cache_dir,
                    self.num_samples_per_source
                )
            except Exception as e:
                bt.logging.error(f"Error processing parquet file {parquet_file}: {e}")
        return extracted_files

    def sample(self) -> Optional[Dict[str, Any]]:
        """
        Sample a random image and its metadata from the cache.

        Returns:
            Dictionary containing:
                - image: PIL Image
                - path: Path to source file
                - dataset: Source dataset name
                - metadata: Metadata dict
            Returns None if no valid image is available.
        """
        cached_files = self._get_cached_files()
        if not cached_files:
            bt.logging.warning("No images available in cache")
            return None

        attempts = 0
        max_attempts = len(cached_files) * 2

        while attempts < max_attempts:
            attempts += 1
            image_path = random.choice(cached_files)

            try:
                image = Image.open(image_path)
                metadata = json.loads(image_path.with_suffix('.json').read_text())
                return {
                    'image': image,
                    'path': str(image_path),
                    'dataset': metadata.get('dataset', None),
                    'index': metadata.get('index', None)
                }

            except Exception as e:
                bt.logging.warning(f"Failed to load image {image_path}: {e}")
                continue

        bt.logging.warning(f"Failed to find valid image after {attempts} attempts")
        return None
