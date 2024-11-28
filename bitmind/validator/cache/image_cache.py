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
        run_updater: bool = False,
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
            compressed_file_extension='.parquet',
            run_updater=run_updater
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

    def sample(
        self,
        k: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Sample random images and their metadata from the cache.

        Args:
            k: Number of images to sample

        Returns:
            Dictionary containing:
                - images: List of sampled PIL Images
                - paths: List of paths to source files
                - datasets: List of source dataset names
                - metadata: List of metadata dicts if return_metadata is True
            Returns None if no valid images are available.
        """
        cached_files = self._get_cached_files()
        if not cached_files:
            bt.logging.warning("No images available in cache")
            return None

        valid_samples: List[Dict[str, Any]] = []
        attempts = 0
        max_attempts = len(cached_files) * 2
        while len(valid_samples) < k and attempts < max_attempts:
            attempts += 1
            image_path = random.choice(cached_files)

            try:
                image = Image.open(image_path)
                metadata = data = json.loads(image_path.with_suffix('.json').read_text())
                data = {
                    'image': image,
                    'path': str(image_path),
                    'dataset': metadata.get('dataset', None),
                    'index': metadata.get('index', None)
                }
                valid_samples.append(data)

            except Exception as e:
                bt.logging.warning(f"Failed to load image {image_path}: {e}")
                continue

        if not valid_samples:
            bt.logging.warning(
                f"Failed to find {k} valid images after {attempts} attempts"
            )
            return None

        return valid_samples
