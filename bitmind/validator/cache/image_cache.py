import random
from typing import Dict, List, Optional, Union

import bittensor as bt
from PIL import Image

from .base_cache import BaseCache
from .image_utils import (
    is_parquet_complete,
    extract_images_from_parquet,
    download_parquet_file
)

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
        compressed_dir: Union[str, Path],
        parquet_update_interval: int = 24,
        image_update_interval: int = 2,
        num_images_per_source: int = 100,
        metadata_columns: Optional[List[str]] = None,
        image_size: Optional[Tuple[int, int]] = None
    ) -> None:
        """        
        Args:
            cache_dir: Path to store extracted images
            compressed_dir: Path to store parquet files
            parquet_update_interval: Hours between parquet cache updates
            image_update_interval: Hours between image cache updates
            num_images_per_source: Number of images to extract per parquet
            metadata_columns: Columns to preserve from parquet metadata
            image_size: Optional tuple of (width, height) to resize images
        """
        super().__init__(
            cache_dir=cache_dir,
            compressed_dir=compressed_dir,
            extracted_update_interval=image_update_interval,
            compressed_update_interval=parquet_update_interval,
            num_samples_per_source=num_images_per_source,
            file_extensions=['.jpg', '.jpeg', '.png']
        )
        
        self.metadata_columns = metadata_columns or []
        self.image_size = image_size
        self.metadata_file = self.cache_dir / 'metadata.json'
        
        # Load existing metadata
        self.metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
                
    def _clear_incomplete_sources(self) -> None:
        """Remove any incomplete or corrupted parquet files."""
        for path in self._get_compressed_files():
            if path.suffix == '.parquet' and not is_parquet_complete(path):
                try:
                    path.unlink()
                    bt.logging.warning(f"Removed incomplete parquet file {path}")
                except Exception as e:
                    bt.logging.error(f"Error removing incomplete parquet {path}: {e}")

    def _refresh_compressed_cache(self) -> None:
        """Download new parquet files from configured sources."""
        try:
            prior_files = list(self.compressed_dir.glob('*.parquet'))
            
            new_files = []
            for source in PARQUET_DATASET_META:  # You'll need to define this
                file_path = download_parquet_file(
                    source['url'],
                    self.compressed_dir
                )
                if file_path:
                    new_files.append(file_path)
            
            if new_files:
                bt.logging.info(f"{len(new_files)} new parquet files added")
                bt.logging.info(f"Removing {len(prior_files)} previous files")
                for file in prior_files:
                    try:
                        file.unlink()
                    except Exception as e:
                        bt.logging.error(f"Error removing file {file}: {e}")
            else:
                bt.logging.error("No new parquet files were added")
                
        except Exception as e:
            bt.logging.error(f"Error during parquet refresh: {e}")
            raise

    def _refresh_extracted_cache(self) -> None:
        """Extract new random images from parquet files."""
        try:
            # Remove existing images and metadata
            prior_images = self._get_cached_files()
            for file in prior_images:
                try:
                    file.unlink()
                    if str(file) in self.metadata:
                        del self.metadata[str(file)]
                except Exception as e:
                    bt.logging.error(f"Error removing file {file}: {e}")
            
            # Extract new images
            parquet_files = list(self.compressed_dir.glob('*.parquet'))
            if not parquet_files:
                bt.logging.warning("No parquet files available")
                return
                
            for parquet_file in parquet_files:
                try:
                    images_and_metadata = extract_images_from_parquet(
                        parquet_file,
                        self.num_samples_per_source,
                        self.metadata_columns