from abc import ABC, abstractmethod
import asyncio
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import bittensor as bt

from .util import get_most_recent_update_time, seconds_to_str


class BaseCache(ABC):
    """
    Abstract base class for managing file caches with compressed sources.
    
    This class provides the basic infrastructure for maintaining both a compressed
    source cache and an extracted cache, with automatic refresh intervals and 
    background update tasks.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        datasets: dict,
        extracted_update_interval: int,
        compressed_update_interval: int,
        num_samples_per_source: int,
        file_extensions: List[str],
        run_updater: bool
    ) -> None:
        """
        Initialize the base cache infrastructure.
        
        Args:
            cache_dir: Path to store extracted files
            extracted_update_interval: Hours between extracted cache updates
            compressed_update_interval: Hours between compressed cache updates
            num_samples_per_source: Number of items to extract per source
            file_extensions: List of valid file extensions for this cache type
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.compressed_dir = self.cache_dir / 'sources'
        self.compressed_dir.mkdir(exist_ok=True, parents=True)

        self.datasets = datasets
        
        self.extracted_update_interval = extracted_update_interval * 60 * 60
        self.compressed_update_interval = compressed_update_interval * 60 * 60
        self.num_samples_per_source = num_samples_per_source
        self.file_extensions = file_extensions

        if run_updater:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = asyncio.get_event_loop()

            # Initialize caches, blocking to ensure data are available for validator
            bt.logging.info(f"Setting up cache at {self.cache_dir}")
            bt.logging.info(f"Clearing incomplete sources in {self.compressed_dir}")
            self._clear_incomplete_sources()

            if self._compressed_cache_empty():
                bt.logging.info(f"Compressed cache {self.compressed_dir} empty; populating")
                # grab 1 zip per source to get started, download more later
                self._refresh_compressed_cache(n_zips_per_source=1)

            if self._extracted_cache_empty():
                bt.logging.info(f"Extracted cache {self.cache_dir} empty; populating")
                self._refresh_extracted_cache()

            # Start background tasks
            bt.logging.info(f"Starting background tasks")
            self._compressed_updater_task = self.loop.create_task(
                self._run_compressed_updater()
            )
            self._extracted_updater_task = self.loop.create_task(
                self._run_extracted_updater()
            )

    def _get_cached_files(self) -> List[Path]:
        """Get list of all extracted files in cache directory."""
        return [
            f for f in self.cache_dir.iterdir() 
            if f.suffix.lower() in self.file_extensions
        ]

    def _get_compressed_files(self) -> List[Path]:
        """Get list of all compressed files in compressed directory."""
        return list(self.compressed_dir.iterdir())

    def _extracted_cache_empty(self) -> bool:
        """Check if extracted cache directory is empty."""
        return len(self._get_cached_files()) == 0

    def _compressed_cache_empty(self) -> bool:
        """Check if compressed cache directory is empty."""
        return len(self._get_compressed_files()) == 0

    async def _run_extracted_updater(self) -> None:
        """Asynchronously refresh extracted files according to update interval."""
        while True:
            try:
                last_update = get_most_recent_update_time(self.cache_dir)
                time_elapsed = time.time() - last_update

                if time_elapsed >= self.extracted_update_interval:
                    bt.logging.info("Running extracted cache refresh...")
                    self._refresh_extracted_cache()
                    bt.logging.info("Extracted cache refresh complete.")

                sleep_time = max(0, self.extracted_update_interval - time_elapsed)
                bt.logging.info(f"Sleeping for {seconds_to_str(sleep_time)}")
                await asyncio.sleep(sleep_time)
            except Exception as e:
                bt.logging.error(f"Error in extracted cache update: {e}")
                await asyncio.sleep(60)

    async def _run_compressed_updater(self) -> None:
        """Asynchronously refresh compressed files according to update interval."""
        while True:
            try:
                self._clear_incomplete_sources()
                last_update = get_most_recent_update_time(self.compressed_dir)
                time_elapsed = time.time() - last_update

                if time_elapsed >= self.compressed_update_interval:
                    bt.logging.info("Running compressed cache refresh...")
                    self._refresh_compressed_cache(n_zips_per_source=1)
                    bt.logging.info("Compressed cache refresh complete.")

                sleep_time = max(0, self.compressed_update_interval - time_elapsed)
                bt.logging.info(f"Sleeping for {seconds_to_str(sleep_time)}")
                await asyncio.sleep(sleep_time)
            except Exception as e:
                bt.logging.error(f"Error in compressed cache update: {e}")
                await asyncio.sleep(60)

    @abstractmethod
    def _clear_incomplete_sources(self) -> None:
        """Remove any incomplete or corrupted source files from cache."""
        pass

    @abstractmethod
    def _refresh_compressed_cache(self) -> None:
        """Refresh the compressed file cache with new downloads."""
        pass

    @abstractmethod
    def _refresh_extracted_cache(self) -> None:
        """Refresh the extracted cache with new selections."""
        pass

    @abstractmethod
    def sample(self, num_samples: int) -> Optional[Dict[str, Any]]:
        """Sample random items from the cache."""
        pass

    def __del__(self) -> None:
        """Cleanup background tasks on deletion."""
        if hasattr(self, '_extracted_updater_task'):
            self._extracted_updater_task.cancel()
        if hasattr(self, '_compressed_updater_task'):
            self._compressed_updater_task.cancel()