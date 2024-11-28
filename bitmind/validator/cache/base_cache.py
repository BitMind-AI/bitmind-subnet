from abc import ABC, abstractmethod
import asyncio
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import bittensor as bt
import huggingface_hub as hf_hub
import numpy as np

from .util import get_most_recent_update_time, seconds_to_str
from .download import download_files, list_hf_files


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
        file_extensions: List[str],
        compressed_file_extension: str,
        run_updater: bool = False,
        datasets: dict = None,
        extracted_update_interval: int = 4,
        compressed_update_interval: int = 24,
        num_samples_per_source: int = 10,
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
        self.compressed_file_extension = compressed_file_extension

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
                self._refresh_compressed_cache(n_per_source=1)

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
        return list(self.compressed_dir.glob(f'*{self.compressed_file_extension}'))

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
                    bt.logging.info(f"Refreshing cache [{self.cache_dir}]")
                    self._refresh_extracted_cache()
                    bt.logging.info(f"Cache refresh complete [{self.cache_dir}]")

                sleep_time = max(0, self.extracted_update_interval - time_elapsed)
                bt.logging.info(f"Next cache refresh in {seconds_to_str(sleep_time)} [{self.compressed_dir}]")
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
                    bt.logging.info(f"Refreshing cache [{self.compressed_dir}]")
                    self._refresh_compressed_cache(n_per_source=1)
                    bt.logging.info(f"Cache refresh complete [{self.cache_dir}]")

                sleep_time = max(0, self.compressed_update_interval - time_elapsed)
                bt.logging.info(f"Next cache refresh in {seconds_to_str(sleep_time)} [{self.compressed_dir}]")
                await asyncio.sleep(sleep_time)
            except Exception as e:
                bt.logging.error(f"Error in compressed cache update: {e}")
                await asyncio.sleep(60)

    def _refresh_compressed_cache(self, n_per_source) -> None:
        """
        Refresh the compressed file cache with new downloads.
        """
        try:
            bt.logging.info(f"{len(self._get_compressed_files())} compressed sources currently cached")

            new_files: List[Path] = []
            for source in self.datasets:
                filenames = list_hf_files(
                    repo_id=source['path'], 
                    extension=self.compressed_file_extension)
                remote_paths = [
                    f"https://huggingface.co/datasets/{source['path']}/resolve/main/{f}"
                    for f in filenames
                ]
                bt.logging.info(f"Downloading {n_per_source} from {source['path']} to {self.compressed_dir}")
                new_files += download_files(
                    urls=np.random.choice(remote_paths, n_per_source),
                    output_dir=self.compressed_dir)

            if new_files:
                bt.logging.info(f"{len(new_files)} new files added to {self.compressed_dir}")
            else:
                bt.logging.error(f"No new files were added to {self.compressed_dir}")

        except Exception as e:
            bt.logging.error(f"Error during compressed refresh for {self.compressed_dir}: {e}")
            raise

    def _refresh_extracted_cache(self) -> None:
        """Refresh the extracted cache with new selections."""
        bt.logging.info(f"{len(self._get_compressed_files())} files currently cached")
        new_files = self._extract_random_items()
        if new_files:
            bt.logging.info(f"{len(new_files)} new files added to {self.cache_dir}")
        else:
            bt.logging.error(f"No new files were added to {self.cache_dir}")

    @abstractmethod
    def _extract_random_items(self) -> List[Path]:
        """Remove any incomplete or corrupted source files from cache."""
        pass

    @abstractmethod
    def _clear_incomplete_sources(self) -> None:
        """Remove any incomplete or corrupted source files from cache."""
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
