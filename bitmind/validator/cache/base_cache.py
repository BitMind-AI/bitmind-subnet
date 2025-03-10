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
        datasets: dict = None,
        extracted_update_interval: int = 4,
        compressed_update_interval: int = 12,
        num_sources_per_dataset: int = 1,
        max_compressed_size_gb: float = 100.0,
        max_extracted_size_gb: float = 10.0,
    ) -> None:
        """
        Initialize the base cache infrastructure.
        
        Args:
            cache_dir: Path to store extracted files
            extracted_update_interval: Hours between extracted cache updates
            compressed_update_interval: Hours between compressed cache updates
            file_extensions: List of valid file extensions for this cache type
            max_compressed_size_gb: Maximum size in GB for compressed cache directory
            max_extracted_size_gb: Maximum size in GB for extracted cache directory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.compressed_dir = self.cache_dir / 'sources'
        self.compressed_dir.mkdir(exist_ok=True, parents=True)

        self.datasets = datasets

        self.extracted_update_interval = extracted_update_interval * 60 * 60
        self.compressed_update_interval = compressed_update_interval * 60 * 60
        self.num_sources_per_dataset = num_sources_per_dataset
        self.file_extensions = file_extensions
        self.compressed_file_extension = compressed_file_extension
        self.max_compressed_size_bytes = max_compressed_size_gb * 1024 * 1024 * 1024
        self.max_extracted_size_bytes = max_extracted_size_gb * 1024 * 1024 * 1024

    def start_updater(self):
        """Start the background updater tasks for compressed and extracted caches."""
        if not self.datasets:
            bt.logging.error("No datasets configured. Cannot start cache updater.")
            return

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.get_event_loop()

        # Initialize caches, blocking to ensure data are available for validator
        bt.logging.info(f"Setting up cache at {self.cache_dir}")
        bt.logging.info(f"Clearing incomplete sources in {self.compressed_dir}")
        self._clear_incomplete_sources()

        if self._extracted_cache_empty():
            if self._compressed_cache_empty():
                bt.logging.info(f"Compressed cache {self.compressed_dir} empty; populating")
                # grab 1 zip first to ensure validator has available data
                for batch_size in [1, None]:
                    self._refresh_compressed_cache(n_sources_per_dataset=1, n_datasets=batch_size)
                    self._refresh_extracted_cache()
            else:
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

    def _get_cached_files(self, max_depth: int = 3) -> List[Path]:
        """Get list of all extracted files in cache directory up to a maximum depth."""
        files = []
        for depth in range(max_depth + 1):
            pattern = '*/' * depth + '*'
            files.extend([
                f for f in self.cache_dir.glob(pattern)
                if f.is_file() and f.suffix.lower() in self.file_extensions
            ])
        return files

    def _get_compressed_files(self) -> List[Path]:
        """Get list of all compressed files in compressed directory. Handles up two two levels
        of directory nesting """
        return list(self.compressed_dir.glob(f'*{self.compressed_file_extension}')) + \
                list(self.compressed_dir.glob(f'*/*{self.compressed_file_extension}')) + \
                list(self.compressed_dir.glob(f'*/*/*{self.compressed_file_extension}'))

    def _extracted_cache_empty(self) -> bool:
        """Check if extracted cache directory is empty."""
        return len(self._get_cached_files()) == 0

    def _compressed_cache_empty(self) -> bool:
        """Check if compressed cache directory is empty."""
        return len(self._get_compressed_files()) == 0

    def prune_cache(self, cache="compressed") -> None:
        """Check extracted cache size and remove oldest files if over limit."""
        if cache == 'compressed':
            cache_dir = self.compressed_dir
            files = self._get_compressed_files()
            max_size = self.max_compressed_size_bytes
        elif cache == 'extracted':
            cache_dir = self.cache_dir
            files = self._get_cached_files()
            max_size = self.max_extracted_size_bytes


        total_size = sum(f.stat().st_size for f in files if f.exists())
        total_size_gb = total_size / (1024*1024*1024)
        bt.logging.info(f"[{cache_dir}] Cache size: {len(files)} files | {total_size_gb:.6f} GB")
        if total_size <= max_size:
            return

        n_removed = 0
        bytes_removed = 0
        remaining_size = total_size
        sorted_files = sorted(files, key=lambda f: f.stat().st_mtime if f.exists() else float('inf'))

        bt.logging.info(f"[{cache_dir}] Pruning...")        
        for file in sorted_files:
            if remaining_size <= max_size:
                break
            try:
                if file.exists():
                    file_size = file.stat().st_size
                    file.unlink()
                    json_file = file.with_suffix('.json')
                    if json_file.exists():
                        json_file.unlink()

                    n_removed += 1
                    bytes_removed += file_size
                    remaining_size -= file_size
            except Exception as e:
                continue
            
        final_size = total_size - bytes_removed
        cache_gb = f"{final_size / (1024*1024*1024):.6f}".rstrip('0')
        removed_gb = f"{bytes_removed / (1024*1024*1024):.6f}".rstrip('0')
        bt.logging.info(f"[{cache_dir}] Removed {n_removed} ({removed_gb} GB) files. New cache size is {cache_gb} GB")

    async def _run_extracted_updater(self) -> None:
        """Asynchronously refresh extracted files according to update interval."""
        while True:
            try:
                self.prune_cache('extracted')
                last_update = get_most_recent_update_time(self.cache_dir)
                time_elapsed = time.time() - last_update

                if time_elapsed >= self.extracted_update_interval:
                    bt.logging.info(f"[{self.cache_dir}] Refreshing cache")
                    self._refresh_extracted_cache()
                    bt.logging.info(f"[{self.cache_dir}] Cache refresh complete ")

                sleep_time = max(0, self.extracted_update_interval - time_elapsed)
                bt.logging.info(f"[{self.cache_dir}] Next media cache refresh in {seconds_to_str(sleep_time)}")
                await asyncio.sleep(sleep_time)
            except Exception as e:
                bt.logging.error(f"[{self.cache_dir}] Error in extracted cache update: {e}")
                await asyncio.sleep(60)

    async def _run_compressed_updater(self) -> None:
        """Asynchronously refresh compressed files according to update interval."""
        while True:
            try:
                self._clear_incomplete_sources()
                self.prune_cache('compressed')
                last_update = get_most_recent_update_time(self.compressed_dir)
                time_elapsed = time.time() - last_update

                if time_elapsed >= self.compressed_update_interval:
                    cache_state_before = self._get_compressed_files()
                    bt.logging.info(f"[{self.compressed_dir}] Refreshing cache")
                    self._refresh_compressed_cache()
                    bt.logging.info(f"[{self.compressed_dir}] Cache refresh complete")
                    if set(cache_state_before) == set(self._get_compressed_files()):
                        bt.logging.warning(f"[{self.compressed_dir}] All datasets small enough to store locally. Stopping updater.")
                        return

                sleep_time = max(0, self.compressed_update_interval - time_elapsed)
                bt.logging.info(f"[{self.compressed_dir}] Next compressed cache refresh in {seconds_to_str(sleep_time)}")
                await asyncio.sleep(sleep_time)
            except Exception as e:
                bt.logging.error(f"[{self.compressed_dir}] Error in compressed cache update: {e}")
                await asyncio.sleep(60)

    def _refresh_compressed_cache(
        self,
        n_sources_per_dataset: Optional[int] = None,
        n_datasets: Optional[int] = None
    ) -> None:
        """
        Refresh the compressed file cache with new downloads.
        """
        if n_sources_per_dataset is None:
            n_sources_per_dataset = self.num_sources_per_dataset

        try:
            bt.logging.info(f"{len(self._get_compressed_files())} compressed sources currently cached")

            new_files: List[Path] = []
            for dataset in self.datasets[:n_datasets]:
                filenames = list_hf_files(
                    repo_id=dataset['path'], 
                    extension=self.compressed_file_extension)
                remote_paths = [
                    f"https://huggingface.co/datasets/{dataset['path']}/resolve/main/{f}"
                    for f in filenames
                ]
                bt.logging.info(f"Downloading {n_sources_per_dataset} from {dataset['path']} to {self.compressed_dir}")
                new_files += download_files(
                    urls=np.random.choice(remote_paths, n_sources_per_dataset),
                    output_dir=self.compressed_dir / dataset['path'].split('/')[1])

            if new_files:
                bt.logging.info(f"{len(new_files)} new files added to {self.compressed_dir}")
            else:
                bt.logging.error(f"No new files were added to {self.compressed_dir}")

        except Exception as e:
            bt.logging.error(f"Error during compressed refresh for {self.compressed_dir}: {e}")
            raise

    def _refresh_extracted_cache(self, n_items_per_source: Optional[int] = None) -> None:
        """Refresh the extracted cache with new selections."""
        bt.logging.info(f"{len(self._get_cached_files())} media files currently cached")
        new_files = self._extract_random_items(n_items_per_source)
        if new_files:
            bt.logging.info(f"{len(new_files)} new files added to {self.cache_dir}")
        else:
            bt.logging.error(f"No new files were added to {self.cache_dir}")

    @abstractmethod
    def _extract_random_items(self, n_items_per_source: Optional[int] = None) -> List[Path]:
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
