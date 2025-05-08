from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from bitmind.cache.cache_fs import CacheFS
from bitmind.cache.datasets import DatasetRegistry
from bitmind.types import CacheUpdaterConfig, CacheConfig, CacheType
from bitmind.cache.util.download import list_hf_files, download_files
from bitmind.cache.util.filesystem import (
    filter_ready_files,
    wait_for_downloads_to_complete,
    is_source_complete,
)


class BaseUpdater(ABC):
    """
    Base class for cache updaters that handle downloading and extracting data.

    This version is designed to work with block callbacks rather than having
    its own internal timing logic.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        updater_config: CacheUpdaterConfig,
        data_manager: DatasetRegistry,
    ):
        self.cache_fs = CacheFS(cache_config)
        self.updater_config = updater_config
        self.dataset_registry = data_manager
        self._datasets = self._get_filtered_datasets()
        self._recently_downloaded_files = []

    def _get_filtered_datasets(
        self,
        modality: Optional[str] = None,
        media_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
    ) -> List[Any]:
        """Get datasets that match the cache configuration"""
        modality = self.cache_fs.config.modality if modality is None else modality
        media_type = (
            self.cache_fs.config.media_type if media_type is None else media_type
        )
        tags = self.cache_fs.config.tags if tags is None else tags

        return self.dataset_registry.get_datasets(
            modality=self.cache_fs.config.modality,
            media_type=self.cache_fs.config.media_type,
            tags=self.cache_fs.config.tags,
            exclude_tags=exclude_tags,
        )

    @property
    @abstractmethod
    def media_file_extensions(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def compressed_file_extension(self) -> str:
        pass

    @abstractmethod
    async def _extract_items_from_source(
        self, source_path: Path, count: int
    ) -> List[Path]:
        pass

    async def initialize_cache(self) -> None:
        """
        This performs a one-time initialization to ensure the cache has
        content available, particularly useful during first startup.
        """
        self.cache_fs._log_debug("Setting up cache")

        if self.cache_fs.is_empty(CacheType.MEDIA):
            if self.cache_fs.is_empty(CacheType.COMPRESSED):
                self.cache_fs._log_debug("Compressed cache empty; populating")
                await self.update_compressed_cache(
                    n_sources_per_dataset=1,
                    n_datasets=1,
                    exclude_tags=["large-zips"],
                    maybe_prune=False,
                )

                self.cache_fs._log_debug(
                    "Waiting for compressed files to finish downloading..."
                )
                await wait_for_downloads_to_complete(
                    self._recently_downloaded_files,
                )
                self._recently_downloaded_files = []

                self.cache_fs._log_debug(
                    "Compressed files downloaded. Updating media cache."
                )
                await self.update_media_cache(maybe_prune=False)
            else:
                self.cache_fs._log_debug(
                    "Compressed sources available; Media cache empty; populating"
                )
                await self.update_media_cache()

    async def update_compressed_cache(
        self,
        n_sources_per_dataset: Optional[int] = None,
        n_datasets: Optional[int] = None,
        exclude_tags: Optional[List[str]] = None,
        maybe_prune: bool = True,
    ) -> None:
        """
        Update the compressed cache by downloading new files.

        Args:
            n_sources_per_dataset: Optional override for number of sources per dataset
            n_datasets: Optional limit on number of datasets to process
        """
        if n_sources_per_dataset is None:
            n_sources_per_dataset = self.updater_config.num_sources_per_dataset

        if maybe_prune:
            await self.cache_fs.maybe_prune_cache(
                cache_type=CacheType.COMPRESSED,
                file_extensions=[self.compressed_file_extension],
            )

        # Reset tracking list before new downloads
        self._recently_downloaded_files = []

        datasets = self._get_filtered_datasets(exclude_tags=exclude_tags)
        if n_datasets is not None and n_datasets > 0:
            datasets = datasets[:n_datasets]
        np.random.shuffle(datasets)

        new_files = []
        for dataset in datasets:
            try:
                filenames = self._list_remote_dataset_files(dataset.path)
                if not filenames:
                    self.cache_fs._log_warning(f"No files found for {dataset.path}")
                    continue

                remote_paths = self._get_download_urls(dataset.path, filenames)
                to_download = self._select_files_to_download(
                    remote_paths, n_sources_per_dataset
                )

                output_dir = self.cache_fs.compressed_dir / dataset.path.split("/")[-1]

                self.cache_fs._log_debug(
                    f"Downloading {len(to_download)} files from {dataset.path}"
                )
                batch_files = await self._download_files(to_download, output_dir)

                # Track downloaded files
                self._recently_downloaded_files.extend(batch_files)
                new_files.extend(batch_files)
            except Exception as e:
                self.cache_fs._log_error(f"Error downloading from {dataset.path}: {e}")

        if new_files:
            self.cache_fs._log_debug(f"Added {len(new_files)} new compressed files")
        else:
            self.cache_fs._log_warning(f"No new files were added to compressed cache")

    async def update_media_cache(
        self, n_items_per_source: Optional[int] = None, maybe_prune: bool = True
    ) -> None:
        """
        Update the media cache by extracting from compressed sources.

        Args:
            n_items_per_source: Optional override for number of items per source
        """
        if n_items_per_source is None:
            n_items_per_source = self.updater_config.num_items_per_source

        if maybe_prune:
            await self.cache_fs.maybe_prune_cache(
                cache_type=CacheType.MEDIA, file_extensions=self.media_file_extensions
            )

        all_compressed_files = self.cache_fs.get_files(
            cache_type=CacheType.COMPRESSED,
            file_extensions=[self.compressed_file_extension],
            use_index=False,
        )

        if not all_compressed_files:
            self.cache_fs._log_warning(f"No compressed sources available")
            return

        compressed_files = filter_ready_files(all_compressed_files)

        if not compressed_files:
            self.cache_fs._log_warning(
                f"No ready compressed sources available. Files may still be downloading."
            )
            return

        valid_compressed_files = []
        for path in compressed_files:
            if not is_source_complete(path):
                try:
                    Path(path).unlink()
                except Exception as del_err:
                    self.cache_fs._log_error(
                        f"Failed to delete corrupted file {path}: {del_err}"
                    )
            else:
                valid_compressed_files.append(path)

        if len(valid_compressed_files) > 10:
            valid_compressed_files = np.random.choice(
                valid_compressed_files, size=10, replace=False
            ).tolist()

        new_files = []
        for source in valid_compressed_files:
            try:
                items = await self._extract_items_from_source(
                    source, n_items_per_source
                )
                new_files.extend(items)
            except Exception as e:
                self.cache_fs._log_error(f"Error extracting from {source}: {e}")

        if new_files:
            self.cache_fs._log_debug(f"Added {len(new_files)} new items to media cache")
        else:
            self.cache_fs._log_warning(f"No new items were added to media cache")

    def num_media_files(self) -> int:
        count = self.cache_fs.num_files(CacheType.MEDIA, self.media_file_extensions)
        return count == 0

    def num_compressed_files(self) -> int:
        count = self.cache_fs.num_files(
            CacheType.COMPRESSED, [self.compressed_file_extension]
        )
        return count == 0

    def _select_files_to_download(self, urls: List[str], count: int) -> List[str]:
        """Select random files to download"""
        return np.random.choice(
            urls, size=min(count, len(urls)), replace=False
        ).tolist()

    def _list_remote_dataset_files(self, dataset_path: str) -> List[str]:
        """List available files in a dataset with the parquet extension"""
        return list_hf_files(
            repo_id=dataset_path, extension=self.compressed_file_extension
        )

    def _get_download_urls(self, dataset_path: str, filenames: List[str]) -> List[str]:
        """Get Hugging Face download URLs for data files"""
        return [
            f"https://huggingface.co/datasets/{dataset_path}/resolve/main/{f}"
            for f in filenames
        ]

    async def _download_files(self, urls: List[str], output_dir: Path) -> List[Path]:
        """Download a subset of a remote dataset's compressed files"""
        return await download_files(urls, output_dir)
