from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import time
import bittensor as bt

from bitmind.types import CacheConfig, CacheType
from bitmind.cache.util.filesystem import (
    analyze_directory,
    scale_size,
    format_size,
    print_directory_tree,
    is_source_complete,
)


class CacheFS:
    def __init__(self, config: CacheConfig):
        self.config = config

        self.cache_dir = config.get_path()
        self.compressed_dir = self.cache_dir / "sources"
        self.compressed_dir.mkdir(exist_ok=True, parents=True)

        self._log_prefix = f"[{config.modality}]"

        self._file_index = {}
        self._index_timestamp = {}
        self._index_ttl = 60

    def set_index_ttl(self, seconds: int) -> None:
        """Set the time-to-live for the file index in seconds."""
        self._index_ttl = max(0, seconds)
        self._log_debug(f"File index TTL set to {self._index_ttl} seconds")

    def invalidate_index(
        self, cache_type: Optional[Union[CacheType, str]] = None
    ) -> None:
        """Invalidate the file index for the specified cache type, or all indexes if None."""
        if cache_type is None:
            self._file_index = {}
            self._index_timestamp = {}
            self._log_info("All file indexes invalidated")
        else:
            if isinstance(cache_type, str):
                cache_type = CacheType(cache_type.lower())

            key = str(cache_type)
            if key in self._file_index:
                del self._file_index[key]
                if key in self._index_timestamp:
                    del self._index_timestamp[key]

    def _is_index_valid(self, cache_type: Union[CacheType, str]) -> bool:
        """Check if the index for the given cache type is still valid based on TTL."""
        if isinstance(cache_type, str):
            cache_type = CacheType(cache_type.lower())

        key = str(cache_type)
        if self._index_ttl <= 0 or key not in self._index_timestamp:
            return False

        return time.time() - self._index_timestamp[key] < self._index_ttl

    def num_files(
        self,
        cache_type: Union[CacheType, str] = CacheType.MEDIA,
        file_extensions: Optional[List[str]] = None,
        use_index: bool = True,
    ) -> int:
        """Returns the number of files of the given type and extensions."""
        files = self.get_files(
            cache_type=cache_type, file_extensions=file_extensions, use_index=use_index
        )
        return len(files)

    def get_files(
        self,
        cache_type: Union[CacheType, str] = CacheType.MEDIA,
        file_extensions: Optional[List[str]] = None,
        group_by_source: bool = False,
        use_index: bool = True,
    ) -> Union[List[Path], Dict[str, List[Path]]]:
        """
        Get files of the specified type with the given extensions.

        Args:
            cache_type: Type of cache to search (Media or Compressed)
            file_extensions: List of file extensions to filter by (e.g., ['.jpg', '.png'])
            group_by_source: Whether to group files by their source directory
            use_index: Whether to use indexed file list if available

        Returns:
            Either a list of file paths or a dictionary mapping source directories to lists of files
        """
        if isinstance(cache_type, str):
            cache_type = CacheType(cache_type.lower())

        if file_extensions is not None:
            file_extensions = set([ext.lower() for ext in file_extensions])

        key = str(cache_type)
        if use_index and self._is_index_valid(cache_type) and key in self._file_index:
            files = self._file_index[key]
            if group_by_source:
                return self._group_files_by_source(files, cache_type)
            return files

        if cache_type == CacheType.MEDIA:
            base_dir = self.cache_dir
        elif cache_type == CacheType.COMPRESSED:
            base_dir = self.compressed_dir

        files = []
        if base_dir.exists():
            dataset_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
            for dataset_dir in dataset_dirs:
                for file in dataset_dir.iterdir():
                    extension_match = (
                        file_extensions is None
                        or file.suffix.lower() in file_extensions
                    )
                    if file.is_file() and extension_match:
                        files.append(file)

        self._file_index[key] = files
        self._index_timestamp[key] = time.time()

        if group_by_source:
            return self._group_files_by_source(files, cache_type)

        return files

    def is_empty(self, cache_type: Union[CacheType, str]) -> bool:
        """
        Efficiently check if a cache directory is empty.

        Args:
            cache_type: Type of cache to check (Media or Compressed)

        Returns:
            bool: True if the cache is empty, False if it contains any files
        """
        if isinstance(cache_type, str):
            cache_type = CacheType(cache_type.lower())

        base_dir = (
            self.cache_dir if cache_type == CacheType.MEDIA else self.compressed_dir
        )

        if not base_dir.exists():
            return True

        try:
            # Check if there are any dataset directories
            dataset_dir = next(
                d for d in base_dir.iterdir() if d.is_dir() and d != self.compressed_dir
            )

            # Check if any dataset directory contains files
            try:
                next(
                    f
                    for f in dataset_dir.iterdir()
                    if f.is_file()
                    and (cache_type == CacheType.MEDIA or is_source_complete(f))
                )
                return False
            except StopIteration:
                return True
        except StopIteration:
            return True

    def _group_files_by_source(
        self, files: List[Path], cache_type: CacheType
    ) -> Dict[str, List[Path]]:
        """Helper method to group files by their source directory.
        TODO make this cache_type agnostic
        """
        if cache_type == CacheType.MEDIA:
            base_dir = self.cache_dir
        else:
            base_dir = self.compressed_dir

        result = {}
        for file in files:
            if file.exists():
                try:
                    rel_path = file.relative_to(base_dir)
                    subdir = str(rel_path.parent)
                except ValueError:
                    subdir = str(file.parent)

                if subdir not in result:
                    result[subdir] = []
                result[subdir].append(file)
        return result

    async def maybe_prune_cache(
        self,
        cache_type: Union[CacheType, str],
        file_extensions: Optional[List[str]],
    ) -> None:
        """
        Prune the cache if it exceeds the configured size limit.

        Args:
            cache_type: Type of cache to prune (Media or Compressed)
            file_extensions: List of file extensions to consider for pruning
        """
        if isinstance(cache_type, str):
            cache_type = CacheType(cache_type.lower())

        if cache_type == CacheType.COMPRESSED:
            max_gb = self.config.max_compressed_gb
        elif cache_type == CacheType.MEDIA:
            max_gb = self.config.max_media_gb

        max_bytes = scale_size(max_gb, "GB", "B")
        current_bytes = self.cache_size(cache_type, file_extensions, unit="B")
        num_files = self.num_files(cache_type, file_extensions)
        self._log_info(
            f"Pruning Check | {cache_type} cache | {num_files} files | {format_size(current_bytes, 'B', 'GB')}"
        )
        if current_bytes <= max_bytes:
            return

        files = self.get_files(
            cache_type=cache_type, file_extensions=file_extensions, use_index=True
        )

        files_dict = self._group_files_by_source(files, cache_type)

        for subdir in files_dict:
            files_dict[subdir] = sorted(
                files_dict[subdir],
                key=lambda f: f.stat().st_mtime if f.exists() else float("inf"),
            )

        self._log_info(f"Pruning cache to stay under {max_gb} GB...")

        n_removed = 0
        bytes_removed = 0
        remaining_bytes = current_bytes

        key = str(cache_type)
        has_index = key in self._file_index

        while remaining_bytes > max_bytes and any(
            files for files in files_dict.values()
        ):
            largest_subdir = max(
                [subdir for subdir, files in files_dict.items() if files],
                key=lambda subdir: len(files_dict[subdir]),
                default=None,
            )

            if largest_subdir is None:
                break

            file = files_dict[largest_subdir].pop(0)
            try:
                if file.exists():
                    file_size = file.stat().st_size
                    file.unlink()
                    if has_index:
                        try:
                            self._file_index[key].remove(file)
                        except ValueError:
                            pass

                    meta_file = file.with_suffix(".json")
                    if meta_file.exists():
                        meta_file.unlink()
                        if has_index:
                            try:
                                self._file_index[key].remove(meta_file)
                            except ValueError:
                                pass

                    n_removed += 1
                    bytes_removed += file_size
                    remaining_bytes -= file_size
            except Exception as e:
                self._log_error(f"Error removing file {file}: {e}")

        removed_gb_str = format_size(bytes_removed, "B", "GB")
        new_gb_str = self.cache_size(cache_type, file_extensions, "GB", as_str=True)
        self._log_info(
            f"Removed: {n_removed} files; {removed_gb_str} | New size: {new_gb_str}"
        )

    def cache_size(
        self,
        cache_type: CacheType,
        file_extensions: Optional[List[str]] = None,
        unit: str = "GB",
        as_str: bool = False,
        use_index: bool = True,
    ) -> Union[str, float]:
        """
        Returns size of media or compressed cache.

        Args:
            cache_type: Type of cache to measure (Media or Compressed)
            file_extensions: List of file extensions to filter by
            unit: Unit to return the size in (e.g., 'B', 'KB', 'MB', 'GB')
            as_str: Whether to return the size as a formatted string
            use_index: Whether to use indexed file list if available

        Returns:
            Size of the cache, either as a float or a formatted string with units
        """
        files = self.get_files(
            cache_type=cache_type, file_extensions=file_extensions, use_index=use_index
        )
        total_bytes = sum(f.stat().st_size for f in files if f.exists())
        if as_str:
            return format_size(total_bytes, "B", unit)
        return scale_size(total_bytes, "B", unit)

    def get_cache_stats(self, use_index: bool = True) -> Dict[str, Any]:
        """
        Get statistics about the cache.

        Args:
            use_index: Whether to use indexed file list if available

        Returns:
            Dictionary with cache statistics including file counts and sizes
        """
        media_files = self.get_files(CacheType.MEDIA, use_index=use_index)
        media_count = len(media_files)
        media_bytes = sum(f.stat().st_size for f in media_files if f.exists())
        media_gb = scale_size(media_bytes, "B", "GB")

        compressed_files = self.get_files(CacheType.COMPRESSED, use_index=use_index)
        compressed_count = len(compressed_files)
        compressed_bytes = sum(f.stat().st_size for f in compressed_files if f.exists())
        compressed_gb = scale_size(compressed_bytes, "B", "GB")

        return {
            "cache_dir": str(self.cache_dir),
            "modality": self.config.modality,
            "media_type": self.config.media_type,
            "media_count": media_count,
            "media_bytes": media_bytes,
            "media_gb": media_gb,
            "compressed_count": compressed_count,
            "compressed_bytes": compressed_bytes,
            "compressed_gb": compressed_gb,
            "total_count": media_count + compressed_count,
            "total_bytes": media_bytes + compressed_bytes,
            "total_gb": media_gb + compressed_gb,
        }

    def print_directory_tree(
        self, min_file_count: int = 1, include_sources: bool = True
    ):
        """Print a tree representation of the cache directory structure."""
        exclude_dirs = [] if include_sources else ["sources"]

        self._log_info(f"Analyzing cache directory structure: {self.cache_dir}")
        tree_data = analyze_directory(
            self.cache_dir,
            exclude_dirs=exclude_dirs,
            min_file_count=min_file_count,
            log_func=self._log_info,
        )

        self._log_info(f"\n{self.cache_dir}")
        self._log_info(f"Modality: {self.config.modality}")
        self._log_info(f"Media Type: {self.config.media_type}")
        self._log_info(f"Total Size: {format_size(tree_data['size'])}")
        self._log_info(f"Total Files: {tree_data['count']}")
        if not include_sources:
            self._log_info(
                "Note: Source directories are excluded from the visualization"
            )
        self._log_info("-" * 80)

        print_directory_tree(tree_data, "", True, "", self._log_info)

    def _log_info(self, message: str) -> None:
        bt.logging.info(f"{self._log_prefix} {message}")

    def _log_warning(self, message: str) -> None:
        bt.logging.warning(f"{self._log_prefix} {message}")

    def _log_error(self, message: str) -> None:
        bt.logging.error(f"{self._log_prefix} {message}")

    def _log_debug(self, message: str) -> None:
        bt.logging.debug(f"{self._log_prefix} {message}")

    def _log_success(self, message: str) -> None:
        bt.logging.debug(f"{self._log_prefix} {message}")

    def _log_trace(self, message: str) -> None:
        bt.logging.trace(f"{self._log_prefix} {message}")
