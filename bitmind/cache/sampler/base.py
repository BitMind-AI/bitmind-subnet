from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


from bitmind.cache.cache_fs import CacheFS
from bitmind.types import CacheConfig


class BaseSampler(ABC):
    """
    Base class for samplers that provide access to cached media.
    """

    def __init__(self, cache_config: CacheConfig):
        self.cache_fs = CacheFS(cache_config)

    @property
    @abstractmethod
    def media_file_extensions(self) -> List[str]:
        """List of file extensions supported by this sampler"""
        pass

    @abstractmethod
    async def sample(self, count: int) -> Dict[str, Any]:
        """
        Sample items from the media cache.

        Args:
            count: Number of items to sample

        Returns:
            Dictionary with sampled items information
        """
        pass

    def get_available_files(self, use_index=True) -> List[Path]:
        """Get list of available files in the media cache"""
        return self.cache_fs.get_files(
            cache_type="media",
            file_extensions=self.media_file_extensions,
            use_index=use_index,
        )

    def get_available_count(self, use_index=True) -> int:
        """Get count of available files in the media cache"""
        return len(self.get_available_files(use_index))
