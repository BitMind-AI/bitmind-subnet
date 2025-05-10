from typing import Dict, Optional, Any

import bittensor as bt

from .base import BaseSampler


class SamplerRegistry:
    """
    Registry for cache samplers.
    """

    def __init__(self):
        self._samplers: Dict[str, BaseSampler] = {}

    def register(self, name: str, sampler: BaseSampler) -> None:
        if name in self._samplers:
            bt.logging.warning(f"Sampler {name} already registered, will be replaced")
        self._samplers[name] = sampler

    def get(self, name: str) -> Optional[BaseSampler]:
        return self._samplers.get(name)

    def get_all(self) -> Dict[str, BaseSampler]:
        return dict(self._samplers)

    def deregister(self, name: str) -> None:
        if name in self._samplers:
            del self._samplers[name]

    async def sample(self, name: str, count: int, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Sample from a specific sampler.

        Args:
            name: Name of the sampler to use
            count: Number of items to sample

        Returns:
            The sampled items or None if sampler not found
        """
        sampler = self.get(name)
        if not sampler:
            bt.logging.error(f"Sampler {name} not found")
            return None

        return await sampler.sample(count, **kwargs)

    async def sample_all(self, count_per_sampler: int = 1) -> Dict[str, Dict[str, Any]]:
        """
        Sample from all samplers.

        Args:
            count_per_sampler: Number of items to sample from each sampler

        Returns:
            Dictionary mapping sampler names to their samples
        """
        results = {}
        for name, sampler in self._samplers.items():
            results[name] = await sampler.sample(count_per_sampler)
        return results
