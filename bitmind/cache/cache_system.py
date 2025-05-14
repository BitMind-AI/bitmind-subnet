from typing import Any, Dict, List, Optional, Type
import traceback

import asyncio
import bittensor as bt

from bitmind.types import CacheUpdaterConfig, CacheConfig, Modality, MediaType
from bitmind.cache.datasets import DatasetRegistry, initialize_dataset_registry
from bitmind.cache.updater import (
    BaseUpdater,
    UpdaterRegistry,
    ImageUpdater,
    VideoUpdater,
)
from bitmind.cache.sampler import (
    BaseSampler,
    SamplerRegistry,
    ImageSampler,
    VideoSampler,
)


class CacheSystem:
    """
    Main facade for the caching system.
    """

    def __init__(self):
        self.dataset_registry = DatasetRegistry()
        self.updater_registry = UpdaterRegistry()
        self.sampler_registry = SamplerRegistry()

    async def initialize(
        self,
        base_dir,
        max_compressed_gb,
        max_media_gb,
        media_files_per_source,
    ):
        try:
            dataset_registry = initialize_dataset_registry()
            for dataset in dataset_registry.datasets:
                self.register_dataset(dataset)

            for modality in Modality:
                for media_type in MediaType:
                    cache_config = CacheConfig(
                        base_dir=base_dir,
                        modality=modality.value,
                        media_type=media_type.value,
                        max_compressed_gb=max_compressed_gb,
                        max_media_gb=max_media_gb,
                    )
                    sampler_class = (
                        ImageSampler if modality == Modality.IMAGE else VideoSampler
                    )
                    self.create_sampler(
                        name=f"{media_type.value}_{modality.value}_sampler",
                        sampler_class=sampler_class,
                        cache_config=cache_config,
                    )

                    # synthetic video updater not currently used, only generate locally
                    if not (
                        modality == Modality.VIDEO and media_type == MediaType.SYNTHETIC
                    ):
                        updater_config = CacheUpdaterConfig(
                            num_sources_per_dataset=1,  # one compressed source per dataset for initialization
                            num_items_per_source=media_files_per_source,
                        )
                        updater_class = (
                            ImageUpdater if modality == Modality.IMAGE else VideoUpdater
                        )
                        self.create_updater(
                            name=f"{media_type.value}_{modality.value}_updater",
                            updater_class=updater_class,
                            cache_config=cache_config,
                            updater_config=updater_config,
                        )

            # Initialize caches (populate if empty)
            bt.logging.info("Starting initial cache population")
            await self.initialize_caches()
            bt.logging.info("Initial cache population complete")

        except Exception as e:
            bt.logging.error(f"Error initializing caches: {e}")
            bt.logging.error(traceback.format_exc())

    def register_dataset(self, dataset) -> None:
        """
        Register a dataset with the system.

        Args:
            dataset: Dataset configuration to register
        """
        self.dataset_registry.register(dataset)

    def register_datasets(self, datasets: List[Any]) -> None:
        """
        Register multiple datasets with the system.

        Args:
            datasets: List of dataset configurations to register
        """
        self.dataset_registry.register_all(datasets)

    def create_updater(
        self,
        name: str,
        updater_class: Type[BaseUpdater],
        cache_config: CacheConfig,
        updater_config: CacheUpdaterConfig,
    ) -> BaseUpdater:
        """
        Create and register an updater.

        Args:
            name: Unique name for the updater
            updater_class: Updater class to instantiate
            cache_config: Cache configuration
            updater_config: Updater configuration

        Returns:
            The created updater instance
        """
        updater = updater_class(
            cache_config=cache_config,
            updater_config=updater_config,
            data_manager=self.dataset_registry,
        )
        self.updater_registry.register(name, updater)
        return updater

    def create_sampler(
        self, name: str, sampler_class: Type[BaseSampler], cache_config: CacheConfig
    ) -> BaseSampler:
        """
        Create and register a sampler.

        Args:
            name: Unique name for the sampler
            sampler_class: Sampler class to instantiate
            cache_config: Cache configuration

        Returns:
            The created sampler instance
        """
        sampler = sampler_class(cache_config=cache_config)
        self.sampler_registry.register(name, sampler)
        return sampler

    async def initialize_caches(self) -> None:
        """
        Initialize all caches to ensure they have content.
        This is typically called during system startup.
        """
        updaters = self.updater_registry.get_all()
        names = [name for name, _ in updaters.items()]
        bt.logging.debug(f"Initializing {len(updaters)} caches: {names}")

        cache_init_tasks = []
        for name, updater in updaters.items():
            cache_init_tasks.append(updater.initialize_cache())

        if cache_init_tasks:
            await asyncio.gather(*cache_init_tasks)

    async def update_compressed_caches(self) -> None:
        """
        Update all compressed caches in parallel
        This is typically called from a block callback.
        """
        updaters = self.updater_registry.get_all()
        names = [name for name, _ in updaters.items()]
        bt.logging.trace(f"Updating {len(updaters)} compressed caches: {names}")

        tasks = []
        for name, updater in updaters.items():
            tasks.append(updater.update_compressed_cache())

        if tasks:
            await asyncio.gather(*tasks)

    async def update_media_caches(self) -> None:
        """
        Update all media caches in parallel.
        This is typically called from a block callback.
        """
        updaters = self.updater_registry.get_all()
        names = [name for name, _ in updaters.items()]
        bt.logging.debug(f"Updating {len(updaters)} media caches: {names}")

        tasks = []
        for name, updater in updaters.items():
            tasks.append(updater.update_media_cache())

        if tasks:
            await asyncio.gather(*tasks)

    async def sample(self, name: str, count: int, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Sample from a specific sampler.

        Args:
            name: Name of the sampler to use
            count: Number of items to sample

        Returns:
            The sampled items or None if sampler not found
        """
        return await self.sampler_registry.sample(name, count, **kwargs)

    async def sample_all(self, count: int = 1) -> Dict[str, Dict[str, Any]]:
        """
        Sample from all samplers.

        Args:
            count: Number of items to sample from each sampler

        Returns:
            Dictionary mapping sampler names to their samples
        """
        return await self.sampler_registry.sample_all(count)

    @property
    def samplers(self):
        """
        Get all registered samplers.

        Returns:
            Dictionary of sampler names to sampler instances
        """
        return self.sampler_registry.get_all()

    @property
    def updaters(self):
        """
        Get all registered updaters.

        Returns:
            Dictionary of updater names to updater instances
        """
        return self.updater_registry.get_all()
