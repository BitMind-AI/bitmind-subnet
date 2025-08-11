from typing import List, Optional

from gas.types import DatasetConfig, MediaType, Modality


class DatasetRegistry:
    """
    Registry for dataset configurations with filtering capabilities.
    """

    def __init__(self):
        self.datasets: List[DatasetConfig] = []

    def register(self, dataset: DatasetConfig) -> None:
        """
        Register a dataset with the system.

        Args:
            dataset: Dataset configuration to register
        """
        self.datasets.append(dataset)

    def register_all(self, datasets: List[DatasetConfig]) -> None:
        """
        Register multiple datasets with the system.

        Args:
            datasets: List of dataset configurations to register
        """
        for dataset in datasets:
            self.register(dataset)

    def get_datasets(
        self,
        modality: Optional[Modality] = None,
        media_type: Optional[MediaType] = None,
        tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        enabled_only: bool = True,
    ) -> List[DatasetConfig]:
        """
        Get datasets filtered by type, media_type, and/or tags.

        Args:
            modality: Filter by dataset type
            media_type: Filter by media_type
            tags: Filter by tags (dataset must have ALL specified tags)
            enabled_only: Only return enabled datasets

        Returns:
            List of matching datasets
        """
        result = self.datasets

        if enabled_only:
            result = [d for d in result if d.enabled]

        if modality:
            if isinstance(modality, str):
                modality = Modality(modality.lower())
            result = [d for d in result if d.modality == modality]

        if media_type:
            if isinstance(media_type, str):
                media_type = MediaType(media_type.lower())
            result = [d for d in result if d.media_type == media_type]

        if tags:
            result = [d for d in result if all(tag in d.tags for tag in tags)]

        if exclude_tags:
            result = [
                d for d in result if all(tag not in d.tags for tag in exclude_tags)
            ]

        return result

    def enable_dataset(self, path: str, enabled: bool = True) -> bool:
        """
        Enable or disable a dataset by path.

        Args:
            path: Dataset path to enable/disable
            enabled: Whether to enable or disable

        Returns:
            True if successful, False if dataset not found
        """
        for dataset in self.datasets:
            if dataset.path == path:
                dataset.enabled = enabled
                return True
        return False

    def get_dataset_by_path(self, path: str) -> Optional[DatasetConfig]:
        """
        Get a dataset by its path.

        Args:
            path: Dataset path to find

        Returns:
            Dataset config or None if not found
        """
        for dataset in self.datasets:
            if dataset.path == path:
                return dataset
        return None