import json
import random
import io
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np

from bitmind.cache.sampler.base import BaseSampler
from bitmind.cache.cache_fs import CacheConfig


class ImageSampler(BaseSampler):
    """
    Sampler for cached image data.

    This class provides access to images in the media cache,
    allowing sampling with or without metadata.
    """

    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config)

    @property
    def media_file_extensions(self) -> List[str]:
        """List of file extensions supported by this sampler"""
        return [".jpg", ".jpeg", ".png", ".webp"]

    async def sample(
        self,
        count: int = 1,
        remove_from_cache: bool = False,
        as_float32: bool = False,
        channels_first: bool = False,
        as_rgb: bool = True,
        require_mask: bool = False,
    ) -> Dict[str, Any]:
        """
        Sample random images and their metadata from the cache.

        Args:
            count: Number of images to sample
            remove_from_cache: Whether to remove sampled images from cache

        Returns:
            Dictionary containing:
                - count: Number of images successfully sampled
                - items: List of dictionaries containing:
                    - image: Image as numpy array in BGR format with shape (H, W, C)
                    - path: Path to the image file
                    - dataset: Source dataset name (if available)
                    - metadata: Additional metadata
        """
        cached_files = self.cache_fs.get_files(
            cache_type="media",
            file_extensions=self.media_file_extensions,
            group_by_source=True,
        )

        if not cached_files:
            self.cache_fs._log_warning("No images available in cache")
            return {"count": 0, "items": []}

        sampled_items = []

        attempts = 0
        max_attempts = count * 3

        while len(sampled_items) < count and attempts < max_attempts:
            attempts += 1

            source = random.choice(list(cached_files.keys()))
            if not cached_files[source]:
                del cached_files[source]
                if not cached_files:
                    break
                continue

            image_path = random.choice(cached_files[source])

            try:
                # Read image directly as numpy array using cv2
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Failed to load image {image_path}")

                if as_float32:  # else np.uint8
                    image = image.astype(np.float32) / 255.0

                if as_rgb:  # else bgr
                    image = image[:, :, [2, 1, 0]]

                if channels_first:  # else channels last
                    image = np.transpose(image, (2, 0, 1))

                metadata = {}
                metadata_path = image_path.with_suffix(".json")
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                    except Exception as e:
                        self.cache_fs._log_warning(
                            f"Error loading metadata for {image_path}: {e}"
                        )
                else:
                    metadata_path = None

                mask = None
                mask_path = Path(str(image_path).replace(image_path.suffix, "_mask.npy"))
                if mask_path.exists():
                    try:
                        mask = np.load(str(mask_path))
                    except Exception as e:
                        self.cache_fs._log_warning(
                            f"Error loading mask for {image_path}: {e}"
                        )
                        mask = None
                elif require_mask:
                    self.cache_fs._log_warning(
                        f"Mask {mask_path} does not exist for {image_path}"
                    )
                    mask_path = None

                if mask is None and require_mask:
                    continue

                item = {
                    "image": image,
                    "mask": mask,
                    "path": str(image_path),
                    "metadata_path": str(metadata_path),
                    "mask_path": str(mask_path),
                    "metadata": metadata,
                }

                if "source_parquet" in metadata:
                    item["source"] = metadata["source_parquet"]

                if "original_index" in metadata:
                    item["index"] = metadata["original_index"]

                sampled_items.append(item)

                if remove_from_cache:
                    try:
                        image_path.unlink(missing_ok=True)
                        metadata_path.unlink(missing_ok=True)
                        cached_files[source].remove(image_path)
                    except Exception as e:
                        self.cache_fs._log_warning(
                            f"Failed to remove {image_path}: {e}"
                        )

            except Exception as e:
                self.cache_fs._log_warning(f"Failed to load image {image_path}: {e}")
                cached_files[source].remove(image_path)
                continue

        return {"count": len(sampled_items), "items": sampled_items}
