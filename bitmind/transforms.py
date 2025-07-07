import math
import random
from typing import Tuple, Union, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2
from scipy import ndimage

from bitmind.generation.util.image import ensure_mask_3d

TARGET_IMAGE_SIZE = (256, 256)


class SourceType(Enum):
    """
    Enum for image source type for resolution augmentation logic.
    REAL: Real-world images (e.g., from cameras, web, etc.)
    GENERATED: AI-generated images (e.g., from diffusion models)
    """
    REAL = "real"
    GENERATED = "generated"


@dataclass
class ResolutionAugmentationConfig:
    """
    Configuration for resolution-based augmentations.

    Attributes:
        generated_to_real_ratio: Percentage of generated images to resize to real resolutions.
        real_to_generated_ratio: Percentage of real images to resize to generated resolutions.
        default_target_size: Default target size for standard processing.
        enable_dynamic_resolution: Whether to enable dynamic resolution sampling.
    """
    generated_to_real_ratio: float = 0.3
    real_to_generated_ratio: float = 0.2
    default_target_size: Tuple[int, int] = (256, 256)
    enable_dynamic_resolution: bool = True


class ResolutionSampler:
    """
    Samples image resolutions from realistic distributions based on source type.
    """
    def __init__(self):
        """
        Initialize the sampler with hardcoded real and generated resolution distributions.
        """
        self.real_resolutions = [
            (1600, 1200),   # 4:3, very common
            (3264, 2448),   # 4:3, common phone
            (2048, 1536),   # 4:3, iPad/camera
            (4000, 3000),   # 4:3, high-res phone
            (800, 600),     # 4:3, legacy
            (1024, 768),    # 4:3, legacy
            (2592, 1944),   # 4:3, common
            (3008, 2000),   # 3:2, DSLR
            (5184, 3456),   # 3:2, DSLR
            (1280, 960),    # 4:3, web
            (3000, 2000),   # 3:2, DSLR
            (3888, 2592),   # 3:2, DSLR
            (2448, 3264),   # 4:3, phone
            (2816, 2112),   # 4:3, camera
            (1280, 720),    # 16:9, HD
            (1920, 1080),   # 16:9, HD
            (2560, 1440),   # 16:9, QHD
            (3840, 2160),   # 16:9, 4K
            (512, 512),     # square, rare
            (1024, 1024),   # square, rare
            (1200, 1200),   # square, rare
            (1080, 1080),   # square, rare
            (720, 720),     # square, rare
            (4096, 2160),   # 17:9, rare
            (5000, 4000),   # 5:4, rare
            (6000, 4000),   # 3:2, rare
        ]
        self.real_weights = [
            0.10,  # 1600x1200
            0.09,  # 3264x2448
            0.08,  # 2048x1536
            0.08,  # 4000x3000
            0.07,  # 800x600
            0.07,  # 1024x768
            0.06,  # 2592x1944
            0.05,  # 3008x2000
            0.05,  # 5184x3456
            0.04,  # 1280x960
            0.04,  # 3000x2000
            0.03,  # 3888x2592
            0.03,  # 2448x3264
            0.03,  # 2816x2112
            0.03,  # 1280x720
            0.03,  # 1920x1080
            0.02,  # 2560x1440
            0.02,  # 3840x2160
            0.01,  # 512x512
            0.01,  # 1024x1024
            0.01,  # 1200x1200
            0.01,  # 1080x1080
            0.01,  # 720x720
            0.01,  # 4096x2160
            0.01,  # 5000x4000
            0.01,  # 6000x4000
        ]
        # Normalize to sum to 1.0
        self.real_weights = np.array(self.real_weights)
        self.real_weights = self.real_weights / self.real_weights.sum()
        # Synthetic (generated) resolutions and weights based on popular model defaults (see models.py)
        # 1024x1024 (GPT-4V, SDXL, etc.) is given the highest weight, followed by 2048x2048 (Midjourney)
        self.generated_resolutions = [
            (1024, 1024),   # GPT-4V, SDXL, etc.
            (2048, 2048),   # Midjourney default
            (512, 512),     # SD, AnimateDiff, etc.
            (768, 768),     # SDXL, AnimateDiff, etc.
            (640, 640),     # AnimateDiff, etc.
            (896, 896),     # AnimateDiff, etc.
            (1536, 1536),   # IF, etc.
            (800, 800),     # AnimateDiff, etc.
            (960, 960),     # AnimateDiff, etc.
            (512, 768),     # SDXL, AnimateDiff, etc.
            (768, 512),     # SDXL, AnimateDiff, etc.
            (512, 1024),    # AnimateDiff, etc.
            (1024, 512),    # AnimateDiff, etc.
            (512, 896),     # AnimateDiff, etc.
            (896, 512),     # AnimateDiff, etc.
            (720, 1280),    # HunyuanVideo, Wan2.1, etc.
            (1280, 720),    # HunyuanVideo, Wan2.1, etc.
            (832, 1104),    # HunyuanVideo
            (1104, 832),    # HunyuanVideo
            (480, 832),     # Mochi, Wan2.1
            (480, 848),     # Mochi
            (720, 720),     # AnimateDiff, etc.
            (854, 480),     # AnimateDiff, etc.
            (1024, 576),    # AnimateDiff, etc.
            (1920, 1080),   # Some video/image models
            (3840, 2160),   # 4K, rare
            (2560, 1440),   # QHD, rare
        ]
        self.generated_weights = [
            0.16,  # 1024x1024 (GPT-4V, SDXL, etc.)
            0.13,  # 2048x2048 (Midjourney)
            0.12,  # 512x512
            0.11,  # 768x768
            0.05,  # 640x640
            0.04,  # 896x896
            0.01,  # 1536x1536
            0.01,  # 800x800
            0.01,  # 960x960
            0.07,  # 512x768
            0.07,  # 768x512
            0.03,  # 512x1024
            0.03,  # 1024x512
            0.01,  # 512x896
            0.01,  # 896x512
            0.02,  # 720x1280
            0.02,  # 1280x720
            0.01,  # 832x1104
            0.01,  # 1104x832
            0.01,  # 480x832
            0.01,  # 480x848
            0.01,  # 720x720
            0.01,  # 854x480
            0.01,  # 1024x576
            0.01,  # 1920x1080
            0.005, # 3840x2160
            0.005, # 2560x1440
        ]
        # Normalize to sum to 1.0
        self.generated_weights = np.array(self.generated_weights)
        self.generated_weights = self.generated_weights / self.generated_weights.sum()

    def sample_resolution(self, source_type: SourceType) -> Tuple[int, int]:
        """
        Sample a resolution based on source type distribution.

        Args:
            source_type: Whether the image is real or generated.

        Returns:
            Tuple of (width, height).
        """
        if source_type == SourceType.REAL:
            idx = np.random.choice(len(self.real_resolutions), p=self.real_weights)
            return self.real_resolutions[idx]
        idx = np.random.choice(len(self.generated_resolutions), p=self.generated_weights)
        return self.generated_resolutions[idx]

    def sample_cross_domain_resolution(self, source_type: SourceType) -> Tuple[int, int]:
        """
        Sample a resolution from the opposite domain for adversarial training.

        Args:
            source_type: Original source type (will sample from opposite domain).

        Returns:
            Tuple of (width, height) from opposite domain.
        """
        if source_type == SourceType.REAL:
            idx = np.random.choice(len(self.generated_resolutions), p=self.generated_weights)
            return self.generated_resolutions[idx]
        idx = np.random.choice(len(self.real_resolutions), p=self.real_weights)
        return self.real_resolutions[idx]


class DynamicResize:
    """
    Resize transform that uses dynamic resolution sampling based on source type.
    """
    def __init__(
        self,
        config: ResolutionAugmentationConfig,
        resolution_sampler: ResolutionSampler,
        target_size: Optional[Tuple[int, int]] = None
    ):
        """
        Args:
            config: ResolutionAugmentationConfig instance.
            resolution_sampler: ResolutionSampler instance.
            target_size: Optional override for target size.
        """
        self.config = config
        self.resolution_sampler = resolution_sampler
        self.target_size = target_size or config.default_target_size

    def __call__(
        self,
        img: np.ndarray,
        mask: Optional[np.ndarray] = None,
        source_type: Optional[SourceType] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply dynamic resize based on source type and configuration.

        Args:
            img: Input image array.
            mask: Optional mask array.
            source_type: Source type for dynamic resolution logic.

        Returns:
            Resized image, or tuple of (image, mask) if mask provided.
        """
        if not self.config.enable_dynamic_resolution or source_type is None:
            target_size = self.target_size
        else:
            should_cross_domain = False
            if source_type == SourceType.GENERATED:
                should_cross_domain = random.random() < self.config.generated_to_real_ratio
            else:
                should_cross_domain = random.random() < self.config.real_to_generated_ratio
            if should_cross_domain:
                target_size = self.resolution_sampler.sample_cross_domain_resolution(source_type)
            else:
                target_size = self.resolution_sampler.sample_resolution(source_type)
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            return img_resized, mask_resized
        return img_resized


class DynamicRandomResizedCrop:
    """
    Random resized crop that uses dynamic resolution sampling.
    """
    def __init__(
        self,
        config: ResolutionAugmentationConfig,
        resolution_sampler: ResolutionSampler,
        scale: Tuple[float, float] = (0.2, 1.0),
        target_size: Optional[Tuple[int, int]] = None
    ):
        """
        Args:
            config: ResolutionAugmentationConfig instance.
            resolution_sampler: ResolutionSampler instance.
            scale: Range for random crop area.
            target_size: Optional override for target size.
        """
        self.config = config
        self.resolution_sampler = resolution_sampler
        self.scale = scale
        self.target_size = target_size or config.default_target_size
        self.params: Dict[str, Any] = {}

    def __call__(
        self,
        img: np.ndarray,
        mask: Optional[np.ndarray] = None,
        source_type: Optional[SourceType] = None,
        crop_params: Optional[Tuple[int, int, int, int]] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply dynamic random resized crop.

        Args:
            img: Input image array.
            mask: Optional mask array.
            source_type: Source type for dynamic resolution logic.
            crop_params: Pre-computed crop parameters (i, j, h, w).

        Returns:
            Cropped and resized image, or tuple of (image, mask) if mask provided.
        """
        if not self.config.enable_dynamic_resolution or source_type is None:
            target_size = self.target_size
        else:
            should_cross_domain = False
            if source_type == SourceType.GENERATED:
                should_cross_domain = random.random() < self.config.generated_to_real_ratio
            else:
                should_cross_domain = random.random() < self.config.real_to_generated_ratio
            if should_cross_domain:
                target_size = self.resolution_sampler.sample_cross_domain_resolution(source_type)
            else:
                target_size = self.resolution_sampler.sample_resolution(source_type)
        img_cropped = self._resized_crop(
            img, mask=mask, crop_params=crop_params,
            target_size=target_size, interpolation=cv2.INTER_LINEAR
        )
        if mask is not None:
            prev_crop_params = self.params.get("crop_params")
            mask_cropped = self._resized_crop(
                mask, mask=None, crop_params=prev_crop_params,
                target_size=target_size, interpolation=cv2.INTER_NEAREST
            )
            return img_cropped, mask_cropped
        return img_cropped

    def _resized_crop(
        self,
        img: np.ndarray,
        mask: Optional[np.ndarray] = None,
        crop_params: Optional[Tuple[int, int, int, int]] = None,
        target_size: Tuple[int, int] = (256, 256),
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        Perform random resized crop with mask-aware cropping.

        Args:
            img: Input image array.
            mask: Optional mask array.
            crop_params: Pre-computed crop parameters (i, j, h, w).
            target_size: Target size for resize.
            interpolation: Interpolation method for resize.

        Returns:
            Cropped and resized image array.
        """
        height, width = img.shape[:2]
        if crop_params is None:
            area = height * width
            target_area = area * np.random.uniform(*self.scale)
            h = w = int(round(np.sqrt(target_area)))
            h = min(h, height)
            w = min(w, width)
            if mask is not None:
                coords = np.where(mask > 0)
                if len(coords[0]) > 0 and len(coords[1]) > 0:
                    ys, xs = coords[0], coords[1]
                    x0, x1 = xs.min(), xs.max()
                    y0, y1 = ys.min(), ys.max()
                    min_i = max(0, y1 - h + 1)
                    max_i = min(y0, height - h)
                    min_j = max(0, x1 - w + 1)
                    max_j = min(x0, width - w)
                    if min_i <= max_i and min_j <= max_j:
                        i = np.random.randint(min_i, max_i + 1)
                        j = np.random.randint(min_j, max_j + 1)
                    else:
                        i = max(0, min(height - h, (y0 + y1) // 2 - h // 2))
                        j = max(0, min(width - w, (x0 + x1) // 2 - w // 2))
                else:
                    i = np.random.randint(0, height - h + 1)
                    j = np.random.randint(0, width - w + 1)
            else:
                i = np.random.randint(0, height - h + 1)
                j = np.random.randint(0, width - w + 1)
        else:
            i, j, h, w = crop_params
        self.params["crop_params"] = (i, j, h, w)
        cropped = img[i:i+h, j:j+w]
        if cropped.ndim == 3:
            cropped = cropped
        return cv2.resize(cropped, target_size, interpolation=interpolation)


class CenterCrop:
    """
    Center crop an image to a square.
    """
    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply center crop transform to an image and optionally a mask.

        Args:
            img: Input image array.
            mask: Optional mask array.

        Returns:
            Cropped image, or tuple of (image, mask) if mask provided.
        """
        img_cropped = self._crop(img)
        if mask is not None:
            mask_cropped = self._crop(mask)
            return img_cropped, mask_cropped
        return img_cropped

    def _crop(self, img: np.ndarray) -> np.ndarray:
        """
        Perform center crop to square.

        Args:
            img: Input image array.

        Returns:
            Center-cropped image array.
        """
        h, w = img.shape[:2]
        m = min(h, w)
        i = (h - m) // 2
        j = (w - m) // 2
        return img[i:i + m, j:j + m]


class RandomRotation:
    """
    Randomly rotate an image (and mask) by a random angle within a range.
    """
    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]] = 20,
        p: float = 0.5,
        reshape: bool = False,
        mode: str = "reflect",
        order: int = 2
    ):
        """
        Args:
            degrees: Range of degrees to select from. If float, uses (-degrees, degrees).
            p: Probability of rotating the image.
            reshape: If True, expands output image to fit rotated image.
            mode: How to fill the border ('reflect', 'constant', etc).
            order: Interpolation order (0-5).
        """
        if isinstance(degrees, (tuple, list)):
            self.degrees = degrees
        else:
            self.degrees = (-degrees, degrees)
        self.p = p
        self.reshape = reshape
        self.mode = mode
        self.order = order
        self.params: Dict[str, Any] = {}

    def __call__(
        self,
        img: np.ndarray,
        mask: Optional[np.ndarray] = None,
        rotate: Optional[bool] = None,
        angle: Optional[float] = None,
        order: Optional[int] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Randomly rotate the image and mask.

        Args:
            img: Input image array.
            mask: Optional mask array.
            rotate: Pre-computed rotation decision.
            angle: Pre-computed rotation angle.
            order: Pre-computed interpolation order.

        Returns:
            Rotated image, or tuple of (image, mask) if mask provided.
        """
        if rotate is None:
            rotate = np.random.random() < self.p
        self.params["rotate"] = rotate
        if not rotate:
            return (img, mask) if mask is not None else img
        order = self.order if order is None else order
        if isinstance(order, (tuple, list)):
            order = random.randint(order[0], order[1])
        if angle is None:
            angle = random.uniform(self.degrees[0], self.degrees[1])
        self.params["order"] = order
        self.params["angle"] = angle
        img_rot = ndimage.rotate(img, angle, reshape=self.reshape, mode=self.mode, order=order, axes=(0, 1))
        if mask is not None:
            mask_rot = ndimage.rotate(mask, angle, reshape=self.reshape, mode=self.mode, order=0, axes=(0, 1))
            return img_rot, mask_rot
        return img_rot


class RandomHorizontalFlip:
    """
    Random horizontal flip with configurable probability.
    """
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of flipping the image.
        """
        self.p = p
        self.params: Dict[str, Any] = {}

    def __call__(
        self,
        img: np.ndarray,
        mask: Optional[np.ndarray] = None,
        flip: Optional[bool] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply random horizontal flip.

        Args:
            img: Input image array.
            mask: Optional mask array.
            flip: Pre-computed flip decision.

        Returns:
            Flipped image, or tuple of (image, mask) if mask provided.
        """
        if flip is None:
            flip = np.random.random() < self.p
        self.params["flip"] = flip
        if flip:
            img = np.fliplr(img)
            if mask is not None:
                mask = np.fliplr(mask)
        if mask is not None:
            return img, mask
        return img


class RandomVerticalFlip:
    """
    Random vertical flip with configurable probability.
    """
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of flipping the image.
        """
        self.p = p
        self.params: Dict[str, Any] = {}

    def __call__(
        self,
        img: np.ndarray,
        mask: Optional[np.ndarray] = None,
        flip: Optional[bool] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply random vertical flip.

        Args:
            img: Input image array.
            mask: Optional mask array.
            flip: Pre-computed flip decision.

        Returns:
            Flipped image, or tuple of (image, mask) if mask provided.
        """
        if flip is None:
            flip = np.random.random() < self.p
        self.params["flip"] = flip
        if flip:
            img = np.flipud(img)
            if mask is not None:
                mask = np.flipud(mask)
        if mask is not None:
            return img, mask
        return img


class ComposeTransforms:
    """
    Compose multiple transforms with source type awareness.
    """
    def __init__(self, transforms: List[Any]):
        """
        Args:
            transforms: List of transform objects to compose.
        """
        self.transforms = transforms
        self.params: Dict[str, Any] = {}

    def __call__(
        self,
        img: np.ndarray,
        mask: Optional[np.ndarray] = None,
        source_type: Optional[SourceType] = None,
        reuse_params: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply composed transforms.

        Args:
            img: Input image array.
            mask: Optional mask array.
            source_type: Source type for dynamic resolution logic.
            reuse_params: Whether to reuse previous parameters.

        Returns:
            Transformed image, or tuple of (image, mask) if mask provided.
        """
        if not reuse_params:
            self.params = {}
        current_img = img
        current_mask = mask
        for transform in self.transforms:
            transform_name = transform.__class__.__name__
            if hasattr(transform, '__call__'):
                import inspect
                sig = inspect.signature(transform.__call__)
                supports_source_type = 'source_type' in sig.parameters
                if supports_source_type:
                    if current_mask is not None:
                        result = transform(current_img, mask=current_mask, source_type=source_type)
                        if isinstance(result, tuple):
                            current_img, current_mask = result
                        else:
                            current_img = result
                    else:
                        current_img = transform(current_img, source_type=source_type)
                else:
                    if current_mask is not None:
                        result = transform(current_img, mask=current_mask)
                        if isinstance(result, tuple):
                            current_img, current_mask = result
                        else:
                            current_img = result
                    else:
                        current_img = transform(current_img)
            if hasattr(transform, 'params'):
                self.params[transform_name] = transform.params
        if current_mask is not None:
            return current_img, current_mask
        return current_img


def get_base_transforms(
    config: ResolutionAugmentationConfig,
    resolution_sampler: ResolutionSampler
) -> ComposeTransforms:
    """
    Get basic transforms with dynamic resolution support.

    Args:
        config: Resolution augmentation configuration.
        resolution_sampler: Resolution sampler instance.

    Returns:
        Composed transform pipeline.
    """
    return ComposeTransforms([
        CenterCrop(),
        DynamicResize(config, resolution_sampler),
    ])


def get_augmentation_transforms(
    config: ResolutionAugmentationConfig,
    resolution_sampler: ResolutionSampler
) -> ComposeTransforms:
    """
    Get augmentation transforms with dynamic resolution support.

    Args:
        config: Resolution augmentation configuration.
        resolution_sampler: Resolution sampler instance.

    Returns:
        Composed transform pipeline with augmentations.
    """
    return ComposeTransforms([
        DynamicRandomResizedCrop(config, resolution_sampler),
        RandomRotation(degrees=20, p=0.5),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.2),
    ])


def apply_resolution_augmentations(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    source_type: Optional[SourceType] = None,
    config: Optional[ResolutionAugmentationConfig] = None,
    use_augmentations: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply resolution-aware augmentations to an image.

    Args:
        img: Input image array.
        mask: Optional mask array.
        source_type: Source type (real or generated).
        config: Augmentation configuration.
        use_augmentations: Whether to use augmentations or just base transforms.

    Returns:
        Augmented image, or tuple of (image, mask) if mask provided.
    """
    if config is None:
        config = ResolutionAugmentationConfig()
    resolution_sampler = ResolutionSampler()
    if use_augmentations:
        transforms = get_augmentation_transforms(config, resolution_sampler)
    else:
        transforms = get_base_transforms(config, resolution_sampler)
    return transforms(img, mask=mask, source_type=source_type)


def apply_random_augmentations(
    inputs,
    target_image_size=None,
    mask=None,
    level_probs=None,
    level=None,
    source_type: Optional[SourceType] = None
):
    """
    Backward compatibility wrapper for the old apply_random_augmentations function.

    Args:
        inputs: np.ndarray or tuple of np.ndarray. Image(s) to transform.
        target_image_size: int or tuple. Output size for resize (ignored in favor of dynamic resolution).
        mask: np.ndarray, optional. Binary mask.
        level_probs: dict, not used in new implementation.
        level: int, not used in new implementation.
        source_type: SourceType, optional. Source type for dynamic resolution logic.

    Returns:
        tuple: (aug_image, aug_mask, 0, {}) for compatibility.
    """
    config = ResolutionAugmentationConfig()
    if isinstance(inputs, tuple):
        aug_A = apply_resolution_augmentations(
            inputs[0], mask=None, source_type=source_type, config=config
        )
        aug_B = apply_resolution_augmentations(
            inputs[1], mask=None, source_type=source_type, config=config
        )
        transformed = np.concatenate([aug_A, aug_B], axis=0)
        return transformed, None, 0, {}
    else:
        aug_image, aug_mask = apply_resolution_augmentations(
            inputs, mask=mask, source_type=source_type, config=config
        )
        return aug_image, aug_mask, 0, {}
