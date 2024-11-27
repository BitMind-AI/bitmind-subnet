from functools import partial
import torchvision.transforms as transforms

from bitmind.validator.config import TARGET_IMAGE_SIZE
from bitmind.utils.image_transforms import (
    center_crop,
    RandomResizedCropWithParams,
    RandomHorizontalFlipWithParams,
    RandomVerticalFlipWithParams,
    RandomRotationWithParams,
    ConvertToRGB,
    ComposeWithParams,
    base_transforms,
    random_aug_transforms
)


TRANSFORMS = [
    center_crop,
    RandomHorizontalFlipWithParams,
    RandomVerticalFlipWithParams,
    partial(RandomRotationWithParams, degrees=20, interpolation=transforms.InterpolationMode.BILINEAR),
    partial(RandomResizedCropWithParams, size=TARGET_IMAGE_SIZE, scale=(0.2, 1.0), ratio=(1.0, 1.0)),
    ConvertToRGB
]

TRANSFORM_PIPELINES = [
    base_transforms,
    random_aug_transforms
]