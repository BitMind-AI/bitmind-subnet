import math
import random
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2

from bitmind.constants import TARGET_IMAGE_SIZE


def center_crop():
    def fn(img):
        m = min(img.size)
        return transforms.CenterCrop(m)(img)

    return fn


class RandomResizedCropWithParams(transforms.RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = None

    def get_params(self, img, scale, ratio):
        params = super().get_params(img, scale, ratio)
        self.params = params
        return params


class RandomHorizontalFlipWithParams(transforms.RandomHorizontalFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = None

    def forward(self, img):
        if torch.rand(1) < self.p:
            self.params = True
            return transforms.functional.hflip(img)
        else:
            self.params = False
            return img


class RandomVerticalFlipWithParams(transforms.RandomVerticalFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = None

    def forward(self, img):
        if torch.rand(1) < self.p:
            self.params = True
            return transforms.functional.vflip(img)
        else:
            self.params = False
            return img


class RandomRotationWithParams(transforms.RandomRotation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = None

    def forward(self, img):
        angle = self.get_params(self.degrees)
        self.params = angle
        return transforms.functional.rotate(img, angle)


class ConvertToRGB:
    def __call__(self, img):
        img = img.convert('RGB')
        return img


# DeeperForensics Distortion Functions

def get_distortion_parameter(distortion_type, level):
    """Get distortion parameter based on type and level."""
    param_dict = {
        'CS': [0.4, 0.3, 0.2, 0.1, 0.0],  # smaller, worse
        'CC': [0.85, 0.725, 0.6, 0.475, 0.35],  # smaller, worse
        'BW': [16, 32, 48, 64, 80],  # larger, worse
        'GNC': [0.001, 0.002, 0.005, 0.01, 0.05],  # larger, worse
        'GB': [7, 9, 13, 17, 21],  # larger, worse
        'JPEG': [2, 3, 4, 5, 6]  # larger, worse
    }
    return param_dict[distortion_type][level - 1]


def get_distortion_function(distortion_type):
    """Get distortion function based on type."""
    func_dict = {
        'CS': color_saturation,
        'CC': color_contrast,
        'BW': block_wise,
        'GNC': gaussian_noise_color,
        'GB': gaussian_blur,
        'JPEG': jpeg_compression
    }
    return func_dict[distortion_type]


def rgb_to_bgr(tensor_img):
    """Convert a PyTorch tensor image from RGB to BGR format.
    
    Args:
        tensor_img: Tensor in format (C, H, W)
    """
    if tensor_img.shape[0] == 3:
        tensor_img = tensor_img[[2, 1, 0], ...]
    return tensor_img


def bgr_to_rgb(tensor_img):
    """Convert a PyTorch tensor image from BGR to RGB format.
    
    Args:
        tensor_img: Tensor in format (C, H, W) with values in [0, 1]
    """
    if tensor_img.shape[0] == 3:
        tensor_img = tensor_img[[2, 1, 0], ...]
    return tensor_img


def bgr2ycbcr(img_bgr):
    """Convert BGR image to YCbCr color space."""
    img_bgr = img_bgr.astype(np.float32)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    img_ycbcr = img_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0
    return img_ycbcr


def ycbcr2bgr(img_ycbcr):
    """Convert YCbCr image to BGR color space."""
    img_ycbcr = img_ycbcr.astype(np.float32)
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)
    img_ycrcb = img_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    img_bgr = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img_bgr


def color_saturation(img, param):
    """Apply color saturation distortion."""
    ycbcr = bgr2ycbcr(img)
    ycbcr[:, :, 1] = 0.5 + (ycbcr[:, :, 1] - 0.5) * param
    ycbcr[:, :, 2] = 0.5 + (ycbcr[:, :, 2] - 0.5) * param
    img = ycbcr2bgr(ycbcr).astype(np.uint8)
    return img


def color_contrast(img, param):
    """Apply color contrast distortion."""
    img = img.astype(np.float32) * param
    return img.astype(np.uint8)


def block_wise(img, param):
    """Apply block-wise distortion."""
    width = 8
    block = np.ones((width, width, 3)).astype(int) * 128
    param = min(img.shape[0], img.shape[1]) // 256 * param
    for _ in range(param):
        r_w = random.randint(0, img.shape[1] - 1 - width)
        r_h = random.randint(0, img.shape[0] - 1 - width)
        img[r_h:r_h + width, r_w:r_w + width, :] = block
    return img


def gaussian_noise_color(img, param):
    """Apply colored Gaussian noise."""
    ycbcr = bgr2ycbcr(img) / 255
    size_a = ycbcr.shape
    b = (ycbcr + math.sqrt(param) * np.random.randn(size_a[0], size_a[1], size_a[2])) * 255
    b = ycbcr2bgr(b)
    return np.clip(b, 0, 255).astype(np.uint8)


def gaussian_blur(img, param):
    """Apply Gaussian blur."""
    return cv2.GaussianBlur(img, (param, param), param * 1.0 / 6)


def jpeg_compression(img, param):
    """Apply JPEG compression distortion."""
    h, w, _ = img.shape
    s_h = h // param
    s_w = w // param
    img = cv2.resize(img, (s_w, s_h))
    return cv2.resize(img, (w, h))


class ApplyDeeperForensicsDistortion:
    """Wrapper for applying DeeperForensics distortions."""
    
    def __init__(self, distortion_type, level_min=0, level_max=3):
        self.distortion_type = distortion_type
        self.level_min = level_min
        self.level_max = level_max

    def __call__(self, img):
        self.level = random.randint(self.level_min, self.level_max)
        if self.level > 0:
            self.distortion_param = get_distortion_parameter(self.distortion_type, self.level)
            self.distortion_func = get_distortion_function(self.distortion_type)
        else:
            self.distortion_func = None
            self.distortion_param = None

        if not self.distortion_func:
            return img

        if isinstance(img, torch.Tensor):
            img = rgb_to_bgr(img)
            img = img.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)

        img = self.distortion_func(img, self.distortion_param)

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.astype(np.float32) / 255.0)
            img = img.permute(2, 0, 1)
            img = bgr_to_rgb(img)

        return img


class CLAHE:
    """Contrast Limited Adaptive Histogram Equalization."""
    
    def __call__(self, image, clip_limit=1.0, tile_grid_size=(8, 8)):
        image_np = np.array(image)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        if len(image_np.shape) == 3:
            channels = cv2.split(image_np)
            clahe_channels = [clahe.apply(ch) for ch in channels]
            clahe_image_np = cv2.merge(clahe_channels)
        else:
            clahe_image_np = clahe.apply(image_np)

        return Image.fromarray(clahe_image_np)


class ComposeWithParams:
    """Compose multiple transforms with parameter tracking."""
    
    def __init__(self, transforms):
        self.transforms = transforms
        self.params = {}

    def __call__(self, img):
        transform_params = {
            RandomResizedCropWithParams: 'RandomResizedCrop',
            RandomHorizontalFlipWithParams: 'RandomHorizontalFlip',
            RandomVerticalFlipWithParams: 'RandomVerticalFlip',
            RandomRotationWithParams: 'RandomRotation'
        }

        for transform in self.transforms:
            img = transform(img)
            if type(transform) in transform_params:
                self.params[transform_params[type(transform)]] = transform.params
        return img


# Transform configurations
base_transforms = transforms.Compose([
    ConvertToRGB(),
    center_crop(),
    transforms.Resize(TARGET_IMAGE_SIZE),
    transforms.ToTensor()
])

random_aug_transforms = ComposeWithParams([
    ConvertToRGB(),
    transforms.ToTensor(),
    RandomRotationWithParams(20, interpolation=transforms.InterpolationMode.BILINEAR),
    RandomResizedCropWithParams(TARGET_IMAGE_SIZE, scale=(0.2, 1.0), ratio=(1.0, 1.0)),
    RandomHorizontalFlipWithParams(),
    RandomVerticalFlipWithParams()
])

ucf_transforms = transforms.Compose([
    ConvertToRGB(),
    center_crop(),
    transforms.Resize(TARGET_IMAGE_SIZE),
    CLAHE(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Medium difficulty transforms with mild distortions
random_aug_transforms_medium = ComposeWithParams([
    ConvertToRGB(),
    transforms.ToTensor(),
    RandomRotationWithParams(20, interpolation=transforms.InterpolationMode.BILINEAR),
    RandomResizedCropWithParams(TARGET_IMAGE_SIZE, scale=(0.2, 1.0), ratio=(1.0, 1.0)),
    RandomHorizontalFlipWithParams(),
    RandomVerticalFlipWithParams(),
    ApplyDeeperForensicsDistortion('CS', level_min=1, level_max=1),
    ApplyDeeperForensicsDistortion('CC', level_min=1, level_max=1),
    ApplyDeeperForensicsDistortion('JPEG', level_min=1, level_max=1)
])

# Hard difficulty transforms with more severe distortions
random_aug_transforms_hard = ComposeWithParams([
    ConvertToRGB(),
    transforms.ToTensor(), 
    RandomRotationWithParams(20, interpolation=transforms.InterpolationMode.BILINEAR),
    RandomResizedCropWithParams(TARGET_IMAGE_SIZE, scale=(0.2, 1.0), ratio=(1.0, 1.0)),
    RandomHorizontalFlipWithParams(),
    RandomVerticalFlipWithParams(),
    ApplyDeeperForensicsDistortion('CS', level_min=1, level_max=2),
    ApplyDeeperForensicsDistortion('CC', level_min=1, level_max=2), 
    ApplyDeeperForensicsDistortion('JPEG', level_min=1, level_max=2),
    ApplyDeeperForensicsDistortion('GNC', level_min=1, level_max=2),
    ApplyDeeperForensicsDistortion('GB', level_min=1, level_max=2)
])