import math
import random
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import torch
import cv2

from bitmind.validator.config import TARGET_IMAGE_SIZE


def center_crop():
    def fn(img):
        m = min(img.size)
        return transforms.CenterCrop(m)(img)
    return fn


class RandomResizedCropWithParams(transforms.RandomResizedCrop):
    def __init__(self, *args, include_point=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = None
        self.include_point = include_point
        print(f"created RRC with point included: {self.include_point}")

    def forward(self, img, crop_params=None):
        """
        Args:
            img: PIL Image to be cropped and resized
            crop_params: Optional pre-computed crop parameters (i, j, h, w)
        """
        if crop_params is None:
            i, j, h, w = super().get_params(img, self.scale, self.ratio)
            if self.include_point is not None:
                x, y = self.include_point
                width, height = img.shape[1:]

                # adjust crop to keep mask point
                if x < j:
                    j = max(0, x - 10)
                elif x > j + w: 
                    j = min(width - w, x - w + 10)

                if y < i:
                    i = max(0, y - 10)
                elif y > i + h:
                    i = min(height - h, y - h + 10)
        else:
            i, j, h, w = crop_params

        self.params = {'crop_params': (i, j, h, w)}
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)


class RandomHorizontalFlipWithParams(transforms.RandomHorizontalFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = {}

    def forward(self, img, do_flip=None):
        if do_flip is not None:
            self.params = {'do_flip': do_flip}
            return transforms.functional.hflip(img) if do_flip else img
        elif not hasattr(self, 'params'):
            do_flip = torch.rand(1) < self.p
            self.params = {'do_flip': do_flip}
            return transforms.functional.hflip(img) if do_flip else img
        else:
            return transforms.functional.hflip(img) if self.params.get('do_flip', False) else img


class RandomVerticalFlipWithParams(transforms.RandomVerticalFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = {}

    def forward(self, img, do_flip=None):
        if do_flip is not None:
            self.params = {'do_flip': do_flip}
            return transforms.functional.vflip(img) if do_flip else img
        elif not hasattr(self, 'params'):
            do_flip = torch.rand(1) < self.p
            self.params = {'do_flip': do_flip}
            return transforms.functional.vflip(img) if do_flip else img
        else:
            return transforms.functional.vflip(img) if self.params.get('do_flip', False) else img


class RandomRotationWithParams(transforms.RandomRotation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = None

    def forward(self, img, angle=None):
        if angle is None:
            angle = self.get_params(self.degrees)
        self.params = {'angle': angle}
        return transforms.functional.rotate(img, angle)


class ConvertToRGB:
    def __call__(self, img):
        img = img.convert('RGB')
        return img


# DeeperForensics Distortion Functions
def get_distortion_parameter(distortion_type, level):
    """Get distortion parameter based on type and level.
    
    Parameters are arranged from least severe (level 1) to most severe (level 5).
    Each distortion type has different parameter behavior:
    
    CS (Color Saturation):
        - Range: [0.4 -> 0.0]
        - Lower values = worse distortion
        - 0.4 = slight desaturation
        - 0.0 = complete desaturation (grayscale)
    
    CC (Color Contrast):
        - Range: [0.85 -> 0.35]
        - Lower values = worse distortion
        - 0.85 = slight contrast reduction
        - 0.35 = severe contrast reduction
    
    BW (Block Wise):
        - Range: [16 -> 80]
        - Higher values = worse distortion
        - Controls number of random blocks added
        - 16 = few blocks
        - 80 = many blocks
    
    GNC (Gaussian Noise Color):
        - Range: [0.001 -> 0.05]
        - Higher values = worse distortion
        - Controls noise variance
        - 0.001 = subtle noise
        - 0.05 = very noisy
    
    GB (Gaussian Blur):
        - Range: [7 -> 21]
        - Higher values = worse distortion
        - Controls blur kernel size
        - 7 = slight blur
        - 21 = heavy blur
    
    JPEG (JPEG Compression):
        - Range: [2 -> 6]
        - Higher values = worse distortion
        - Controls downsampling factor
        - 2 = mild compression
        - 6 = severe compression
    """
    param_dict = {
        'CS': [0.4, 0.3, 0.2, 0.1, 0.0],
        'CC': [0.85, 0.725, 0.6, 0.475, 0.35],
        'BW': [16, 32, 48, 64, 80],
        'GNC': [0.001, 0.002, 0.005, 0.01, 0.05],
        'GB': [7, 9, 13, 17, 21],
        'JPEG': [2, 3, 4, 5, 6]
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

    def __call__(self, img, level=None):
        if level is None:
            self.level = random.randint(self.level_min, self.level_max)
        else:
            self.level = level

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

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        
    def __call__(self, image):
        # Convert PIL image to NumPy array
        image_np = np.array(image)
        
        # Apply CLAHE to each channel separately if it's a color image
        if len(image_np.shape) == 3:  # Color image
            channels = cv2.split(image_np)
            clahe_channels = [self.clahe.apply(ch) for ch in channels]
            clahe_image_np = cv2.merge(clahe_channels)
        else:  # Grayscale image
            clahe_image_np = self.clahe.apply(image_np)

        # Convert back to PIL image
        clahe_image = Image.fromarray(clahe_image_np)

        return clahe_image
    
    
class TensorCLAHE:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        
    def __call__(self, tensor):
        # Convert tensor to numpy array (H,W,C) format
        img_np = tensor.permute(1, 2, 0).numpy() * 255
        img_np = img_np.astype(np.uint8)
        
        # Apply CLAHE to each channel
        channels = cv2.split(img_np)
        clahe_channels = [self.clahe.apply(ch) for ch in channels]
        clahe_image_np = cv2.merge(clahe_channels)
        
        # Convert back to tensor
        tensor = torch.from_numpy(clahe_image_np).float() / 255.0
        return tensor.permute(2, 0, 1)


class ComposeWithParams:
    def __init__(self, transforms):
        self.transforms = transforms
        self.params = {}

    def __call__(self, input_data, clear_params=True):
        if clear_params:
            self.params = {}

        output_data = []
        list_input = True
        if not isinstance(input_data, list):
            input_data = [input_data]
            list_input = False

        for img in input_data:
            for transform in self.transforms:
                try:
                    name = transform.__name__
                except AttributeError:
                    name = transform.__class__.__name__

                if name in self.params:
                    img = transform(img, **self.params[name])
                else:
                    img = transform(img)
                    if hasattr(transform, 'params'):
                        self.params[name] = transform.params
            output_data.append(img)

        if list_input:
            return output_data
        return output_data[0]


# Transform configurations
def get_base_transforms(target_image_size=TARGET_IMAGE_SIZE):
    return ComposeWithParams([
        ConvertToRGB(),
        center_crop(),
        transforms.Resize(target_image_size),
        transforms.ToTensor()
    ])


def get_random_augmentations(target_image_size=TARGET_IMAGE_SIZE, mask_point=None):
    return ComposeWithParams([
        ConvertToRGB(),
        transforms.ToTensor(),
        RandomRotationWithParams(20, interpolation=transforms.InterpolationMode.BILINEAR),
        RandomResizedCropWithParams(
            TARGET_IMAGE_SIZE, scale=(0.2, 1.0), ratio=(1.0, 1.0), include_point=mask_point),
        RandomHorizontalFlipWithParams(),
        RandomVerticalFlipWithParams()
    ])

def get_ucf_base_transforms(target_image_size=TARGET_IMAGE_SIZE):
    return transforms.Compose([
        ConvertToRGB(),
        center_crop(),
        transforms.Resize(target_image_size),
        CLAHE(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_tall_base_transforms(target_image_size=TARGET_IMAGE_SIZE):
    return ComposeWithParams([
        transforms.Resize(target_image_size),
        transforms.ToTensor()
    ])

# Medium difficulty transforms with mild distortions
def get_random_augmentations_medium(target_image_size=TARGET_IMAGE_SIZE, mask_point=None):
    return ComposeWithParams([
        ConvertToRGB(),
        transforms.ToTensor(),
        RandomRotationWithParams(20, interpolation=transforms.InterpolationMode.BILINEAR),
        RandomResizedCropWithParams(
            TARGET_IMAGE_SIZE, scale=(0.2, 1.0), ratio=(1.0, 1.0), include_point=mask_point),
        RandomHorizontalFlipWithParams(),
        RandomVerticalFlipWithParams(),
        ApplyDeeperForensicsDistortion('CS', level_min=0, level_max=1),
        ApplyDeeperForensicsDistortion('CC', level_min=0, level_max=1),
        ApplyDeeperForensicsDistortion('JPEG', level_min=0, level_max=1)
    ])

# Hard difficulty transforms with more severe distortions
def get_random_augmentations_hard(target_image_size=TARGET_IMAGE_SIZE, mask_point=None):
    return ComposeWithParams([
        ConvertToRGB(),
        transforms.ToTensor(), 
        RandomRotationWithParams(20, interpolation=transforms.InterpolationMode.BILINEAR),
        RandomResizedCropWithParams(
            TARGET_IMAGE_SIZE, scale=(0.2, 1.0), ratio=(1.0, 1.0), include_point=mask_point),
        RandomHorizontalFlipWithParams(),
        RandomVerticalFlipWithParams(),
        ApplyDeeperForensicsDistortion('CS', level_min=0, level_max=2),
        ApplyDeeperForensicsDistortion('CC', level_min=0, level_max=2), 
        ApplyDeeperForensicsDistortion('JPEG', level_min=0, level_max=2),
        ApplyDeeperForensicsDistortion('GNC', level_min=0, level_max=2),
        ApplyDeeperForensicsDistortion('GB', level_min=0, level_max=2)
    ])


def apply_augmentation_by_level(
    image, 
    target_image_size, 
    mask_point=None, 
    level_probs={
        0: 0.25,  # No augmentations (base transforms)
        1: 0.45,  # Basic augmentations
        2: 0.15,  # Medium distortions
        3: 0.15   # Hard distortions
    }):
    """
    Apply image transformations based on randomly selected level.
    
    Args:
        image: PIL Image to transform
        level_probs: dict with augmentation levels and their probabilities.
            Default probabilities:
            - Level 0 (25%): No augmentations (base transforms)
            - Level 1 (45%): Basic augmentations  
            - Level 2 (15%): Medium distortions
            - Level 3 (15%): Hard distortions
        
    Returns:
        tuple: (transformed_image, level, transform_params)
        
    Raises:
        ValueError: If probabilities don't sum to 1.0 (within floating point precision)
    """
    # Validate probabilities
    if not math.isclose(sum(level_probs.values()), 1.0, rel_tol=1e-9):
        raise ValueError("Probabilities of levels must sum to 1.0")
    
    # Calculate cumulative probabilities
    cumulative_probs = {}
    cumsum = 0
    for level, prob in sorted(level_probs.items()):
        cumsum += prob
        cumulative_probs[level] = cumsum
    
    # Select augmentation level
    rand_val = np.random.random()
    for curr_level, cum_prob in cumulative_probs.items():
        if rand_val <= cum_prob:
            level = curr_level
            break

    # Apply appropriate transform
    if level == 0:
        tforms = get_base_transforms(target_image_size)
    elif level == 1:
        tforms = get_random_augmentations(target_image_size, mask_point)
    elif level == 2:
        tforms = get_random_augmentations_medium(target_image_size, mask_point)
    else:  # level == 3
        tforms = get_random_augmentations_hard(target_image_size, mask_point)

    transformed = tforms(image)
        
    return transformed, level, tforms.params
