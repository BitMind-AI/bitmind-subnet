import math
import random
from scipy import ndimage
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import cv2


def apply_random_augmentations(
    inputs, target_image_size, mask_point=None, level_probs=None
):
    """
    Apply image transformations based on randomly selected level.

    Args:
        image: image, video, or tuple of videos to transform
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
    if level_probs is None:
        level_probs = {  # Difficult level probabilities
            0: 0.25,  # No augmentations (base transforms)
            1: 0.25,  # Basic augmentations
            2: 0.25,  # Medium distortions
            3: 0.25,  # Hard distortions
        }

    if not math.isclose(sum(level_probs.values()), 1.0, rel_tol=1e-9):
        raise ValueError("Probabilities of levels must sum to 1.0")

    # get cumulative probs and select augmentation level
    cumulative_probs = {}
    cumsum = 0
    for level, prob in sorted(level_probs.items()):
        cumsum += prob
        cumulative_probs[level] = cumsum

    rand_val = np.random.random()
    for curr_level, cum_prob in cumulative_probs.items():
        if rand_val <= cum_prob:
            level = curr_level
            break

    if level == 0:
        tforms = get_base_transforms(target_image_size)
    elif level == 1:
        tforms = get_random_augmentations(target_image_size, mask_point)
    elif level == 2:
        tforms = get_random_augmentations_medium(target_image_size, mask_point)
    else:  # level == 3
        tforms = get_random_augmentations_hard(target_image_size, mask_point)

    if isinstance(inputs, tuple):
        transformed_A = tforms(inputs[0], clear_params=False)
        transformed_B = tforms(inputs[1])
        transformed = np.concatenate([transformed_A, transformed_B], axis=0)
    else:
        transformed = tforms(inputs)

    return transformed, level, tforms.params


def get_base_transforms(target_image_size):
    return ComposeWithParams(
        [
            CenterCrop(),
            Resize(),
        ]
    )


def get_random_augmentations(target_image_size, mask_point=None):
    """Basic augmentations with geometric transforms"""
    base_augmentations = [
        RandomRotationWithParams(degrees=20, order=2),
        RandomResizedCropWithParams(
            target_image_size, scale=(0.2, 1.0), include_point=mask_point
        ),
        RandomHorizontalFlipWithParams(),
        RandomVerticalFlipWithParams(),
    ]
    return ComposeWithParams(base_augmentations)


def get_random_augmentations_medium(target_image_size, mask_point=None):
    """Medium difficulty transforms with mild distortions"""
    base_augmentations = get_random_augmentations(target_image_size, mask_point)

    distortions = [
        ApplyDeeperForensicsDistortion("CS", level_min=0, level_max=1),
        ApplyDeeperForensicsDistortion("CC", level_min=0, level_max=1),
        ApplyDeeperForensicsDistortion("JPEG", level_min=0, level_max=1),
    ]

    return ComposeWithParams(base_augmentations.transforms + distortions)


def get_random_augmentations_hard(target_image_size, mask_point=None):
    """Hard difficulty transforms with more severe distortions"""
    base_augmentations = get_random_augmentations(target_image_size, mask_point)

    distortions = [
        ApplyDeeperForensicsDistortion("CS", level_min=0, level_max=2),
        ApplyDeeperForensicsDistortion("CC", level_min=0, level_max=2),
        ApplyDeeperForensicsDistortion("JPEG", level_min=0, level_max=2),
        ApplyDeeperForensicsDistortion("GNC", level_min=0, level_max=2),
        ApplyDeeperForensicsDistortion("GB", level_min=0, level_max=2),
    ]

    return ComposeWithParams(base_augmentations.transforms + distortions)


class ComposeWithParams:
    def __init__(self, transforms):
        self.transforms = transforms
        self.params = {}

    def __call__(self, input_data, clear_params=True):
        if clear_params:
            self.params = {}

        is_single_image = input_data.ndim == 3  # (H, W, C)
        if is_single_image:
            input_data = input_data[None, ...]  # Add fake temporal dim → (1, H, W, C)

        output_frames = []
        for frame in input_data:
            for transform in self.transforms:
                name = getattr(transform, "__name__", transform.__class__.__name__)
                if name in self.params:
                    frame = transform(frame, **self.params[name])
                else:
                    frame = transform(frame)
                    if hasattr(transform, "params"):
                        self.params[name] = transform.params
            output_frames.append(frame)

        output = np.stack(output_frames)
        return output[0] if is_single_image else output


class ApplyDeeperForensicsDistortion:
    """Wrapper for applying DeeperForensics distortions."""

    def __init__(self, distortion_type, level_min=0, level_max=3):
        self.__name__ = distortion_type
        self.distortion_type = distortion_type
        self.level = None
        self.level_min = level_min
        self.level_max = level_max
        self.params = {}  # level
        self.distortion_params = {}  # distortion_type specific

    def __call__(self, img, level=None):
        if level is None and self.level is None:
            self.level = random.randint(self.level_min, self.level_max)
            self.params = {"level": self.level}
        elif self.level is None:
            self.level = level
            self.params = {"level": self.level}

        if self.level > 0:
            self.distortion_func = get_distortion_function(self.distortion_type)
            if len(self.distortion_params) == 0:
                self.distortion_param = get_distortion_parameter(
                    self.distortion_type, self.level
                )
                self.distortion_params = {"param": self.distortion_param}
        else:
            return img

        output = self.distortion_func(img, **self.distortion_params)
        if isinstance(output, tuple):
            self.distortion_params.update(output[1])
            return output[0]
        else:
            return output


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
        "CS": [0.4, 0.3, 0.2, 0.1, 0.0],
        "CC": [0.85, 0.725, 0.6, 0.475, 0.35],
        "BW": [16, 32, 48, 64, 80],
        "GNC": [0.001, 0.002, 0.005, 0.01, 0.05],
        "GB": [7, 9, 13, 17, 21],
        "JPEG": [2, 3, 4, 5, 6],
    }
    return param_dict[distortion_type][level - 1]


def get_distortion_function(distortion_type):
    """Get distortion function based on type."""
    func_dict = {
        "CS": color_saturation,
        "CC": color_contrast,
        "BW": block_wise,
        "GNC": gaussian_noise_color,
        "GB": gaussian_blur,
        "JPEG": jpeg_compression,
    }
    return func_dict[distortion_type]


def Resize():
    def resize(img):
        return cv2.resize(img, (256, 256))

    return resize


def rgb2ycbcr(img_rgb):
    """Convert RGB image to YCbCr color space.

    Args:
        img_rgb (np.ndarray): RGB image array of shape (H, W, 3)

    Returns:
        np.ndarray: YCbCr image array of shape (H, W, 3) with values normalized to [0,1]
    """
    img_rgb = img_rgb.astype(np.float32)
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCR_CB)
    img_ycbcr = img_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0
    return img_ycbcr


def ycbcr2rgb(img_ycbcr):
    """Convert YCbCr image to RGB color space.

    Args:
        img_ycbcr (np.ndarray): YCbCr image array of shape (H, W, 3)

    Returns:
        np.ndarray: RGB image array of shape (H, W, 3) with values in [0,255]
    """
    img_ycbcr = img_ycbcr.astype(np.float32)
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)
    img_ycrcb = img_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    img_rgb = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2RGB)
    return img_rgb


def color_saturation(img, param):
    """Apply color saturation distortion.

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (float): Saturation multiplier parameter

    Returns:
        np.ndarray: Distorted RGB image array with modified saturation
    """
    ycbcr = rgb2ycbcr(img)
    ycbcr[:, :, 1] = 0.5 + (ycbcr[:, :, 1] - 0.5) * param
    ycbcr[:, :, 2] = 0.5 + (ycbcr[:, :, 2] - 0.5) * param
    img = ycbcr2rgb(ycbcr).astype(np.uint8)
    return img


def color_contrast(img, param):
    """Apply color contrast distortion.

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (float): Contrast multiplier parameter

    Returns:
        np.ndarray: Distorted RGB image array with modified contrast
    """
    img = img.astype(np.float32) * param
    return img.astype(np.uint8)


def block_wise(img, param):
    """Apply block-wise distortion by adding random gray blocks.

    NOTE: CURRENTLY NOT USED

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (int): Number of blocks to add, scaled by image dimensions

    Returns:
        np.ndarray: Distorted RGB image array with added gray blocks
    """
    width = 8
    block = np.ones((width, width, 3)).astype(int) * 128
    param = min(img.shape[0], img.shape[1]) // 256 * param
    for _ in range(param):
        r_w = random.randint(0, img.shape[1] - 1 - width)
        r_h = random.randint(0, img.shape[0] - 1 - width)
        img[r_h : r_h + width, r_w : r_w + width, :] = block
    return img


def gaussian_noise_color(img, param, b=None):
    """Apply colored Gaussian noise in YCbCr color space.

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (float): Variance of the Gaussian noise

    Returns:
        np.ndarray: Distorted RGB image array with added color noise
    """
    ycbcr = rgb2ycbcr(img) / 255
    size_a = ycbcr.shape
    if b is None:
        b = (
            ycbcr + math.sqrt(param) * np.random.randn(size_a[0], size_a[1], size_a[2])
        ) * 255
        b = ycbcr2rgb(b)
    return np.clip(b, 0, 255).astype(np.uint8), {"b": b}


def gaussian_blur(img, param):
    """Apply Gaussian blur with specified kernel size.

    Args
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (int): Gaussian kernel size (must be odd)

    Returns:
        np.ndarray: Blurred RGB image array
    """
    return cv2.GaussianBlur(img, (param, param), param * 1.0 / 6)


def jpeg_compression(img, param):
    """Apply JPEG compression-like distortion through downsampling.

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)
        param (int): Downsampling factor

    Returns:
        np.ndarray: Distorted RGB image array with compression artifacts
    """
    h, w, _ = img.shape
    s_h = h // param
    s_w = w // param
    img = cv2.resize(img, (s_w, s_h))
    return cv2.resize(img, (w, h))


def CenterCrop():
    """Center crop an image to a square.

    Args:
        img (np.ndarray): Input RGB image array of shape (H, W, 3)

    Returns:
        np.ndarray: Center cropped RGB image array with equal height and width
    """

    def crop(img):
        h, w = img.shape[:2]
        m = min(h, w)
        i = (h - m) // 2
        j = (w - m) // 2
        return img[i : i + m, j : j + m]

    return crop


class RandomResizedCropWithParams:
    """Randomly crop and resize an image while optionally preserving a point.

    Args:
        scale (tuple): Range of size of the origin size cropped
        size (int or tuple): Target output size
        include_point (tuple, optional): (x,y) point coordinates that must be preserved in crop
    """

    def __init__(self, size, scale, include_point=None):
        self.params = None
        self.scale = scale
        self.size = size
        self.include_point = include_point

    def __call__(self, img, crop_params=None):
        """Perform random resized crop transform.

        Args:
            img (np.ndarray): Input RGB image array of shape (H, W, C)
            crop_params (tuple, optional): Pre-computed crop parameters (i, j, h, w)
                where image will be cropped to [i:i+h, j:j+w] before resizing

        Returns:
            np.ndarray: Randomly cropped and resized RGB image array
        """
        # Convert numpy array to shape expected by parent class
        height, width = img.shape[:2]

        if crop_params is None:
            area = height * width
            target_area = area * np.random.uniform(*self.scale)
            h = w = int(round(np.sqrt(target_area)))
            h = min(h, height)
            w = min(w, width)
            i = np.random.randint(0, height - h + 1)
            j = np.random.randint(0, width - w + 1)
            if self.include_point is not None:
                x, y = self.include_point

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

        self.params = {"crop_params": (i, j, h, w)}
        cropped = img[i : i + h, j : j + w, :]

        if isinstance(self.size, int):
            size = (self.size, self.size)
        else:
            size = self.size
        resized = cv2.resize(cropped, size, interpolation=cv2.INTER_LINEAR)
        return resized


class RandomHorizontalFlipWithParams:
    """Randomly flip an image horizontally.

    Args:
        p (float): Probability of flipping the image
    """

    def __init__(self, p=0.5):
        self.p = p
        self.params = {}

    def __call__(self, img, flip=None):
        """Perform horizontal flip transform.

        Args:
            img (np.ndarray): Input RGB image array of shape (H, W, C)
            flip (bool, optional): Pre-computed flip decision

        Returns:
            np.ndarray: Horizontally flipped RGB image array if flip is True
        """
        if flip is not None:
            self.params = {"flip": flip}
            return np.fliplr(img) if flip else img
        elif not hasattr(self, "params") or len(self.params) == 0:
            flip = np.random.random() < self.p
            self.params = {"flip": flip}
            return np.fliplr(img) if flip else img
        else:
            return np.fliplr(img) if self.params.get("flip", False) else img


class RandomVerticalFlipWithParams:
    """Randomly flip an image vertically.

    Args:
        p (float): Probability of flipping the image
    """

    def __init__(self, p=0.5):
        self.p = p
        self.params = {}

    def __call__(self, img, flip=None):
        """Perform vertical flip transform.

        Args:
            img (np.ndarray): Input RGB image array of shape (H, W, C)
            flip (bool, optional): Pre-computed flip decision

        Returns:
            np.ndarray: Vertically flipped RGB image array if flip is True
        """
        if flip is not None:
            self.params = {"flip": flip}
            return np.flipud(img) if flip else img
        elif not hasattr(self, "params") or len(self.params) == 0:
            flip = np.random.random() < self.p
            self.params = {"flip": flip}
            return np.flipud(img) if flip else img
        else:
            return np.flipud(img) if self.params.get("flip", False) else img


class RandomRotationWithParams:
    """Randomly rotate an image.

    Args:
        degrees (float or tuple): Range of degrees to select from. If float, uses (-degrees, degrees)
        p (float): Probability of rotating the image
        reshape (bool): If True, expands output image to fit rotated image
        mode (str): How to fill the border ('reflect', 'constant', etc)
        order (int): Interpolation order (0-5)
    """

    def __init__(self, degrees, p=0.5, reshape=False, mode="reflect", order=2):
        if isinstance(degrees, (tuple, list)):
            self.degrees = degrees
        else:
            self.degrees = (-degrees, degrees)
        self.p = p
        self.params = None
        self.reshape = reshape
        self.mode = mode
        self.order = order

    def __call__(self, img, rotate=None, angle=None, order=None):
        """Perform rotation transform.

        Args:
            img (np.ndarray): Input RGB image array of shape (H, W, C)
            rotate (bool, optional): Pre-computed rotation decision
            angle (float, optional): Pre-computed rotation angle
            order (int, optional): Pre-computed interpolation order

        Returns:
            np.ndarray: Rotated RGB image array
        """
        if rotate is None:
            rotate = np.random.random() < self.p
            self.params = {"rotate": rotate}

        if not rotate:
            return img

        order = self.order if order is None else order
        if isinstance(order, (tuple, list)):
            order = random.randint(order[0], order[1])

        if angle is None:
            angle = random.uniform(self.degrees[0], self.degrees[1])

        self.params.update({"order": order, "angle": angle})
        return ndimage.rotate(
            img, angle, reshape=self.reshape, mode=self.mode, order=order, axes=(0, 1)
        )
