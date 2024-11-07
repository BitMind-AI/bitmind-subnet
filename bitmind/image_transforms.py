from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2

from bitmind.constants import TARGET_IMAGE_SIZE


def CenterCrop():
    def fn(img):
        m = min(img.size)
        return transforms.CenterCrop(m)(img)

    return fn


class RandomResizedCropWithParams(transforms.RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = None

    #def get_params(self, img, scale, ratio):
    #    params = super().get_params(img, scale, ratio)
    #    self.params = params
    #    return params

    def __call__(self, img, scale=None, ratio=None):
        effective_scale = scale if scale is not None else self.scale
        effective_ratio = ratio if ratio is not None else self.ratio
        self.params = {'scale': scale, 'ratio': ratio}
        i, j, h, w = self.get_params(img, effective_scale, effective_ratio)
        return transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)


class RandomHorizontalFlipWithParams(transforms.RandomHorizontalFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = None

    def forward(self, img, do_flip=False):
        if torch.rand(1) < self.p or do_flip:
            self.params = {'do_flip': True}
            return transforms.functional.hflip(img)
        else:
            self.params = {'do_flip': False}
            return img


class RandomVerticalFlipWithParams(transforms.RandomVerticalFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = None

    def forward(self, img, do_flip=False):
        if torch.rand(1) < self.p or do_flip:
            self.params = {'do_flip': True}
            return transforms.functional.vflip(img)
        else:
            self.params = {'do_flip': False}
            return img


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


class CLAHE:
    def __call__(self, image, clip_limit=1.0, tile_grid_size=(8, 8)):
        # Convert PIL image to NumPy array
        image_np = np.array(image)

        # Create a CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # Apply CLAHE to each channel separately if it's a color image
        if len(image_np.shape) == 3:  # Color image
            channels = cv2.split(image_np)
            clahe_channels = [clahe.apply(ch) for ch in channels]
            clahe_image_np = cv2.merge(clahe_channels)
        else:  # Grayscale image
            clahe_image_np = clahe.apply(image_np)

        # Convert back to PIL image
        clahe_image = Image.fromarray(clahe_image_np)
        return clahe_image


class ComposeWithParams:
    def __init__(self, transforms):
        self.transforms = transforms
        self.params = {}

    def __call__(self, input_data):
        transform_params = {
            RandomResizedCropWithParams: 'RandomResizedCrop',
            RandomHorizontalFlipWithParams: 'RandomHorizontalFlip',
            RandomVerticalFlipWithParams: 'RandomVerticalFlip',
            RandomRotationWithParams: 'RandomRotation'
        }
        output_data = []
        list_input = True
        if not isinstance(input_data, list):
            input_data = [input_data]
            list_input = False

        for img in input_data:
            for t in self.transforms:
                if type(t) in transform_params and transform_params[type(t)] in self.params:
                    params = self.params[transform_params[type(t)]]
                    img = t(img, **params)
                else:
                    img = t(img)
                    if type(t) in transform_params:
                        self.params[transform_params[type(t)]] = t.params
            output_data.append(img)

        if list_input:
            return output_data
        return output_data[0]


# transforms to prepare an image for the base miner
base_transforms = ComposeWithParams([
    ConvertToRGB(),
    CenterCrop(),
    transforms.Resize(TARGET_IMAGE_SIZE),
    transforms.ToTensor()
])

# data augmentation
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
    CenterCrop(),
    transforms.Resize(TARGET_IMAGE_SIZE),
    CLAHE(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
