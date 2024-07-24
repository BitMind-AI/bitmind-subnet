import torchvision.transforms as transforms
import torch

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


class ComposeWithParams:
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

        for t in self.transforms:
            img = t(img)
            if type(t) in transform_params:
                self.params[transform_params[type(t)]] = t.params
        return img


# transforms to prepare an image for the base miner
base_transforms = transforms.Compose([
    CenterCrop(),
    transforms.Resize(TARGET_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.shape[0] == 1 else t),
])

# data augmentation
random_aug_transforms = ComposeWithParams([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.shape[0] == 1 else t),
    RandomRotationWithParams(20, interpolation=transforms.InterpolationMode.BILINEAR),
    RandomResizedCropWithParams(TARGET_IMAGE_SIZE, scale=(0.2, 1.0), ratio=(1.0, 1.0)),
    RandomHorizontalFlipWithParams(),
    RandomVerticalFlipWithParams()
])
