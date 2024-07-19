import torchvision.transforms as transforms
import torch


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
        for t in self.transforms:
            if isinstance(t, RandomResizedCropWithParams):
                img = t(img)
                self.params['RandomResizedCrop'] = t.params
            elif isinstance(t, RandomHorizontalFlipWithParams):
                img = t(img)
                self.params['RandomHorizontalFlip'] = t.params
            elif isinstance(t, RandomVerticalFlipWithParams):
                img = t(img)
                self.params['RandomVerticalFlip'] = t.params
            elif isinstance(t, RandomRotationWithParams):
                img = t(img)
                self.params['RandomRotation'] = t.params
            else:
                img = t(img)
        return img


# transforms to prepare an image for the base miner.
base_transforms = transforms.Compose([
    CenterCrop(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.shape[0] == 1 else t),
])

# example data augmentation
random_image_transforms = ComposeWithParams([
    RandomResizedCropWithParams(256, scale=(0.2, 1.0), ratio=(1.0, 1.0)),
    RandomHorizontalFlipWithParams(),
    RandomVerticalFlipWithParams(),
    RandomRotationWithParams(20)
])
