import torchvision.transforms as transforms


def CenterCrop():
    def fn(img):
        m = min(img.size)
        return transforms.CenterCrop(m)(img)

    return fn


# transforms to prepare an image for the base miner.
base_transforms = transforms.Compose([
    CenterCrop(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.shape[0] == 1 else t),
])

# example data augmentation
random_image_transforms = transforms.Compose([
    transforms.RandomResizedCrop(256), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), 
    transforms.RandomRotation(20)
])
