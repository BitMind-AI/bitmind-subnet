from PIL import Image
import torch

from bitmind.image_transforms import base_transforms


def predict(model: torch.nn.Module, image: Image.Image) -> float:
    """
    Perform prediction using a given PyTorch model on an image. You may need to modify this
    if you train a custom model.

    Args:
        model (torch.nn.Module): The PyTorch model to use for prediction.
        image (Image.Image): The input image as a PIL Image.

    Returns:
        float: The predicted output value.
    """
    image = base_transforms(image).unsqueeze(0).float()
    out = model(image).sigmoid().flatten().tolist()
    return out[0]