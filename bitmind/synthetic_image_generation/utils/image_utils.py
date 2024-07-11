import PIL
import torch
import numpy as np

def resize_image(image: PIL.Image.Image, max_width: int, max_height: int) -> PIL.Image.Image:
    """Resize the image to fit within specified dimensions while maintaining aspect ratio."""
    original_width, original_height = image.size

    # Calculate the aspect ratio and determine new dimensions
    aspect_ratio = original_width / original_height
    new_width = min(max_width, original_width)
    new_height = int(new_width / aspect_ratio)

    if new_height > max_height:
        new_height = max_height
        new_width = int(new_height * aspect_ratio)

    # Resize the image using the high-quality LANCZOS filter
    resized_image = image.resize((new_width, new_height), PIL.Image.LANCZOS)
    return resized_image

def normalize_image(image : PIL.Image.Image):
    """
    Normalize a PIL Image to the range [-1, 1].
    
    Parameters:
    image (PIL.Image.Image): The input image.
    
    Returns:
    np.ndarray: The normalized image as a NumPy array.
    """
    # Convert image to NumPy array
    image_array = np.array(image).astype(np.float32)
    
    # Normalize to [0, 1]
    image_array /= 255.0
    
    # Normalize to [-1, 1]
    normalized_array = (image_array * 2.0) - 1.0
    return normalized_array

def convert_image_to_tensor(image : PIL.Image.Image):
    """
    Convert a normalized RGB image to a PyTorch tensor with shape (1, 3, w, h).
    
    Parameters:
    image (PIL.Image.Image): The input PIL image.
    
    Returns:
    torch.Tensor: The image as a PyTorch tensor with shape (1, 3, 612, 612).
    """
    # Normalize the image to [-1, 1]
    normalized_image = normalize_image(image)
    
    tensor = torch.tensor(normalized_image)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)
    
    return tensor

def image_pair_to_resized_tensor(image1 : PIL.Image.Image, image2 : PIL.Image.Image):
    """
    Preprocess PIL image pair for LPIP evaluation by resizing to
    the smallest image dimensions and converting to tensor.
    
    Parameters:
    image (PIL.Image.Image): The input PIL image.
    
    Returns:
    torch.Tensor: The image as a PyTorch tensor with shape (1, 3, 612, 612).
    """
    width1, height1 = image1.size
    width2, height2 = image2.size
    
    # Determine the dimensions of the smallest image
    new_width = min(width1, width2)
    new_height = min(height1, height2)
    
    # Resize both images to the smallest dimensions while maintaining aspect ratio
    image1_resized = image1.resize((new_width, new_height), PIL.Image.LANCZOS)
    image2_resized = image2.resize((new_width, new_height), PIL.Image.LANCZOS)
    return convert_image_to_tensor(image1_resized), convert_image_to_tensor(image2_resized)