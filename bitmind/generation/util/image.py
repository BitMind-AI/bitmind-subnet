import numpy as np
import PIL
import os
from PIL import Image, ImageDraw
from typing import Tuple, Union, List


def resize_image(
    image: PIL.Image.Image, max_width: int, max_height: int
) -> PIL.Image.Image:
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


def resize_images_in_directory(directory, target_width, target_height):
    """
    Resize all images in the specified directory to the target width and height.

    Args:
    directory (str): Path to the directory containing images.
    target_width (int): Target width for resizing the images.
    target_height (int): Target height for resizing the images.
    """
    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".gif")
        ):  # Check for image file extensions
            filepath = os.path.join(directory, filename)
            with PIL.Image.open(filepath) as img:
                # Resize the image and save back to the file location
                resized_img = resize_image(
                    img, max_width=target_width, max_height=target_height
                )
                resized_img.save(filepath)


def save_images_to_disk(
    image_dataset, start_index, num_images, save_directory, resize=True
):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for i in range(start_index, start_index + num_images):
        try:
            image_data = image_dataset[i]  # Retrieve image using the __getitem__ method
            image = image_data["image"]  # Extract the image
            image_id = image_data["id"]  # Extract the image ID
            file_path = os.path.join(
                save_directory, f"{image_id}.jpg"
            )  # Construct file path
            # if resize:
            #    image = resize_image(image, TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1])
            image.save(file_path, "JPEG")  # Save the image
            print(f"Saved: {file_path}")
        except Exception as e:
            print(f"Failed to save image {i}: {e}")


def create_random_mask(size: Tuple[int, int]) -> Image.Image:
    """
    Create a random mask for i2i transformation.
    """
    w, h = size
    mask = Image.new("RGB", size, "black")

    if np.random.rand() < 0.5:
        # Rectangular mask with smoother edges
        width = np.random.randint(w // 4, w // 2)
        height = np.random.randint(h // 4, h // 2)

        # Center the rectangle with some random offset
        x = (w - width) // 2 + np.random.randint(-width // 4, width // 4)
        y = (h - height) // 2 + np.random.randint(-height // 4, height // 4)

        # Create mask with PIL draw for smoother edges
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle(
            [x, y, x + width, y + height],
            radius=min(width, height) // 10,  # Smooth corners
            fill="white",
        )
    else:
        # Circular mask with feathered edges
        draw = ImageDraw.Draw(mask)
        x = w // 2
        y = h // 2

        # Make radius proportional to image size
        radius = min(w, h) // 4

        # Add small random offset to center
        x += np.random.randint(-radius // 4, radius // 4)
        y += np.random.randint(-radius // 4, radius // 4)

        # Draw multiple circles with decreasing opacity for feathered edge
        for r in range(radius, radius - 10, -1):
            opacity = int(255 * (r - (radius - 10)) / 10)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, opacity))

    return mask, (x, y)


def is_black_output(
    modality: str, output: Union[List[Image.Image], Image.Image], threshold: int = 10
) -> bool:
    """
    Returns True if the image or frames are (almost) completely black.
    """
    if modality == "image":
        arr = np.array(output[modality].images[0])
        return np.mean(arr) < threshold
    elif modality == "video":
        return np.all([np.mean(np.array(arr)) < threshold for arr in output[modality].frames[0]])
