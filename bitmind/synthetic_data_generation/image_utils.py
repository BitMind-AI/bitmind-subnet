import PIL
import os
import json
from bitmind.validator.config import TARGET_IMAGE_SIZE


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


def resize_images_in_directory(directory, target_width=TARGET_IMAGE_SIZE[0], target_height=TARGET_IMAGE_SIZE[1]):
    """
    Resize all images in the specified directory to the target width and height.

    Args:
    directory (str): Path to the directory containing images.
    target_width (int): Target width for resizing the images.
    target_height (int): Target height for resizing the images.
    """
    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Check for image file extensions
            filepath = os.path.join(directory, filename)
            with PIL.Image.open(filepath) as img:
                # Resize the image and save back to the file location
                resized_img = resize_image(img, max_width=target_width, max_height=target_height)
                resized_img.save(filepath)
                

def save_images_to_disk(image_dataset, start_index, num_images, save_directory, resize=True):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for i in range(start_index, start_index + num_images):
        try:
            image_data = image_dataset[i]  # Retrieve image using the __getitem__ method
            image = image_data['image']  # Extract the image
            image_id = image_data['id']  # Extract the image ID
            file_path = os.path.join(save_directory, f"{image_id}.jpg")  # Construct file path
            if resize:
                image = resize_image(image, TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1])
            image.save(file_path, 'JPEG')  # Save the image
            print(f"Saved: {file_path}")
        except Exception as e:
            print(f"Failed to save image {i}: {e}")