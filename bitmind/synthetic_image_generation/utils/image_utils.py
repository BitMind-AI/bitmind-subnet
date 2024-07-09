import PIL

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
