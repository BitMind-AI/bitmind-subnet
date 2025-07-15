import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from bitmind.generation.util.image import create_random_mask


def overlay_mask_on_image(image: Image.Image, mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    """
    Overlay a mask (white on black) onto an image with given alpha transparency.
    """
    image = image.convert("RGBA")
    mask = mask.convert("L")
    mask_rgba = Image.new("RGBA", image.size, (255, 0, 0, 0))
    mask_rgba_np = np.array(mask_rgba)
    mask_np = np.array(mask)
    mask_rgba_np[..., 0] = 255  # Red channel
    mask_rgba_np[..., 3] = (mask_np * alpha).astype(np.uint8)  # Alpha channel
    mask_rgba = Image.fromarray(mask_rgba_np, mode="RGBA")
    return Image.alpha_composite(image, mask_rgba)


def show_mask(mask: Image.Image, title: str = "Mask"):
    plt.figure()
    plt.imshow(mask)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_overlay(image: Image.Image, mask: Image.Image, title: str = "Overlay"):
    overlay = overlay_mask_on_image(image, mask)
    plt.figure()
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")
    plt.show()


def demo_generate_and_visualize_masks(
    image_size=(256, 256),
    n_examples=5,
    mask_kwargs=None,
):
    """
    Generate and visualize several random masks on blank images.
    """
    if mask_kwargs is None:
        mask_kwargs = {}
    for i in range(n_examples):
        blank = Image.new("RGB", image_size, (200, 200, 200))
        mask, meta = create_random_mask(image_size, **mask_kwargs)
        print(f"Mask {i+1} metadata:", meta)
        show_mask(mask, title=f"Mask {i+1}")
        show_overlay(blank, mask, title=f"Overlay {i+1}")


if __name__ == "__main__":
    demo_generate_and_visualize_masks() 