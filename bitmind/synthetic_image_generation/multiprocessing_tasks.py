import logging
from diffusers import DiffusionPipeline
from multiprocessing import current_process
import torch
from torchvision.transforms import ToPILImage
import os

def load_diffuser(model_name, device, DIFFUSER_ARGS):
    logging.info(f"Loading image generation model ({model_name})...")
    model = DiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float32 if device == "cpu" else torch.float16, **DIFFUSER_ARGS[model_name]
    )
    model.to(device)
    return model

def worker_initializer(model_name, device, DIFFUSER_ARGS):
    global diffuser
    diffuser = load_diffuser(model_name, device, DIFFUSER_ARGS)
    logging.info(f"Model loaded in process: {current_process().name}")

def generate_images_for_chunk(annotations, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    generated_images = []
    for i, annotation in enumerate(annotations):
        prompt = annotation['description']
        logging.info(f"Generating image for annotation {i}: {prompt}")
        generated_image = diffuser(prompt=prompt).images[0] 
        img_filename = os.path.join(save_dir, f"{prompt[:50].replace(' ', '_')}-{i}.png")
        if isinstance(generated_image, torch.Tensor):
            img = ToPILImage()(generated_image)
        else:
            img = generated_image  # No conversion needed
        img.save(img_filename)
        generated_images.append(img_filename)
        logging.info(f"Image saved to {img_filename}")
    return generated_images
