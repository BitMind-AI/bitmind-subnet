import argparse
from pathlib import Path
from PIL import Image
import bittensor as bt
import numpy as np

from bitmind.synthetic_data_generation.in_painting_generator import InPaintingGenerator

def create_inpaint_only_image(original: Image.Image, mask: Image.Image, result: Image.Image) -> Image.Image:
    """Create an image showing only the inpainted region."""
    # Ensure all images are the same size as the result
    size = result.size
    original = original.resize(size, Image.Resampling.LANCZOS)
    mask = mask.resize(size, Image.Resampling.LANCZOS)
    
    # Convert mask to boolean array (True where white/inpainted)
    mask_array = np.array(mask.convert('L')) > 128
    
    # Create blank (black) image
    inpaint_only = Image.new('RGB', size, 'black')
    
    # Copy only the inpainted region
    inpaint_array = np.array(result)
    inpaint_only_array = np.array(inpaint_only)
    inpaint_only_array[mask_array] = inpaint_array[mask_array]
    
    return Image.fromarray(inpaint_only_array)


def create_final_image(original: Image.Image, mask: Image.Image, inpainted: Image.Image) -> Image.Image:
    """
    Create the final image by combining original and inpainted regions.
    Only use inpainted content where the mask is white.
    """
    # Ensure all images are in RGB mode
    original = original.convert('RGB')
    inpainted = inpainted.convert('RGB')
    
    # Ensure all images are the same size
    size = original.size
    mask = mask.resize(size, Image.Resampling.LANCZOS)
    inpainted = inpainted.resize(size, Image.Resampling.LANCZOS)
    
    # Convert to numpy arrays with float32 for precise calculations
    original_array = np.array(original, dtype=np.float32)
    inpainted_array = np.array(inpainted, dtype=np.float32)
    
    # Create mask array (values between 0 and 1)
    mask_array = np.array(mask.convert('L'), dtype=np.float32) / 255.0
    
    # Expand mask dimensions to match RGB
    mask_array = np.expand_dims(mask_array, axis=-1)
    
    # Blend images using the mask as alpha
    result_array = (original_array * (1 - mask_array) + 
                   inpainted_array * mask_array)
    
    # Convert back to uint8 with proper rounding
    result_array = np.clip(np.round(result_array), 0, 255).astype(np.uint8)
    
    return Image.fromarray(result_array, mode='RGB')


def main():
    parser = argparse.ArgumentParser(description='Test InPainting Generator on a single image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--custom_prompt', type=str, help='Optional custom prompt to use')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load image
        bt.logging.info(f"Loading image from {args.image_path}")
        image = Image.open(args.image_path)
        
        # Initialize generator
        generator = InPaintingGenerator(
            use_random_i2i_model=True,
            output_dir=output_dir,
            device=args.device
        )
        
        # Generate transformation
        result = generator.run_i2i(
            prompt=args.custom_prompt if args.custom_prompt else generator.generate_prompt(image),
            original_image=image
        )
        
        # Get final size from generated image
        final_size = result['gen_output'].images[0].size
        
        # Save outputs
        timestamp = str(int(result['time']))
        
        # 1. Save the original image (resized to match output)
        original_resized = image.resize(final_size, Image.Resampling.LANCZOS)
        original_path = output_dir / f"1_original_{timestamp}.png"
        original_resized.save(original_path)
        bt.logging.info(f"Saved original image to {original_path}")
        
        # 2. Save the mask (resized to match output)
        mask = generator.create_random_mask(final_size)
        mask_path = output_dir / f"2_mask_{timestamp}.png"
        mask.save(mask_path)
        bt.logging.info(f"Saved mask to {mask_path}")
        
        # 3. Save the inpainted region only
        inpaint_only = create_inpaint_only_image(
            original_resized,
            mask,
            result['gen_output'].images[0]
        )
        inpaint_path = output_dir / f"3_inpaint_only_{timestamp}.png"
        inpaint_only.save(inpaint_path)
        bt.logging.info(f"Saved inpainted region to {inpaint_path}")
        
        # 4. Save the final transformed image (properly combined)
        final_image = create_final_image(
            original_resized,
            mask,
            result['gen_output'].images[0]
        )
        final_path = output_dir / f"4_final_{timestamp}.png"
        final_image.save(final_path)
        bt.logging.info(f"Saved final image to {final_path}")
        
        # Save the prompt/caption to a text file
        caption_path = output_dir / f"caption_{timestamp}.txt"
        with open(caption_path, 'w') as f:
            f.write(f"Generated Prompt: {result['prompt']}\n")
            if result['prompt'] != result['prompt_long']:
                f.write(f"Full Prompt: {result['prompt_long']}\n")
            f.write(f"\nGeneration Time: {result['gen_time']:.2f} seconds\n")
            f.write(f"Model: {result['model_name']}\n")
        bt.logging.info(f"Saved caption to {caption_path}")
        
        # Print generation info
        bt.logging.info(f"Prompt used: {result['prompt']}")
        bt.logging.info(f"Generation time: {result['gen_time']:.2f} seconds")
        
    except Exception as e:
        bt.logging.error(f"Error during inpainting generation: {e}")
        raise


if __name__ == "__main__":
    main() 