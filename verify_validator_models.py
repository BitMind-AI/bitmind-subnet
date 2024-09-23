from bitmind.synthetic_image_generation.synthetic_image_generator import SyntheticImageGenerator
from bitmind.constants import DIFFUSER_NAMES
import bittensor as bt

def main():
    synthetic_image_generator = SyntheticImageGenerator(prompt_type='annotation',
                                        use_random_diffuser=True,
                                        diffuser_name=None)
    synthetic_image_generator.image_annotation_generator.load_models()
    synthetic_image_generator.image_annotation_generator.clear_gpu()
    
    for model_name in DIFFUSER_NAMES:
        synthetic_image_generator = SyntheticImageGenerator(prompt_type='annotation',
                                        use_random_diffuser=False,
                                        diffuser_name=model_name)
        synthetic_image_generator.load_diffuser(model_name)
        synthetic_image_generator.clear_gpu()
        
if __name__ == "__main__":
    main()