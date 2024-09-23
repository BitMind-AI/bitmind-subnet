from bitmind.synthetic_image_generation.synthetic_image_generator import SyntheticImageGenerator
from bitmind.constants import DIFFUSER_NAMES

def main():
    for model_name in DIFFUSER_NAMES:
        print(f"Verifying {model_name}...")
        synthetic_image_generator = SyntheticImageGenerator(prompt_type='annotation',
                                        use_random_diffuser=False,
                                        diffuser_name=model_name)
        print(f"{model_name} successfully loaded.")
        
if __name__ == "__main__":
    main()