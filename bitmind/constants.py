import os


WANDB_PROJECT = 'bitmind'
WANDB_ENTITY = 'bitmindai'

DATASET_META = {
    "real": [
        {"path": "bitmind/open-images-v7", "create_splits": False},
        {"path": "bitmind/ffhq-256", "create_splits": False},
        {"path": "bitmind/celeb-a-hq", "create_splits": False}
    ],
    "fake": [
        {"path": "bitmind/realvis-xl", "create_splits": False},
        {"path": "bitmind/stable-diffusion-xl", "create_splits": False},
    ]
}

VALIDATOR_MODEL_META = {
    "prompt_generators": [
        {
            "model": "Gustavosta/MagicPrompt-Stable-Diffusion",
            "tokenizer": "gpt2",
            "device": -1
        }
    ],
    "diffusers": [
        {
            "path": "black-forest-labs/FLUX.1-schnell",
            "use_safetensors": True
        },
        {
            'path': 'black-forest-labs/FLUX.1-dev',
            "use_safetensors": True
        }
    ]
}

HUGGINGFACE_CACHE_DIR = os.path.expanduser('~/.cache/huggingface')

TARGET_IMAGE_SIZE = (256, 256)

PROMPT_TYPES = ('random', 'annotation')

PROMPT_GENERATOR_ARGS = {
    m['model']: m for m in VALIDATOR_MODEL_META['prompt_generators']
}

PROMPT_GENERATOR_NAMES = list(PROMPT_GENERATOR_ARGS.keys())

DIFFUSER_ARGS = {
    m['path']: {k: v for k, v in m.items() if k != 'path'}  
    for m in VALIDATOR_MODEL_META['diffusers']
}

DIFFUSER_NAMES = list(DIFFUSER_ARGS.keys())

IMAGE_ANNOTATION_MODEL = "Salesforce/blip2-opt-2.7b-coco"

