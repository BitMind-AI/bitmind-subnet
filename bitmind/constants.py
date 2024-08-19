import os


WANDB_PROJECT = 'bitmind-subnet'
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
            "path": "stabilityai/stable-diffusion-xl-base-1.0",
            "use_safetensors": True,
            "variant": "fp16",
            "pipeline": "StableDiffusionXLPipeline"
        },
        {
            "path": "SG161222/RealVisXL_V4.0",
            "use_safetensors": True,
            "variant": "fp16",
            "pipeline": "StableDiffusionXLPipeline"
        },
        {
            "path": "Corcelio/mobius",
            "use_safetensors": True,
            "pipeline": "StableDiffusionXLPipeline"
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
    m['path']: {k: v for k, v in m.items() if k != 'path' and k != 'pipeline'}  
    for m in VALIDATOR_MODEL_META['diffusers']
}

DIFFUSER_PIPELINE = {
    m['path']: m['pipeline'] for m in VALIDATOR_MODEL_META['diffusers'] if 'pipeline' in m
}

DIFFUSER_NAMES = list(DIFFUSER_ARGS.keys())

IMAGE_ANNOTATION_MODEL = "Salesforce/blip2-opt-2.7b-coco"
