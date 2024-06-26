from pathlib import Path
import json

WANDB_PROJECT = 'bitmind'

DATASET_META = {
    "real": [
        {"path": "dalle-mini/open-images", "create_splits": False},
        {"path": "merkol/ffhq-256", "create_splits": True},
        {"path": "saitsharipov/CelebA-HQ", "create_splits": True}
    ],
    "fake": [
        {"path": "bitmind/RealVisXL_V4.0_images", "create_splits": True},
        {"path": "bitmind/stable-diffusion-xl-base-1.0-images", "create_splits": True},
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
            "variant": "fp16"
        },
        {
            "path": "SG161222/RealVisXL_V4.0",
            "use_safetensors": True,
            "variant": "fp16"
        },
        {
            "path": "Corcelio/mobius",
            "use_safetensors": True
        }
    ]
}

PROMPT_GENERATOR_ARGS = {
    m['model']: m for m in VALIDATOR_MODEL_META['prompt_generators']
}

PROMPT_GENERATOR_NAMES = list(PROMPT_GENERATOR_ARGS.keys())

DIFFUSER_ARGS = {
    m['path']: {k: v for k, v in m.items() if k != 'path'}  
    for m in VALIDATOR_MODEL_META['diffusers']
}

DIFFUSER_NAMES = list(DIFFUSER_ARGS.keys())

