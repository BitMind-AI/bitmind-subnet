import os
import torch


WANDB_PROJECT = 'bitmind'
WANDB_ENTITY = 'bitmindai'

DATASET_META = {
    "real": [
        {"path": "bitmind/bm-real"},
        {"path": "bitmind/open-images-v7"},
        {"path": "bitmind/celeb-a-hq"},
        {"path": "bitmind/ffhq-256"},
        {"path": "bitmind/MS-COCO-unique-256"}
    ],
    "fake": [
        {"path": "bitmind/bm-realvisxl"},
        {"path": "bitmind/bm-mobius"},
        {"path": "bitmind/bm-sdxl"}
    ]
}

FACE_TRAINING_DATASET_META = {
    "real": [
        {"path": "bitmind/ffhq-256_training_faces", "name": "base_transforms"},
        {"path": "bitmind/celeb-a-hq_training_faces", "name": "base_transforms"}

    ],
    "fake": [
        {"path": "bitmind/ffhq-256___stable-diffusion-xl-base-1.0_training_faces", "name": "base_transforms"},
        {"path": "bitmind/celeb-a-hq___stable-diffusion-xl-base-1.0___256_training_faces", "name": "base_transforms"}
    ]
}

VALIDATOR_DATASET_META = {
    "real": [
        {"path": "bitmind/bm-real"},
        {"path": "bitmind/open-images-v7"},
        {"path": "bitmind/celeb-a-hq"},
        {"path": "bitmind/ffhq-256"},
        {"path": "bitmind/MS-COCO-unique-256"},
        {"path": "bitmind/AFHQ"},
        {"path": "bitmind/lfw"},
        {"path": "bitmind/caltech-256"},
        {"path": "bitmind/caltech-101"},
        {"path": "bitmind/dtd"}
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
    "t2i_models": [
        {
            "path": "stabilityai/stable-diffusion-xl-base-1.0",
            "use_safetensors": True,
            "torch_dtype": torch.float16,
            "variant": "fp16",
            "pipeline": "StableDiffusionXLPipeline"
        },
        {
            "path": "SG161222/RealVisXL_V4.0",
            "use_safetensors": True,
            "torch_dtype": torch.float16,
            "variant": "fp16",
            "pipeline": "StableDiffusionXLPipeline"
        },
        {
            "path": "Corcelio/mobius",
            "use_safetensors": True,
            "torch_dtype": torch.float16,
            "pipeline": "StableDiffusionXLPipeline"
        }
    ],
    "t2v_models": [
        {
            "path": 'THUDM/CogVideoX-5b',
            "use_safetensors": True,
            "torch_dtype": torch.bfloat16,
            "enable_cpu_offload": False,
            "vae_enable_tiling": True,
            "generate_args": {
                "guidance_scale": 2,
                "num_videos_per_prompt": 1,
                "num_inference_steps": {"min": 50, "max": 125},
                "num_frames": {"min": 30, "max": 60},
                "generator": torch.Generator("cuda" if torch.cuda.is_available() else "cpu"),
            },
            "pipeline": "CogVideoXPipeline",
        },
    ]
}

HUGGINGFACE_CACHE_DIR = os.path.expanduser('~/.cache/huggingface')

TARGET_IMAGE_SIZE = (256, 256)

PROMPT_TYPES = ('random', 'annotation', 'none')

PROMPT_GENERATOR_ARGS = {
    m['model']: m for m in VALIDATOR_MODEL_META['prompt_generators']
}

PROMPT_GENERATOR_NAMES = list(PROMPT_GENERATOR_ARGS.keys())

T2V_MODEL_NAMES = list([m['path'] for m in VALIDATOR_MODEL_META['t2v_models']])
T2I_MODEL_NAMES = list([m['path'] for m in VALIDATOR_MODEL_META['t2i_models']])
MODEL_NAMES = T2I_MODEL_NAMES + T2V_MODEL_NAMES

IMAGE_ANNOTATION_MODEL = "Salesforce/blip2-opt-6.7b-coco"

TEXT_MODERATION_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# args for .from_pretrained
MODEL_INIT_ARGS = {
    m['path']: {
        k: v for k, v in m.items()
        if k not in ('path', 'pipeline', 'generate_args', 'enable_cpu_offload', 'vae_enable_tiling')
    } for m in VALIDATOR_MODEL_META['t2v_models'] + VALIDATOR_MODEL_META['t2i_models']
}

GENERATE_ARGS = {
    m['path']: m['generate_args']
    for m in VALIDATOR_MODEL_META['t2v_models'] + VALIDATOR_MODEL_META['t2i_models']
    if 'generate_args' in m
}

MODEL_CPU_OFFLOAD_ENABLED = {
    m['path']: m.get('enable_cpu_offload', False)
    for m in VALIDATOR_MODEL_META['t2v_models'] + VALIDATOR_MODEL_META['t2i_models']
}

MODEL_VAE_ENABLE_TILING = {
    m['path']: m.get('vae_enable_tiling', False)
    for m in VALIDATOR_MODEL_META['t2v_models']
}

MODEL_PIPELINE = {
    m['path']: m['pipeline'] 
    for m in VALIDATOR_MODEL_META['t2v_models'] + VALIDATOR_MODEL_META['t2i_models'] 
    if 'pipeline' in m
}
