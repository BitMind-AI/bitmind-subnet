from pathlib import Path
from typing import Dict, List, Union, Optional, Any

import numpy as np
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    FluxPipeline,
    CogVideoXPipeline,
    AnimateDiffPipeline,
    EulerDiscreteScheduler
)

from .model_utils import load_annimatediff_motion_adapter


TARGET_IMAGE_SIZE: tuple[int, int] = (224, 224)  # (256, 256)

# Project constants
WANDB_PROJECT: str = 'bitmind'
WANDB_ENTITY: str = 'bitmindai'

# Cache directories
HUGGINGFACE_CACHE_DIR: Path = Path.home() / '.cache' / 'huggingface'
VIDEO_CACHE_DIR: Path = Path.home() / '.cache' / 'sn34' / 'video'
IMAGE_CACHE_DIR:  Path = Path.home() / '.cache' / 'sn34' / 'image'

# Image datasets configuration
IMAGE_DATASETS: Dict[str, List[Dict[str, str]]] = {
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

VIDEO_DATASETS = {
    "real": [
        {
            "path": "nkp37/OpenVid-1M",
            "filetype": "zip"
        },
        {
            "path": "shangxd/imagenet-vidvrd",
            "filetype": "zip"
    	}
    ]
}



# Prompt generation model configurations
IMAGE_ANNOTATION_MODEL: str = "Salesforce/blip2-opt-6.7b-coco"
TEXT_MODERATION_MODEL: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# Text-to-image model configurations
T2I_MODELS: Dict[str, Dict[str, Any]] = {
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "pipeline_cls": StableDiffusionXLPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.float16,
            "variant": "fp16"
        }
    },
    "SG161222/RealVisXL_V4.0": {
        "pipeline_cls": StableDiffusionXLPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.float16,
            "variant": "fp16"
        }
    },
    "Corcelio/mobius": {
        "pipeline_cls": StableDiffusionXLPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.float16
        }
    }
}
T2I_MODEL_NAMES: List[str] = list(T2I_MODELS.keys())

# Text-to-video model configurations
T2V_MODELS: Dict[str, Dict[str, Any]] = {
    'THUDM/CogVideoX-5b': {
        "pipeline_cls": CogVideoXPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.bfloat16
        },
        "generate_args": {
            "guidance_scale": 2,
            "num_videos_per_prompt": 1,
            "num_inference_steps": {"min": 50, "max": 125},
            "num_frames": 48,
        },
        "enable_model_cpu_offload": True,
        "enable_sequential_cpu_offload": True,
        "vae_enable_slicing": True,
        "vae_enable_tiling": True
    },
    'ByteDance/AnimateDiff-Lightning': {
        "pipeline_cls": AnimateDiffPipeline,
        "from_pretrained_args": {
            "base": "emilianJR/epiCRealism",
            "torch_dtype": torch.bfloat16,
            "motion_adapter": load_annimatediff_motion_adapter()
        },
        "generate_args": {
            "guidance_scale": 2,
            "num_inference_steps": {"min": 50, "max": 125},
        },
        "scheduler": {
            "cls": EulerDiscreteScheduler,
            "from_config_args": {
                "timestep_spacing": "trailing",
                "beta_schedule": "linear"
            }
        }
    }
}
T2V_MODEL_NAMES: List[str] = list(T2V_MODELS.keys())

# Combined model configurations
T2VIS_MODELS: Dict[str, Dict[str, Any]] = {**T2I_MODELS, **T2V_MODELS}
T2VIS_MODEL_NAMES: List[str] = list(T2VIS_MODELS.keys())


def select_random_t2vis_model(modality: Optional[str] = None) -> str:
    """
    Select a random text-to-image or text-to-video model based on the specified
    modality.

    Args:
        modality: The type of model to select ('image', 'video', or 'random').
            If None or 'random', randomly chooses between image and video.

    Returns:
        The name of the selected model.

    Raises:
        NotImplementedError: If the specified modality is not supported.
    """
    if modality is None or modality == 'random':
        modality = np.random.choice(['image', 'video'])

    if modality == 'image':
        return np.random.choice(T2I_MODEL_NAMES)
    elif modality == 'video':
        return np.random.choice(T2V_MODEL_NAMES)
    else:
        raise NotImplementedError(f"Unsupported modality: {modality}")
