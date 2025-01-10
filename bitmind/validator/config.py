from pathlib import Path
from typing import Dict, List, Union, Optional, Any

import numpy as np
import torch
from diffusers import (
    StableDiffusionPipeline,    
    StableDiffusionXLPipeline,
    FluxPipeline,
    CogVideoXPipeline,
    MochiPipeline,
    AnimateDiffPipeline,
    EulerDiscreteScheduler,
    AutoPipelineForInpainting,
    IFPipeline,
    IFSuperResolutionPipeline
)

from .model_utils import load_annimatediff_motion_adapter


TARGET_IMAGE_SIZE: tuple[int, int] = (256, 256)

MAINNET_UID = 34
TESTNET_UID = 168

# Project constants
MAINNET_WANDB_PROJECT: str = 'bitmind-subnet'
TESTNET_WANDB_PROJECT: str = 'bitmind'
WANDB_ENTITY: str = 'bitmindai'

# Cache directories
HUGGINGFACE_CACHE_DIR: Path = Path.home() / '.cache' / 'huggingface'
SN34_CACHE_DIR: Path = Path.home() / '.cache' / 'sn34'
SN34_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VALIDATOR_INFO_PATH: Path = SN34_CACHE_DIR / 'validator.yaml'

REAL_CACHE_DIR: Path = SN34_CACHE_DIR / 'real'
SYNTH_CACHE_DIR: Path = SN34_CACHE_DIR / 'synthetic'

REAL_VIDEO_CACHE_DIR: Path = REAL_CACHE_DIR / 'video'
REAL_IMAGE_CACHE_DIR: Path = REAL_CACHE_DIR / 'image'

T2V_CACHE_DIR: Path = SYNTH_CACHE_DIR / 't2v' 
T2I_CACHE_DIR: Path = SYNTH_CACHE_DIR / 't2i'
I2I_CACHE_DIR: Path = SYNTH_CACHE_DIR / 'i2i'

# Update intervals in hours
VIDEO_ZIP_CACHE_UPDATE_INTERVAL = 3
IMAGE_PARQUET_CACHE_UPDATE_INTERVAL = 2
VIDEO_CACHE_UPDATE_INTERVAL = 1
IMAGE_CACHE_UPDATE_INTERVAL = 1

MAX_COMPRESSED_GB = 100
MAX_EXTRACTED_GB = 10

CHALLENGE_TYPE = {
    0: 'real',
    1: 'synthetic'
}

# Image datasets configuration
IMAGE_DATASETS: Dict[str, List[Dict[str, str]]] = {
    "real": [
        {"path": "bitmind/bm-real"},
        {"path": "bitmind/open-image-v7-256"},
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
        },
        "use_autocast": False
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
    },
    "black-forest-labs/FLUX.1-dev": {
        "pipeline_cls": FluxPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.bfloat16,
        },
        "generate_args": {
            "guidance_scale": 2,
            "num_inference_steps": {"min": 50, "max": 125},
            "generator": torch.Generator("cuda" if torch.cuda.is_available() else "cpu"),
            "height": [512, 768],
            "width": [512, 768]
        },
        "enable_model_cpu_offload": False
    },
    "prompthero/openjourney-v4" : {
        "pipeline_cls": StableDiffusionPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.float16,
        }
    },
    "cagliostrolab/animagine-xl-3.1": {
        "pipeline_cls": StableDiffusionXLPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.float16,
        },
    },
    "DeepFloyd/IF": {
        "pipeline_cls": {
            "stage1": IFPipeline,
            "stage2": IFSuperResolutionPipeline
        },
        "from_pretrained_args": {
            "stage1": {
                "base": "DeepFloyd/IF-I-XL-v1.0",
                "torch_dtype": torch.float16,
                "variant": "fp16"
            },
            "stage2": {
                "base": "DeepFloyd/IF-II-L-v1.0",
                "torch_dtype": torch.float16,
                "variant": "fp16"
            }
        },
        "pipeline_stages": [
            {
                "name": "stage1",
                "output_attr": "images"
            },
            {
                "name": "stage2",
                "input_key": "image",
                "output_attr": "images"
            }
        ],
        "generate_args": {
            "num_inference_steps": {"min": 25, "max": 50},
            "guidance_scale": 7.0
        },
        "enable_model_cpu_offload": True
    },
}
T2I_MODEL_NAMES: List[str] = list(T2I_MODELS.keys())

# Image-to-image model configurations
I2I_MODELS: Dict[str, Dict[str, Any]] = {
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1": {
        "pipeline_cls": AutoPipelineForInpainting,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.float16,
            "variant": "fp16"
        },
        "generate_args": {
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "strength": 0.99,
            "generator": torch.Generator("cuda" if torch.cuda.is_available() else "cpu"),
        }
    }
}
I2I_MODEL_NAMES: List[str] = list(I2I_MODELS.keys())

# Text-to-video model configurations
T2V_MODELS: Dict[str, Dict[str, Any]] = {
    "genmo/mochi-1-preview": {
        "pipeline_cls": MochiPipeline,
        "from_pretrained_args": {
            "variant": "bf16", 
            "torch_dtype": torch.bfloat16
        },
        "generate_args": {
            "num_frames": 84
        },
        #"enable_model_cpu_offload": True,
        "vae_enable_tiling": True
    },
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
        #"enable_sequential_cpu_offload": True,
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
MODELS: Dict[str, Dict[str, Any]] = {**T2I_MODELS, **I2I_MODELS, **T2V_MODELS}
MODEL_NAMES: List[str] = list(MODELS.keys())


def get_modality(model_name):
     if model_name in T2V_MODEL_NAMES:
        return 'video'
     elif model_name in T2I_MODEL_NAMES + I2I_MODEL_NAMES:
        return 'image'   


def get_task(model_name):
    if model_name in T2V_MODEL_NAMES:
        return 't2v'
    elif model_name in T2I_MODEL_NAMES:
        return 't2i'
    elif model_name in I2I_MODEL_NAMES:
        return 'i2i'


def select_random_model(task: Optional[str] = None) -> str:
    """
    Select a random text-to-image or text-to-video model based on the specified
    modality.

    Args:
        modality: The type of model to select ('t2v', 't2i', 'i2i', or 'random').
            If None or 'random', randomly chooses between the valid options

    Returns:
        The name of the selected model.

    Raises:
        NotImplementedError: If the specified modality is not supported.
    """
    if task is None or task == 'random':
        task = np.random.choice(['t2i', 'i2i', 't2v'])

    if task == 't2i':
        return np.random.choice(T2I_MODEL_NAMES)
    elif task == 't2v':
        return np.random.choice(T2V_MODEL_NAMES)
    elif task == 'i2i':
        return np.random.choice(I2I_MODEL_NAMES)
    else:
        raise NotImplementedError(f"Unsupported task: {task}")


def create_pipeline_generator(model_config: Dict[str, Any], model: Any) -> callable:
    """Creates a generator function based on pipeline configuration."""
    
    if isinstance(model_config.get('pipeline_stages'), list):
        def generate(prompt: str, **kwargs):
            output = None
            for stage in model_config['pipeline_stages']:
                stage_args = {**kwargs}  # Copy base args
                
                # Add stage-specific args
                if stage.get('input_key') and output is not None:
                    stage_args[stage['input_key']] = output
                
                # Add any stage-specific generation args
                if stage.get('args'):
                    stage_args.update(stage['args'])
                
                # Run stage
                result = model[stage['name']](prompt=prompt, **stage_args)
                
                # Extract output based on stage config
                output = getattr(result, stage.get('output_attr', 'images'))[0]
            return result
        return generate
    
    # Default single-stage pipeline
    return lambda prompt, **kwargs: model(prompt=prompt, **kwargs)