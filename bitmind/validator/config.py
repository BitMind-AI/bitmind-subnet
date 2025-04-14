from strenum import StrEnum
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
    HunyuanVideoPipeline,
    AnimateDiffPipeline,
    IFPipeline,
    IFSuperResolutionPipeline,
    EulerDiscreteScheduler,
    DEISMultistepScheduler,
    AutoPipelineForInpainting,
    StableDiffusionInpaintPipeline,
    CogView4Pipeline,
    CogVideoXImageToVideoPipeline
)

from .model_utils import (
    load_annimatediff_motion_adapter,
    load_hunyuanvideo_transformer,
    JanusWrapper
)


TARGET_IMAGE_SIZE: tuple[int, int] = (256, 256)

MAINNET_UID = 34
TESTNET_UID = 168

# Project constants
MAINNET_WANDB_PROJECT: str = 'bitmind-subnet'
TESTNET_WANDB_PROJECT: str = 'bitmind'
WANDB_ENTITY: str = 'bitmindai'


# Enums
class MediaType(StrEnum):
    REAL = "real"
    SYNTHETIC = "synthetic"
    SEMISYNTHETIC = "semisynthetic"


class Modality(StrEnum):
    IMAGE = "image"
    VIDEO = "video"


# Cache directories
HUGGINGFACE_CACHE_DIR: Path = Path.home() / '.cache' / 'huggingface'
SN34_CACHE_DIR: Path = Path.home() / '.cache' / 'sn34'
SN34_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VALIDATOR_INFO_PATH: Path = SN34_CACHE_DIR / 'validator.yaml'

IMAGE_CACHE_DIR: Path = SN34_CACHE_DIR / Modality.IMAGE
VIDEO_CACHE_DIR: Path = SN34_CACHE_DIR / Modality.VIDEO

REAL_IMAGE_CACHE_DIR: Path = IMAGE_CACHE_DIR / MediaType.REAL
SYNTH_IMAGE_CACHE_DIR: Path = IMAGE_CACHE_DIR / MediaType.SYNTHETIC
SEMISYNTH_IMAGE_CACHE_DIR: Path = IMAGE_CACHE_DIR / MediaType.SEMISYNTHETIC

REAL_VIDEO_CACHE_DIR: Path = VIDEO_CACHE_DIR / MediaType.REAL
SYNTH_VIDEO_CACHE_DIR: Path = VIDEO_CACHE_DIR / MediaType.SYNTHETIC
SEMISYNTH_VIDEO_CACHE_DIR: Path = VIDEO_CACHE_DIR / MediaType.SEMISYNTHETIC

LABELS = (0, 1, 2)
LABEL_TO_TYPE = {
    0: MediaType.REAL,
    1: MediaType.SYNTHETIC,
    2: MediaType.SEMISYNTHETIC
}

P_REAL: float = 0.5
P_SYNTH: float = 0.4
P_SEMISYNTH: float = 0.1
LABEL_PROBS: List[float] = (P_REAL, P_SYNTH, P_SEMISYNTH)

MODALITY_PROBS = (0.5, 0.5)

# Probability of concatenating together two videos
# Will only ever combine videos of the same type 
# i.e. real + real, synth + synth, semisynth + semisynth
P_STITCH: float = 0.2

# Number of frames in challenge 
MIN_FRAMES = 8
MAX_FRAMES = 129

# Update intervals in hours
VIDEO_ZIP_CACHE_UPDATE_INTERVAL = 2
IMAGE_PARQUET_CACHE_UPDATE_INTERVAL = 2
VIDEO_CACHE_UPDATE_INTERVAL = 1
IMAGE_CACHE_UPDATE_INTERVAL = 1

MAX_COMPRESSED_GB = 50
MAX_EXTRACTED_GB = 5


# dataset configurations
IMAGE_DATASETS = {
    "real": [
        {"path": "bitmind/bm-eidon-image"},
        {"path": "bitmind/bm-real"},
        {"path": "bitmind/open-image-v7-256"},
        {"path": "bitmind/celeb-a-hq"},
        {"path": "bitmind/ffhq-256"},
        {"path": "bitmind/MS-COCO-unique-256"},
        {"path": "bitmind/AFHQ"},
        {"path": "bitmind/lfw"},
        {"path": "bitmind/caltech-256"},
        {"path": "bitmind/caltech-101"},
        {"path": "bitmind/dtd"},
        {"path": "bitmind/idoc-mugshots-images"}
    ],
    "semisynthetic": [
        {"path": "bitmind/face-swap"}
    ],
    "synthetic": [
        {"path": "bitmind/JourneyDB"},
        {"path": "bitmind/GenImage_MidJourney"}
    ]
}

VIDEO_DATASETS = {
    "real": [
        {"path": "bitmind/bm-eidon-video", "filetype": "zip"},
        {"path": "shangxd/imagenet-vidvrd", "filetype": "zip"},
        {"path": "nkp37/OpenVid-1M", "filetype": "zip"}
    ],
    "semisynthetic": [
        {"path": "bitmind/semisynthetic-video", "filetype": "zip"}
    ]
}


# Prompt generation model configurations
IMAGE_ANNOTATION_MODEL: str = "Salesforce/blip2-opt-6.7b-coco"
TEXT_MODERATION_MODEL: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# Text-to-image model configurations
T2I_MODELS: Dict[str, Dict[str, Any]] = {
    "THUDM/CogView4-6B": {
        "pipeline_cls": CogView4Pipeline,
        "from_pretrained_args": {
            "torch_dtype": torch.bfloat16,
            "use_safetensors": True
        },
        "generate_args": {
            "guidance_scale": 3.5,
            "num_images_per_prompt": 1,
            "num_inference_steps": 50,
            "width": 512,
            "height": 512
        },
        "use_autocast": False
    },
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
            "resolution": [512, 768]
        },
        "enable_model_cpu_offload": False
    },
    "runwayml/stable-diffusion-v1-5-midjourney-v6": {
        "pipeline_cls": StableDiffusionPipeline,
        "from_pretrained_args": {
            "model_id": "runwayml/stable-diffusion-v1-5",
            "use_safetensors": True,
            "torch_dtype": torch.float16,
        },
        "lora_model_id": "Kvikontent/midjourney-v6",
        "lora_loading_args": {
            "use_peft_backend": True
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
                "variant": "fp16",
                "clean_caption": False,
                "watermarker": None,
                "requires_safety_checker": False
            },
            "stage2": {
                "base": "DeepFloyd/IF-II-L-v1.0",
                "torch_dtype": torch.float16,
                "variant": "fp16",
                "text_encoder": None,
                "watermarker": None,
                "requires_safety_checker": False
            }
        },
        "pipeline_stages": [
            {
                "name": "stage1",
                "args": {
                    "output_type": "pt",
                    "num_images_per_prompt": 1,
                    "return_dict": True
                },
                "output_attr": "images",
                "output_transform": lambda x: x[0].unsqueeze(0),
                "save_prompt_embeds": True
            },
            {
                "name": "stage2",
                "input_key": "image",
                "args": {
                    "output_type": "pil",
                    "num_images_per_prompt": 1
                },
                "output_attr": "images",
                "use_prompt_embeds": True
            }
        ],
        "clear_memory_on_stage_end": True
    },
    "deepseek-ai/Janus-Pro-7B": {
        "pipeline_cls": JanusWrapper,
        "from_pretrained_args": {
            "torch_dtype": torch.bfloat16,
            "use_safetensors": True,
        },
        "generate_args": {
            "temperature": 1.0,
            "parallel_size": 4,
            "cfg_weight": 5.0,
            "image_token_num_per_image": 576,
            "img_size": 384,
            "patch_size": 16
        },
        "use_autocast": False,
        "enable_model_cpu_offload": False
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
    },
    "Lykon/dreamshaper-8-inpainting": {
        "pipeline_cls": AutoPipelineForInpainting,
        "from_pretrained_args": {
            "torch_dtype": torch.float16,
            "variant": "fp16"
        },
        "generate_args": {
            "num_inference_steps": {"min": 40, "max": 60},
        },
        "scheduler": {
            "cls": DEISMultistepScheduler
        }
    },
    "stable-diffusion-v1-5/stable-diffusion-inpainting": {
        "pipeline_cls": StableDiffusionInpaintPipeline,
        "generate_args": {
            "num_inference_steps": {"min": 40, "max": 60},
        }
    }
}
I2I_MODEL_NAMES: List[str] = list(I2I_MODELS.keys())

# Text-to-video model configurations
T2V_MODELS: Dict[str, Dict[str, Any]] = {
    "tencent/HunyuanVideo": {
        "pipeline_cls": HunyuanVideoPipeline,
        "from_pretrained_args": {
            "model_id": "tencent/HunyuanVideo",
            "transformer": (  # custom functions supplied as tuple of (fn, args)
                load_hunyuanvideo_transformer,
                { 
                    "model_id": "tencent/HunyuanVideo",
                    "subfolder": "transformer",
                    "torch_dtype": torch.bfloat16,
                    "revision": 'refs/pr/18'
                }
            ),
            "revision": 'refs/pr/18',
            "torch_dtype": torch.bfloat16
        },
        "generate_args": {
            "num_frames": {"min": 61, "max": 129},
            "resolution": {"options": [
                [720, 1280], [1280, 720], [1104, 832], [832,1104], [960,960],
                [544, 960], [960, 544],	[624, 832], [832, 624],	[720, 720]
            ]},
            "num_inference_steps": {"min": 30, "max": 50},
        },
        "save_args": {"fps": 30},
        "use_autocast": False,
        "vae_enable_tiling": True
    },
    "genmo/mochi-1-preview": {
        "pipeline_cls": MochiPipeline,
        "from_pretrained_args": {
            "variant": "bf16", 
            "torch_dtype": torch.bfloat16
        },
        "generate_args": {
            "num_frames": 84,
            "num_inference_steps": {"min": 30, "max": 65},
            "resolution": [480, 848]
        },
        "save_args": {"fps": 30},
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
            "num_frames": 48
        },
        "save_args": {"fps": 8},
        "enable_model_cpu_offload": True,
        #"enable_sequential_cpu_offload": True,
        "vae_enable_slicing": True,
        "vae_enable_tiling": True
    },
    'ByteDance/AnimateDiff-Lightning': {
        "pipeline_cls": AnimateDiffPipeline,
        "from_pretrained_args": {
            "model_id": "emilianJR/epiCRealism",
            "torch_dtype": torch.bfloat16,
            "motion_adapter": (
                load_annimatediff_motion_adapter,
                {"step": 4}
            )
        },
        "generate_args": {
            "guidance_scale": 2,
            "num_inference_steps": {"min": 50, "max": 125},
            "resolution": {"options": [
                [512, 512], [512, 768], [512, 1024],
                [768, 512], [768, 768], [768, 1024],
                [1024, 512], [1024, 768], [1024, 1024]
            ]}
        },
        "save_args": {"fps": 15},
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

# Image-to-video model configurations
I2V_MODELS: Dict[str, Dict[str, Any]] = {}
I2V_MODEL_NAMES: List[str] = list(I2V_MODELS.keys())

# Combined model configurations
MODELS: Dict[str, Dict[str, Any]] = {**T2I_MODELS, **I2I_MODELS, **T2V_MODELS, **I2V_MODELS}
MODEL_NAMES: List[str] = list(MODELS.keys())

def get_modality(model_name):
     if model_name in T2V_MODEL_NAMES + I2V_MODEL_NAMES:
        return Modality.VIDEO
     elif model_name in T2I_MODEL_NAMES + I2I_MODEL_NAMES:
        return Modality.IMAGE

def get_output_media_type(model_name):
     if model_name in I2I_MODEL_NAMES:
        return MediaType.SEMISYNTHETIC
     elif model_name in T2I_MODEL_NAMES + T2V_MODEL_NAMES + I2V_MODEL_NAMES:
        return MediaType.SYNTHETIC

def get_task(model_name):
    if model_name in T2V_MODEL_NAMES:
        return 't2v'
    elif model_name in T2I_MODEL_NAMES:
        return 't2i'
    elif model_name in I2I_MODEL_NAMES:
        return 'i2i'
    elif model_name in I2V_MODEL_NAMES:
        return 'i2v'


def select_random_model(task: Optional[str] = None) -> str:
    """
    Select a random text-to-image, text-to-video, image-to-image, or image-to-video model based on the specified
    modality.

    Args:
        modality: The type of model to select ('t2v', 't2i', 'i2i', 'i2v', or 'random').
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
    elif task == 'i2v':
        if not I2V_MODEL_NAMES:
            raise NotImplementedError("I2V models are not currently configured")
        return np.random.choice(I2V_MODEL_NAMES)
    else:
        raise NotImplementedError(f"Unsupported task: {task}")