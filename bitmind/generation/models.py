from typing import List

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionInpaintPipeline,
    FluxPipeline,
    StableDiffusionPipeline,
    DEISMultistepScheduler,
    EulerDiscreteScheduler,
    IFPipeline,
    IFSuperResolutionPipeline,
    HunyuanVideoPipeline,
    MochiPipeline,
    CogVideoXPipeline,
    AnimateDiffPipeline,
    AutoPipelineForInpainting,
    CogView4Pipeline,
    CogVideoXImageToVideoPipeline,
    WanPipeline,
    AutoencoderKLWan
)

from bitmind.generation.model_registry import ModelRegistry
from bitmind.generation.util.model import (
    load_hunyuanvideo_transformer,
    load_annimatediff_motion_adapter,
    load_vae,
    JanusWrapper,
)
from bitmind.types import ModelConfig, ModelTask


def get_text_to_image_models() -> List[ModelConfig]:
    """
    Get the list of text-to-image models.

    Returns:
        List of text-to-image model configurations
    """
    return [
        ModelConfig(
            path="stabilityai/stable-diffusion-xl-base-1.0",
            task=ModelTask.TEXT_TO_IMAGE,
            pipeline_cls=StableDiffusionXLPipeline,
            pretrained_args={
                "use_safetensors": True,
                "torch_dtype": torch.float16,
                "variant": "fp16",
            },
            use_autocast=False,
            tags=["stable-diffusion", "xl"],
        ),
        ModelConfig(
            path="SG161222/RealVisXL_V4.0",
            task=ModelTask.TEXT_TO_IMAGE,
            pipeline_cls=StableDiffusionXLPipeline,
            pretrained_args={
                "use_safetensors": True,
                "torch_dtype": torch.float16,
                "variant": "fp16",
            },
            tags=["stable-diffusion", "xl", "realistic"],
        ),
        ModelConfig(
            path="Corcelio/mobius",
            task=ModelTask.TEXT_TO_IMAGE,
            pipeline_cls=StableDiffusionXLPipeline,
            pretrained_args={"use_safetensors": True, "torch_dtype": torch.float16},
            tags=["stable-diffusion", "xl"],
        ),
        ModelConfig(
            path="black-forest-labs/FLUX.1-dev",
            task=ModelTask.TEXT_TO_IMAGE,
            pipeline_cls=FluxPipeline,
            pretrained_args={
                "use_safetensors": True,
                "torch_dtype": torch.bfloat16,
            },
            generate_args={
                "guidance_scale": 2,
                "num_inference_steps": {"min": 50, "max": 125},
                "generator": torch.Generator(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
                "resolution": [512, 768],
            },
            enable_model_cpu_offload=False,
            tags=["flux"],
        ),
        ModelConfig(
            path="prompthero/openjourney-v4",
            task=ModelTask.TEXT_TO_IMAGE,
            pipeline_cls=StableDiffusionPipeline,
            pretrained_args={
                "use_safetensors": True,
                "torch_dtype": torch.float16,
            },
            tags=["stable-diffusion", "midjourney-style"],
        ),
        ModelConfig(
            path="cagliostrolab/animagine-xl-3.1",
            task=ModelTask.TEXT_TO_IMAGE,
            pipeline_cls=StableDiffusionXLPipeline,
            pretrained_args={
                "use_safetensors": True,
                "torch_dtype": torch.float16,
            },
            tags=["stable-diffusion", "xl", "anime"],
        ),
        ModelConfig(
            path="DeepFloyd/IF",
            task=ModelTask.TEXT_TO_IMAGE,
            pipeline_cls={"stage1": IFPipeline, "stage2": IFSuperResolutionPipeline},
            pretrained_args={
                "stage1": {
                    "base": "DeepFloyd/IF-I-XL-v1.0",
                    "torch_dtype": torch.float16,
                    "variant": "fp16",
                    "clean_caption": False,
                    "watermarker": None,
                    "requires_safety_checker": False,
                },
                "stage2": {
                    "base": "DeepFloyd/IF-II-L-v1.0",
                    "torch_dtype": torch.float16,
                    "variant": "fp16",
                    "text_encoder": None,
                    "watermarker": None,
                    "requires_safety_checker": False,
                },
            },
            pipeline_stages=[
                {
                    "name": "stage1",
                    "args": {
                        "output_type": "pt",
                        "num_images_per_prompt": 1,
                        "return_dict": True,
                    },
                    "output_attr": "images",
                    "output_transform": lambda x: x[0].unsqueeze(0),
                    "save_prompt_embeds": True,
                },
                {
                    "name": "stage2",
                    "input_key": "image",
                    "args": {"output_type": "pil", "num_images_per_prompt": 1},
                    "output_attr": "images",
                    "use_prompt_embeds": True,
                },
            ],
            clear_memory_on_stage_end=True,
            tags=["deepfloyd", "multi-stage"],
        ),
        ModelConfig(
            path="deepseek-ai/Janus-Pro-7B",
            task=ModelTask.TEXT_TO_IMAGE,
            pipeline_cls=JanusWrapper,
            pretrained_args={
                "torch_dtype": torch.bfloat16,
                "use_safetensors": True,
            },
            generate_args={
                "temperature": 1.0,
                "parallel_size": 4,
                "cfg_weight": 5.0,
                "image_token_num_per_image": 576,
                "img_size": 384,
                "patch_size": 16,
            },
            use_autocast=False,
            enable_model_cpu_offload=False,
            tags=["llm-based", "multimodal"],
        ),
        ModelConfig(
            path="runwayml/stable-diffusion-v1-5-midjourney-v6",
            task=ModelTask.TEXT_TO_IMAGE,
            pipeline_cls=StableDiffusionPipeline,
            pretrained_args={
                "model_id": "runwayml/stable-diffusion-v1-5",
                "torch_dtype": torch.float16,
                "use_safetensors": True,
            },
            lora_model_id="Kvikontent/midjourney-v6",
            lora_loading_args={"use_peft_backend": True},
            use_autocast=False,
            enable_model_cpu_offload=False,
            tags=["stable-diffusion"],
        ),
        ModelConfig(
            path="THUDM/CogView4-6B",
            task=ModelTask.TEXT_TO_IMAGE,
            pipeline_cls=CogView4Pipeline,
            pretrained_args={
                "torch_dtype": torch.bfloat16,
                "use_safetensors": True,
            },
            generate_args={
                "guidance_scale": 3.5,
                "num_images_per_prompt": 1,
                "num_inference_steps": 50,
                "width": 512,
                "height": 512,
            },
            use_autocast=False,
            tags=[],
        ),
    ]


def get_image_to_image_models() -> List[ModelConfig]:
    """
    Get the list of image-to-image models.

    Returns:
        List of image-to-image model configurations
    """
    return [
        ModelConfig(
            path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            task=ModelTask.IMAGE_TO_IMAGE,
            pipeline_cls=AutoPipelineForInpainting,
            pretrained_args={
                "use_safetensors": True,
                "torch_dtype": torch.float16,
                "variant": "fp16",
            },
            generate_args={
                "guidance_scale": 7.5,
                "num_inference_steps": 50,
                "strength": 0.99,
                "generator": torch.Generator(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            },
            tags=["stable-diffusion", "xl", "inpainting"],
        ),
        ModelConfig(
            path="Lykon/dreamshaper-8-inpainting",
            task=ModelTask.IMAGE_TO_IMAGE,
            pipeline_cls=AutoPipelineForInpainting,
            pretrained_args={"torch_dtype": torch.float16, "variant": "fp16"},
            generate_args={
                "num_inference_steps": {"min": 40, "max": 60},
            },
            scheduler={"cls": DEISMultistepScheduler},
            tags=["stable-diffusion", "inpainting", "dreamshaper"],
        ),
    ]


def get_text_to_video_models() -> List[ModelConfig]:
    """
    Get the list of text-to-video models.

    Returns:
        List of text-to-video model configurations
    """
    return [
        ModelConfig(
            path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            task=ModelTask.TEXT_TO_VIDEO,
            pipeline_cls=WanPipeline,
            pretrained_args={
                "vae": (
                    load_vae,
                    {
                        "vae_cls": AutoencoderKLWan,
                        "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        "subfolder": "vae",
                        "torch_dtype": torch.float32
                    }
                ),
                "torch_dtype": torch.bfloat16
            },
            generate_args={
                "resolution": [480, 832],
                "num_frames": 81,
                "guidance_scale": 5.0
            },
            save_args={"fps": 15},
            use_autocast=False,
            tags=["wan2.1"]
        ),
        ModelConfig(
            path="tencent/HunyuanVideo",
            task=ModelTask.TEXT_TO_VIDEO,
            pipeline_cls=HunyuanVideoPipeline,
            pretrained_args={
                "model_id": "tencent/HunyuanVideo",
                "transformer": (  # custom functions supplied as tuple of (fn, args)
                    load_hunyuanvideo_transformer,
                    {
                        "model_id": "tencent/HunyuanVideo",
                        "subfolder": "transformer",
                        "torch_dtype": torch.bfloat16,
                        "revision": "refs/pr/18",
                    },
                ),
                "revision": "refs/pr/18",
                "torch_dtype": torch.bfloat16,
            },
            generate_args={
                "num_frames": {"min": 61, "max": 129},
                "resolution": {
                    "options": [
                        [720, 1280],
                        [1280, 720],
                        [1104, 832],
                        [832, 1104],
                        [960, 960],
                        [544, 960],
                        [960, 544],
                        [624, 832],
                        [832, 624],
                        [720, 720],
                    ]
                },
                "num_inference_steps": {"min": 30, "max": 50},
            },
            save_args={"fps": 30},
            use_autocast=False,
            vae_enable_tiling=True,
            tags=["high-quality", "high-resolution"],
        ),
        ModelConfig(
            path="genmo/mochi-1-preview",
            task=ModelTask.TEXT_TO_VIDEO,
            pipeline_cls=MochiPipeline,
            pretrained_args={"variant": "bf16", "torch_dtype": torch.bfloat16},
            generate_args={
                "num_frames": 84,
                "num_inference_steps": {"min": 30, "max": 65},
                "resolution": [480, 848],
            },
            save_args={"fps": 30},
            vae_enable_tiling=True,
            tags=["mochi"],
        ),
        ModelConfig(
            path="THUDM/CogVideoX-5b",
            task=ModelTask.TEXT_TO_VIDEO,
            pipeline_cls=CogVideoXPipeline,
            pretrained_args={"use_safetensors": True, "torch_dtype": torch.bfloat16},
            generate_args={
                "guidance_scale": 2,
                "num_videos_per_prompt": 1,
                "num_inference_steps": {"min": 50, "max": 125},
                "num_frames": 48,
            },
            save_args={"fps": 8},
            enable_model_cpu_offload=True,
            vae_enable_slicing=True,
            vae_enable_tiling=True,
            tags=["cogvideo"],
        ),
        ModelConfig(
            path="ByteDance/AnimateDiff-Lightning",
            task=ModelTask.TEXT_TO_VIDEO,
            pipeline_cls=AnimateDiffPipeline,
            pretrained_args={
                "model_id": "emilianJR/epiCRealism",
                "torch_dtype": torch.bfloat16,
                "motion_adapter": (load_annimatediff_motion_adapter, {"step": 4}),
            },
            generate_args={
                "guidance_scale": 2,
                "num_inference_steps": {"min": 50, "max": 125},
                "resolution": {
                    "options": [
                        [512, 512],
                        [512, 768],
                        [512, 1024],
                        [768, 512],
                        [768, 768],
                        [768, 1024],
                        [1024, 512],
                        [1024, 768],
                        [1024, 1024],
                    ]
                },
            },
            save_args={"fps": 15},
            scheduler={
                "cls": EulerDiscreteScheduler,
                "from_config_args": {
                    "timestep_spacing": "trailing",
                    "beta_schedule": "linear",
                },
            },
            tags=["animate-diff", "motion-adapter"],
        ),
    ]


def get_image_to_video_models() -> List[ModelConfig]:

    return [
        ModelConfig(
            path="THUDM/CogVideoX1.5-5B-I2V",
            task=ModelTask.IMAGE_TO_VIDEO,
            pipeline_cls=CogVideoXImageToVideoPipeline,
            pretrained_args={"use_safetensors": True, "torch_dtype": torch.bfloat16},
            generate_args={
                "guidance_scale": 2,
                "num_videos_per_prompt": 1,
                "num_inference_steps": {"min": 50, "max": 125},
                "num_frames": 49,
                "height": 768,
                "width": 768,
            },
            save_args={"fps": 8},
            enable_model_cpu_offload=True,
            vae_enable_slicing=True,
            vae_enable_tiling=True,
        )
    ]


def initialize_model_registry() -> ModelRegistry:
    """
    Initialize and populate the model registry.

    Returns:
        Fully populated ModelRegistry instance
    """
    registry = ModelRegistry()

    registry.register_all(get_text_to_image_models())
    registry.register_all(get_image_to_image_models())
    registry.register_all(get_text_to_video_models())
    registry.register_all(get_image_to_video_models())

    return registry
