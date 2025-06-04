import gc
import json
import random
import time
import asyncio
from pathlib import Path
from typing import Dict, Optional, Any, Union, List
import traceback

import bittensor as bt
import numpy as np
import torch
from diffusers.utils import export_to_video
from PIL import Image

from bitmind.types import CacheConfig, ModelTask
from bitmind.generation.util.image import create_random_mask, is_black_output
from bitmind.generation.util.prompt import truncate_prompt_if_too_long
from bitmind.generation.prompt_generator import PromptGenerator
from bitmind.generation.util.model import (
    create_pipeline_generator,
    enable_model_optimizations,
)
from bitmind.generation.model_registry import ModelRegistry
from bitmind.generation.models import initialize_model_registry

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

IMAGE_ANNOTATION_MODEL: str = "Salesforce/blip2-opt-6.7b-coco"
TEXT_MODERATION_MODEL: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"


class GenerationPipeline:
    """
    A class for generating synthetic images and videos.

    This class supports different prompt generation strategies and can utilize
    various text-to-video (t2v), text-to-image (t2i), and image-to-image (i2i) models.

    Attributes:
        model_name: Name of the specific model to use (if not random)
        model_registry: Registry of available models
        output_dir: Directory to write generated data
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        model_registry: Optional[ModelRegistry] = None,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the SyntheticDataGenerator.

        Args:
            model_name: Name of the generative image/video model
            output_dir: Directory to write generated data
            model_registry: Optional ModelRegistry instance
            device: Device identifier

        Raises:
            ValueError: If an invalid model name or configuration is provided
        """
        self.output_dir = Path(output_dir)
        self.model_registry = model_registry or initialize_model_registry()
        self.device = device
        self.loop = asyncio.get_event_loop()

        self.prompt_generator = PromptGenerator(
            vlm_name=IMAGE_ANNOTATION_MODEL, llm_name=TEXT_MODERATION_MODEL
        )

    def generate(
        self,
        image_samples: List[dict],
        tasks: Optional[Union[str, List[str]]] = None,
        model_names: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate synthetic data based on input parameters.

        Args:
            image_samples: Image samples returned by ImageSampler
            task: Optional task type.
            model_name: Optional model name.

        Returns:
            Dictionary containing generated data information.

        Raises:
            ValueError: If image is None and cannot be sampled.
        """
        bt.logging.info(f"---------- Starting Generation ----------")
        prompts = self.generate_prompts(image_samples, downstream_tasks=tasks, clear_gpu=True)
        paths, stats = self.generate_media(prompts, model_names, image_samples, tasks)

        def log_stats(stats):
            model_names = list(stats.keys())
            total_successes = sum([stats[name]["success"] for name in model_names])

            if total_successes == 0:
                log_fn = bt.logging.error
            elif total_successes == len(image_samples) * len(model_names):
                log_fn = bt.logging.success
            else:
                log_fn = bt.logging.warning

            log_fn(json.dumps(stats, indent=2))

        log_stats(stats)
        bt.logging.info(f"---------- Generation Complete ----------")
        return paths

    def generate_prompts(
        self,
        image_samples: List[dict],
        downstream_tasks: Optional[List[str]] = None,
        clear_gpu: bool = True,
    ) -> str:
        """
        Generate a prompts based on input images and downstream tasks.
        """
        if downstream_tasks is None:
            downstream_tasks = [
                ModelTask.TEXT_TO_IMAGE.value,
                ModelTask.TEXT_TO_VIDEO.value,
                ModelTask.IMAGE_TO_IMAGE.value,
                ModelTask.IMAGE_TO_VIDEO.value,
            ]

        k = len(image_samples)
        bt.logging.info(f"Generating {k} prompt{'s' if k > 1 else ''}")

        self.prompt_generator.load_models()

        # organize prompts in a dict to avoid failed prompt generations causing misaligned images/prompts
        prompts = {task: {} for task in downstream_tasks}
        for i in range(k):
            image_path = image_samples[i].get('path')
            image = image_samples[i].get('image')
            for task in downstream_tasks:
                try:
                    prompts[task][i] = self.prompt_generator.generate(
                        image, downstream_task=task
                    )

                    bt.logging.info(f"Generated prompt {i+1}/{k}: {prompts[task][i]} from {image_path}")
                except Exception as e:
                    prompts[task][i] = None
                    bt.logging.error(f"Error generating prompt for image {i+1}: {e} from {image_path}")
                    bt.logging.error(traceback.format_exc())
                    continue

        if clear_gpu:
            self.prompt_generator.clear_gpu()

        return prompts

    def generate_media(
        self,
        prompts: Union[dict, str],
        model_names: Optional[Union[str, List[str]]] = None,
        image_samples: List[dict] = None,
        tasks: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate synthetic data based on a text prompt.

        Args:
            prompt: The text prompt used for generation, or a dictionary with
                the outer key of generation task type, inner key of the image index,
                and value of the prompt
            task: The generation task type ('t2i', 't2v', 'i2i', or None).
            model_name: Optional model name to use for generation.
            image: Optional input image for image-to-image generation.

        Returns:
            Dictionary containing generated data and metadata.

        Raises:
            RuntimeError: If generation fails.
        """
        model_names = self._validate_model_names(model_names, tasks)

        if isinstance(prompts, str):
            prompts = [prompts]

        n_models = len(model_names)
        n_prompts = len(prompts)

        stats = {model_name: {"total": 0, "success": 0} for model_name in model_names}
        save_paths = []

        for model_idx, model_name in enumerate(model_names):
            modality = self.model_registry.get_modality(model_name)
            task = self.model_registry.get_task(model_name)

            if isinstance(prompts, list):
                task_prompts = {i: p for i, p in enumerate(prompts)}
            else:
                # task-specific prompts (motion enhancement for video)
                task_prompts = prompts[task]

            for prompt_idx in task_prompts:
                stats[model_name]["total"] += 1
                bt.logging.info(
                    f"Starting batch | Model {model_idx+1}/{n_models} | Prompt {prompt_idx+1}/{n_prompts}"
                )
                bt.logging.info(f"  Model: {model_name}")
                bt.logging.info(f"  Prompt: {task_prompts[prompt_idx]}")

                try:
                    image = None
                    if image_samples is not None and len(image_samples) > prompt_idx:
                        image = image_samples[prompt_idx].get('image')

                    # 3 retries for black (NSFW filtered) output
                    for _ in range(3):
                        gen_output = self._generate_media_with_model(
                            model_name, task_prompts[prompt_idx], image
                        )
                        if is_black_output(modality, gen_output):
                            # sanitize and retry
                            self.clear_gpu()
                            self.prompt_generator.load_llm()
                            task_prompts[prompt_idx] = self.prompt_generator.sanitize(
                                task_prompts[prompt_idx]
                            )
                            self.prompt_generator.clear_gpu()
                        else:
                            break

                    bt.logging.info(
                        {
                            k: v
                            for k, v in gen_output.items()
                            if k not in (modality, "source_image", "mask_image")
                        }
                    )
                    save_paths.append(self._save_media_and_metadata(gen_output))
                    stats[model_name]["success"] += 1
                except Exception as e:
                    bt.logging.error(f"Failed to either generate or save media: {e}")
                    bt.logging.error(f"  Model: {model_name}")
                    bt.logging.error(f"  Prompt: {task_prompts[prompt_idx]}")
                    bt.logging.error(traceback.format_exc())

        return save_paths, stats

    def _load_model(
        self,
        model_name: Optional[str] = None,
    ) -> None:
        bt.logging.info(f"Loading {model_name}")
        try:
            model_config = self.model_registry.get_model_dict(model_name)
            bt.logging.info(
                json.dumps({k: str(v) for k, v in model_config.items()}, indent=2)
            )

            pipeline_cls = model_config["pipeline_cls"]
            pipeline_args = model_config.get("from_pretrained_args", {}).copy()

            # Handle custom loading functions passed as tuples
            for k, v in pipeline_args.items():
                if isinstance(v, tuple) and callable(v[0]):
                    pipeline_args[k] = v[0](**v[1])

            model_id = pipeline_args.pop("model_id", model_name)

            if isinstance(pipeline_cls, dict):
                # Multi-stage pipeline
                MODEL = {}
                for stage_name, stage_cls in pipeline_cls.items():
                    stage_args = pipeline_args.get(stage_name, {})
                    base_model = stage_args.get("base", model_id)
                    stage_args_filtered = {
                        k: v for k, v in stage_args.items() if k != "base"
                    }

                    bt.logging.debug(f"Loading {stage_name} from {base_model}")
                    MODEL[stage_name] = stage_cls.from_pretrained(
                        base_model,
                        **stage_args_filtered,
                        add_watermarker=False,
                    )

                    enable_model_optimizations(
                        model=MODEL[stage_name],
                        device=self.device,
                        enable_cpu_offload=model_config.get(
                            "enable_model_cpu_offload", False
                        ),
                        enable_sequential_cpu_offload=model_config.get(
                            "enable_sequential_cpu_offload", False
                        ),
                        enable_vae_slicing=model_config.get(
                            "vae_enable_slicing", False
                        ),
                        enable_vae_tiling=model_config.get("vae_enable_tiling", False),
                        stage_name=stage_name,
                    )

                    MODEL[stage_name].watermarker = None
            else:
                # Single-stage pipeline
                MODEL = pipeline_cls.from_pretrained(
                    model_id,
                    **pipeline_args,
                    add_watermarker=False,
                )

                # Load LoRA weights if specified
                if "lora_model_id" in model_config:
                    bt.logging.info(
                        f"Loading LoRA weights from {model_config['lora_model_id']}"
                    )
                    lora_loading_args = model_config.get("lora_loading_args", {})
                    self.model.load_lora_weights(
                        model_config["lora_model_id"], **lora_loading_args
                    )

                # Load scheduler if specified
                scheduler_config = model_config.get("scheduler", {})
                if scheduler_config:
                    sched_cls = scheduler_config["cls"]
                    sched_args = scheduler_config.get("from_config_args", {})
                    MODEL.scheduler = sched_cls.from_config(
                        MODEL.scheduler.config, **sched_args
                    )

                enable_model_optimizations(
                    model=MODEL,
                    device=self.device,
                    enable_cpu_offload=model_config.get(
                        "enable_model_cpu_offload", False
                    ),
                    enable_sequential_cpu_offload=model_config.get(
                        "enable_sequential_cpu_offload", False
                    ),
                    enable_vae_slicing=model_config.get("vae_enable_slicing", False),
                    enable_vae_tiling=model_config.get("vae_enable_tiling", False),
                )
                MODEL.watermarker = None

            self.model = MODEL
            bt.logging.info(f"Loaded {model_name}")
            return True

        except Exception as e:
            bt.logging.error(f"Error loading model: {model_name}")
            bt.logging.error(traceback.format_exc())
            return False

    def _generate_media_with_model(self, model_name, prompt, image):
        model_config = self.model_registry.get_model_dict(model_name)
        task = self.model_registry.get_task(model_name)

        if task == "i2i" and image is None:
            raise ValueError(
                "An image must be provided for image-to-image model {model_name}"
            )

        if not self._load_model(model_name):
            raise RuntimeError(f"Failed to load {model_name}")
            return {}

        bt.logging.debug("Preparing generation arguments")
        gen_args = model_config.get("generate_args", {}).copy()
        mask_center = None

        # prep inptask-specific generation args
        if task == "i2i":
            if image is None:
                raise ValueError("image cannot be None for image-to-image generation")
            image = Image.fromarray(image)
            target_size = (1024, 1024)
            if image.size[0] > target_size[0] or image.size[1] > target_size[1]:
                image = image.resize(target_size, Image.Resampling.LANCZOS)

            gen_args["mask_image"], mask_center = create_random_mask(image.size)
            gen_args["image"] = image

        elif task == "i2v":
            if image is None:
                raise ValueError("image cannot be None for image-to-video generation")
            image = Image.fromarray(image)
            # Get target size from gen_args if specified, otherwise use default
            target_size = (gen_args.get("height", 768), gen_args.get("width", 768))
            if image.size[0] > target_size[0] or image.size[1] > target_size[1]:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            gen_args["image"] = image

        # Prepare generation arguments
        for k, v in gen_args.items():
            if isinstance(v, dict):
                if "min" in v and "max" in v:
                    # For i2v, use minimum values to save memory
                    if task == "i2v":
                        gen_args[k] = v["min"]
                    else:
                        gen_args[k] = np.random.randint(v["min"], v["max"])

                if "options" in v:
                    gen_args[k] = random.choice(v["options"])

        if "resolution" in gen_args:
            gen_args["height"] = gen_args["resolution"][0]
            gen_args["width"] = gen_args["resolution"][1]
            del gen_args["resolution"]

        truncated_prompt = truncate_prompt_if_too_long(prompt, self.model)
        bt.logging.debug(f"Generating media from prompt: {truncated_prompt}")
        bt.logging.debug(f"Generation args: {gen_args}")

        generate_fn = create_pipeline_generator(model_config, self.model)

        torch.cuda.empty_cache()
        gc.collect()

        start_time = time.time()

        bt.logging.debug("Generating media")

        if model_config.get("use_autocast", True):
            pretrained_args = model_config.get("from_pretrained_args", {})
            torch_dtype = pretrained_args.get("torch_dtype", torch.bfloat16)

            with torch.autocast(self.device, torch_dtype, cache_enabled=False):
                gen_output = generate_fn(truncated_prompt, **gen_args)
        else:
            gen_output = generate_fn(truncated_prompt, **gen_args)

        gen_time = time.time() - start_time

        hours = int(gen_time // 3600)
        minutes = int((gen_time % 3600) // 60)
        seconds = int(gen_time % 60)
        bt.logging.info(
            f"Finished generation in {hours:02d}:{minutes:02d}:{seconds:02d}"
        )

        modality = self.model_registry.get_modality(model_name)
        media_type = self.model_registry.get_output_media_type(model_name)
        output = {
            modality: gen_output,  # image or video
            "modality": modality,
            "media_type": media_type,
            "prompt": truncated_prompt,
            "model_name": model_name,
            "time": time.time(),
            "gen_duration": gen_time,
            "mask_center": mask_center,
        }
        for k in ["num_inference_steps", "guidance_scale", "resolution"]:
            output[k] = gen_args.get(k, "")

        source_image = gen_args.get("image", None)
        if source_image is not None:
            output["source_image"] = source_image

        mask_image = gen_args.get("mask_image", None)
        if mask_image is not None:
            output["mask_image"] = mask_image

        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        return output

    def _validate_model_names(self, model_names, tasks) -> str:
        if model_names is None:
            if tasks is None:
                model_names = self.model_registry.get_interleaved_model_names()
            else:
                tasks = [tasks] if not isinstance(tasks, (list, tuple)) else tasks
                model_names = self.model_registry.get_model_names_by_task(tasks)

        elif isinstance(model_names, str):
            model_names = [model_names]

        invalid_models = [
            name for name in model_names if name not in self.model_registry.model_names
        ]
        if invalid_models:
            raise ValueError(
                f"Invalid model names {invalid_models}. "
                f"Options are {self.model_registry.model_names}"
            )
        return model_names

    def _save_media_and_metadata(self, media_sample):
        modality = media_sample["modality"]
        media_type = media_sample["media_type"]
        model_name = media_sample["model_name"]

        ouptput_dir = (
            CacheConfig(
                base_dir=self.output_dir, modality=modality, media_type=media_type
            ).get_path()
            / model_name.split("/")[1]
        )

        ouptput_dir.mkdir(parents=True, exist_ok=True)
        base_path = ouptput_dir / str(media_sample["time"])
        bt.logging.debug(f"[{modality}:{media_type}] Writing to cache")

        metadata = {
            k: v
            for k, v in media_sample.items()
            if k not in (modality, "source_image", "mask_image")
        }

        if modality == "image":
            save_path = str(base_path.with_suffix(".png"))
            media_sample[modality].images[0].save(save_path)
            if "mask_image" in media_sample:
                mask_path = str(save_path).replace(".png", "_mask.npy")
                metadata["mask_path"] = mask_path
                np.save(mask_path, np.array(media_sample["mask_image"]))
        elif modality == "video":
            save_path = str(base_path.with_suffix(".mp4"))
            export_to_video(media_sample[modality].frames[0], save_path, fps=30)

        base_path.with_suffix(".json").write_text(json.dumps(metadata))

        bt.logging.info(f"Wrote to {save_path}")
        return save_path

    def clear_gpu(self):
        if hasattr(self, "model") and self.model is not None:
            bt.logging.trace("Deleting model")
            if isinstance(self.model, dict):
                for stage_name, stage_model in self.model.items():
                    del stage_model
            else:
                del self.model

        if hasattr(self, "prompt_generator"):
            bt.logging.trace("Deleting prompt generator")
            self.prompt_generator.clear_gpu()

        gc.collect()
        if torch.cuda.is_available():
            bt.logging.trace("Clearing CUDA cache")
            torch.cuda.empty_cache()

    def shutdown(self):
        self.clear_gpu()
