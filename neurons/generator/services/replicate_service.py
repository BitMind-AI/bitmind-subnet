import os
import time
import requests
import bittensor as bt
from typing import Dict, Any, Optional

from .base_service import BaseGenerationService
from ..task_manager import GenerationTask


class Models:
    """Symbolic constants for Replicate models."""

    FLUX_SCHNELL = "flux-schnell"
    FLUX_DEV = "flux-dev"
    FLUX_PRO = "flux-pro"
    SDXL = "sdxl"


MODEL_INFO = {
    Models.FLUX_SCHNELL: {
        "version": "black-forest-labs/flux-schnell",
        "name": "FLUX.1 Schnell",
        "family": "flux",
        "supports_negative_prompt": False,
    },
    Models.FLUX_DEV: {
        "version": "black-forest-labs/flux-dev",
        "name": "FLUX.1 Dev",
        "family": "flux",
        "supports_negative_prompt": False,
    },
    Models.FLUX_PRO: {
        "version": "black-forest-labs/flux-pro",
        "name": "FLUX.1 Pro",
        "family": "flux",
        "supports_negative_prompt": False,
    },
    Models.SDXL: {
        "version": "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        "name": "Stable Diffusion XL",
        "family": "sdxl",
        "supports_negative_prompt": True,
    },
}


class ReplicateService(BaseGenerationService):
    """
    Replicate API service for FLUX and SDXL image generation.

    Features:
    - FLUX.1 models (Schnell, Dev, Pro)
    - SDXL support
    - Async prediction with polling
    - Multiple aspect ratio support
    """

    API_BASE = "https://api.replicate.com/v1"
    POLL_INTERVAL = 1.0
    MAX_POLL_TIME = 120

    def __init__(self, config: Any = None):
        super().__init__(config)

        self.api_key = os.getenv("REPLICATE_API_TOKEN")
        self.timeout = 30
        self.default_model = Models.FLUX_SCHNELL

        if not self.api_key:
            bt.logging.warning("REPLICATE_API_TOKEN not found.")
        else:
            bt.logging.info("ReplicateService initialized with API token")

    # ---------------------------------------------------------------------
    # Base methods
    # ---------------------------------------------------------------------
    def is_available(self) -> bool:
        return self.api_key is not None and self.api_key.strip() != ""

    def supports_modality(self, modality: str) -> bool:
        return modality == "image"

    def get_supported_tasks(self) -> Dict[str, list]:
        return {
            "image": ["image_generation"],
            "video": []
        }

    def get_api_key_requirements(self) -> Dict[str, str]:
        return {"REPLICATE_API_TOKEN": "API token for Replicate image generation"}

    # ---------------------------------------------------------------------
    # Processing logic
    # ---------------------------------------------------------------------
    def process(self, task: GenerationTask) -> Dict[str, Any]:
        if task.modality != "image":
            raise ValueError(f"ReplicateService does not support modality: {task.modality}")

        return self._generate_image(task)

    # ---------------------------------------------------------------------
    # API helpers
    # ---------------------------------------------------------------------
    def _create_prediction(self, model_version: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.API_BASE}/predictions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if "/" in model_version and ":" not in model_version:
            payload = {"model": model_version, "input": input_data}
        else:
            version_id = model_version.split(":")[-1] if ":" in model_version else model_version
            payload = {"version": version_id, "input": input_data}

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

        if response.status_code == 201:
            return response.json()
        else:
            raise RuntimeError(f"Replicate API error {response.status_code}: {response.text}")

    def _poll_prediction(self, prediction_url: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        start_time = time.time()

        while time.time() - start_time < self.MAX_POLL_TIME:
            response = requests.get(prediction_url, headers=headers, timeout=self.timeout)

            if response.status_code != 200:
                raise RuntimeError(f"Replicate poll error {response.status_code}: {response.text}")

            result = response.json()
            status = result.get("status")

            if status == "succeeded":
                return result
            elif status == "failed":
                error = result.get("error", "Unknown error")
                raise RuntimeError(f"Replicate prediction failed: {error}")
            elif status == "canceled":
                raise RuntimeError("Replicate prediction was canceled")

            time.sleep(self.POLL_INTERVAL)

        raise RuntimeError(f"Replicate prediction timed out after {self.MAX_POLL_TIME}s")

    def _download_image(self, url: str) -> bytes:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.content

    # ---------------------------------------------------------------------
    # Image generation core
    # ---------------------------------------------------------------------
    def _generate_image(self, task: GenerationTask) -> Dict[str, Any]:
        try:
            params = task.parameters or {}
            model = params.get("model", self.default_model)
            prompt = task.prompt

            bt.logging.info(f"Replicate generating image with model={model}")

            if model not in MODEL_INFO:
                raise ValueError(f"Unknown Replicate model: {model}. "
                                f"Available models: {list(MODEL_INFO.keys())}")

            model_info = MODEL_INFO[model]
            model_version = model_info["version"]

            input_data = {"prompt": prompt}

            if model_info.get("supports_negative_prompt") and "negative_prompt" in params:
                input_data["negative_prompt"] = params["negative_prompt"]

            if "width" in params:
                input_data["width"] = params["width"]
            if "height" in params:
                input_data["height"] = params["height"]
            if "num_inference_steps" in params:
                input_data["num_inference_steps"] = params["num_inference_steps"]
            if "guidance_scale" in params:
                input_data["guidance_scale"] = params["guidance_scale"]
            if "seed" in params:
                input_data["seed"] = params["seed"]
            if "aspect_ratio" in params:
                input_data["aspect_ratio"] = params["aspect_ratio"]

            start_time = time.time()

            prediction = self._create_prediction(model_version, input_data)
            prediction_url = prediction.get("urls", {}).get("get")

            if not prediction_url:
                raise RuntimeError("No prediction URL returned from Replicate")

            bt.logging.info("Replicate prediction created, polling for result...")

            result = self._poll_prediction(prediction_url)
            gen_time = time.time() - start_time

            output = result.get("output")
            if not output:
                raise RuntimeError("No output returned from Replicate")

            image_url = output[0] if isinstance(output, list) else output

            bt.logging.info(f"Replicate generated image in {gen_time:.2f}s, downloading...")

            img_bytes = self._download_image(image_url)

            bt.logging.success(f"Downloaded {len(img_bytes)} bytes from Replicate")

            return {
                "data": img_bytes,
                "metadata": {
                    "model": model,
                    "provider": "replicate",
                    "generation_time": round(gen_time, 2),
                    "prediction_id": result.get("id"),
                }
            }

        except Exception as e:
            bt.logging.error(f"Replicate image generation failed: {e}")
            raise

    def get_service_info(self) -> Dict[str, Any]:
        return {
            "name": "Replicate",
            "type": "api",
            "provider": "api.replicate.com",
            "available": self.is_available(),
            "supported_tasks": self.get_supported_tasks(),
            "default_model": self.default_model
        }
