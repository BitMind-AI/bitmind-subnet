import os
import time
import base64
import requests
import bittensor as bt
from typing import Dict, Any, Optional

from .base_service import BaseGenerationService
from ..task_manager import GenerationTask


class Models:
    """Symbolic constants for Runway models."""

    GEN3A_TURBO = "gen3a_turbo"
    GEN4_5 = "gen4.5"


MODEL_INFO = {
    Models.GEN3A_TURBO: {
        "name": "Gen-3 Alpha Turbo",
        "description": "Gen-3 Alpha Turbo - fast and cost-effective",
        "max_duration": 10,
    },
    Models.GEN4_5: {
        "name": "Gen-4.5",
        "description": "Latest Gen-4.5 - highest quality",
        "max_duration": 10,
    },
}


class RunwayService(BaseGenerationService):
    """
    Runway ML API service for video generation using Gen-4 Turbo.

    This enables generative miners to participate in video generation tasks
    without requiring local GPU infrastructure.

    Features:
    - Text-to-video generation
    - Image-to-video (animate still images)
    - Multiple aspect ratios and durations
    """

    VALID_RATIOS = {"1280:768", "768:1280"}
    VALID_DURATIONS = {5, 10}

    def __init__(self, config: Any = None):
        super().__init__(config)

        self.api_key = os.getenv("RUNWAYML_API_SECRET")
        self.base_url = "https://api.dev.runwayml.com/v1"
        self.api_version = "2024-11-06"
        self.timeout = 300
        self.poll_interval = 5.0

        # Read configurable options from environment
        model_env = os.getenv("RUNWAY_MODEL", Models.GEN3A_TURBO)
        self.default_model = model_env if model_env in MODEL_INFO else Models.GEN3A_TURBO

        duration_env = os.getenv("RUNWAY_DEFAULT_DURATION", "5")
        try:
            self.default_duration = int(duration_env)
            if self.default_duration not in self.VALID_DURATIONS:
                self.default_duration = 5
        except ValueError:
            self.default_duration = 5

        if not self.api_key:
            bt.logging.warning("RUNWAYML_API_SECRET not found.")
        else:
            bt.logging.info(f"RunwayService initialized: model={self.default_model}, duration={self.default_duration}s")

    def is_available(self) -> bool:
        return self.api_key is not None and self.api_key.strip() != ""

    def supports_modality(self, modality: str) -> bool:
        return modality == "video"

    def get_supported_tasks(self) -> Dict[str, list]:
        return {
            "image": [],
            "video": ["video_generation", "image_to_video"],
        }

    def get_api_key_requirements(self) -> Dict[str, str]:
        return {
            "RUNWAYML_API_SECRET": "Runway ML API secret (https://dev.runwayml.com/)"
        }

    def process(self, task: GenerationTask) -> Dict[str, Any]:
        if task.modality != "video":
            raise ValueError(f"RunwayService does not support modality: {task.modality}")

        return self._generate_video(task)

    def _generate_video(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate a video using Runway API."""
        try:
            params = task.parameters or {}
            prompt = task.prompt

            model = params.get("model", self.default_model)
            if model not in MODEL_INFO:
                bt.logging.warning(f"Unknown model {model}, using {self.default_model}")
                model = self.default_model

            duration = params.get("duration", self.default_duration)
            if duration not in self.VALID_DURATIONS:
                duration = self.default_duration

            ratio = self._get_aspect_ratio(
                params.get("width", 1280),
                params.get("height", 720)
            )

            bt.logging.info(f"Runway generating video: model={model}, duration={duration}s, ratio={ratio}")

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Runway-Version": self.api_version,
            }

            # Check for input image (URL or bytes)
            input_image_url = params.get("input_image_url")
            
            if task.input_data:
                endpoint = f"{self.base_url}/image_to_video"
                payload = {
                    "model": model,
                    "promptImage": self._encode_image(task.input_data),
                    "promptText": prompt,
                    "ratio": ratio,
                    "duration": duration,
                }
            elif input_image_url:
                endpoint = f"{self.base_url}/image_to_video"
                payload = {
                    "model": model,
                    "promptImage": input_image_url,
                    "promptText": prompt,
                    "ratio": ratio,
                    "duration": duration,
                }
            else:
                endpoint = f"{self.base_url}/text_to_video"
                payload = {
                    "model": model,
                    "promptText": prompt,
                    "ratio": ratio,
                    "duration": duration,
                }

            start_time = time.time()
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code != 200:
                raise RuntimeError(f"Runway API error {response.status_code}: {response.text}")

            task_data = response.json()
            task_id = task_data.get("id")

            if not task_id:
                raise RuntimeError("No task ID returned from Runway API")

            bt.logging.info(f"Runway task created: {task_id}")

            video_data = self._poll_for_completion(task_id, headers)
            gen_time = time.time() - start_time

            bt.logging.success(f"Runway video generated: {len(video_data)} bytes in {gen_time:.1f}s")

            return {
                "data": video_data,
                "metadata": {
                    "model": model,
                    "provider": "runway",
                    "duration": duration,
                    "ratio": ratio,
                    "generation_time": gen_time,
                }
            }

        except Exception as e:
            bt.logging.error(f"Runway video generation failed: {e}")
            raise

    def _poll_for_completion(self, task_id: str, headers: Dict[str, str]) -> bytes:
        """Poll Runway API until task completes and return video bytes."""
        status_url = f"{self.base_url}/tasks/{task_id}"
        start_time = time.time()

        while True:
            if time.time() - start_time > self.timeout:
                try:
                    requests.delete(status_url, headers=headers, timeout=10)
                except Exception:
                    pass
                raise TimeoutError(f"Video generation timed out after {self.timeout}s")

            response = requests.get(status_url, headers=headers, timeout=30)

            if response.status_code != 200:
                bt.logging.warning(f"Failed to get task status: {response.status_code}")
                time.sleep(self.poll_interval)
                continue

            task_status = response.json()
            status = task_status.get("status")

            if status == "SUCCEEDED":
                output = task_status.get("output", [])
                if not output:
                    raise RuntimeError("No output in completed task")

                video_url = output[0] if isinstance(output, list) else output

                video_response = requests.get(video_url, timeout=120)
                video_response.raise_for_status()
                return video_response.content

            elif status == "FAILED":
                error = task_status.get("failure", "Unknown error")
                raise RuntimeError(f"Video generation failed: {error}")

            elif status == "CANCELLED":
                raise RuntimeError("Video generation was cancelled")

            time.sleep(self.poll_interval)

    def _get_aspect_ratio(self, width: int, height: int) -> str:
        """Convert dimensions to Runway aspect ratio string."""
        if height <= 0:
            return "1280:768"  # Default to landscape
        ratio = width / height

        # gen3a_turbo only supports 768:1280 and 1280:768
        if ratio >= 1:
            return "1280:768"  # Landscape
        else:
            return "768:1280"  # Portrait

    def _encode_image(self, image_data: bytes) -> str:
        """Encode image bytes to data URI for Runway API."""
        if image_data[:8] == b'\x89PNG\r\n\x1a\n':
            mime_type = "image/png"
        elif image_data[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        elif image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
            mime_type = "image/webp"
        else:
            mime_type = "image/png"

        base64_data = base64.b64encode(image_data).decode('utf-8')
        return f"data:{mime_type};base64,{base64_data}"

    def get_service_info(self) -> Dict[str, Any]:
        """Return information about this service."""
        return {
            "name": "Runway",
            "type": "api",
            "provider": "api.dev.runwayml.com",
            "available": self.is_available(),
            "supported_tasks": self.get_supported_tasks(),
            "default_model": self.default_model,
        }
