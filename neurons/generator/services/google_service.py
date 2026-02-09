import os
import time
import base64
from typing import Dict, Any, Optional

import bittensor as bt

from .base_service import BaseGenerationService
from ..task_manager import GenerationTask


# Default models
DEFAULT_IMAGE_MODEL = "gemini-3-pro-image-preview"
DEFAULT_VIDEO_MODEL = "veo-3.0-generate-001"

# Valid aspect ratios for Gemini image generation
VALID_ASPECT_RATIOS = {
    "1:1", "2:3", "3:2", "3:4", "4:3",
    "4:5", "5:4", "9:16", "16:9", "21:9",
}


class GoogleService(BaseGenerationService):
    """
    Google Gemini API service for media generation.

    - Images via Gemini 3 Pro Image Preview (with C2PA content credentials
      automatically embedded and signed by Google LLC).
    - Video via Veo on Vertex AI (requires GOOGLE_CLOUD_PROJECT +
      GOOGLE_APPLICATION_CREDENTIALS in addition to GEMINI_API_KEY).

    Image generation uses the simple API-key flow (GEMINI_API_KEY).
    Video generation requires Vertex AI credentials and is only available
    when those credentials are configured.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        self.veo_client = None

        # Image client via direct Gemini API key
        if self.api_key and self.api_key.strip():
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                bt.logging.info("Google Gemini client initialized")
            except ImportError:
                bt.logging.warning(
                    "google-genai package not installed. Run: pip install google-genai"
                )
            except Exception as e:
                bt.logging.error(f"Failed to initialize Google Gemini client: {e}")
        else:
            bt.logging.warning(
                "GEMINI_API_KEY not found. Set GEMINI_API_KEY environment variable."
            )

        # Optional: Vertex AI client for Veo video generation
        self._veo_available = False
        gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")
        gcp_location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        gcp_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if gcp_project and gcp_creds:
            try:
                from google import genai as genai_vertex
                self.veo_client = genai_vertex.Client(
                    vertexai=True,
                    project=gcp_project,
                    location=gcp_location,
                )
                self._veo_available = True
                bt.logging.info(
                    f"Google Veo (Vertex AI) client initialized "
                    f"(project={gcp_project}, location={gcp_location})"
                )
            except ImportError:
                bt.logging.warning(
                    "google-genai package not installed for Vertex AI. "
                    "Run: pip install google-genai"
                )
            except Exception as e:
                bt.logging.warning(f"Failed to initialize Veo (Vertex AI) client: {e}")
        else:
            bt.logging.info(
                "Vertex AI credentials not set (GOOGLE_CLOUD_PROJECT / "
                "GOOGLE_APPLICATION_CREDENTIALS). Veo video generation disabled."
            )

    # -----------------------------------------------------------------
    # Base interface
    # -----------------------------------------------------------------
    def is_available(self) -> bool:
        """Check if Google service is available (at least image generation)."""
        return (
            self.api_key is not None
            and self.api_key.strip() != ""
            and self.client is not None
        )

    def supports_modality(self, modality: str) -> bool:
        """Check if this service supports the given modality."""
        if modality == "image":
            return True
        if modality == "video":
            return self._veo_available
        return False

    def get_supported_tasks(self) -> Dict[str, list]:
        """Return supported tasks by modality."""
        tasks: Dict[str, list] = {
            "image": ["image_generation"],
        }
        if self._veo_available:
            tasks["video"] = ["video_generation"]
        else:
            tasks["video"] = []
        return tasks

    def get_api_key_requirements(self) -> Dict[str, str]:
        """Return Google API key requirements."""
        return {
            "GEMINI_API_KEY": (
                "Google Gemini API key for image generation (C2PA-signed). "
                "Get one at https://aistudio.google.com/apikey"
            ),
        }

    # -----------------------------------------------------------------
    # Dispatch
    # -----------------------------------------------------------------
    def process(self, task: GenerationTask) -> Dict[str, Any]:
        """Dispatch task by modality."""
        if task.modality == "image":
            return self._generate_image(task)
        elif task.modality == "video":
            return self._generate_video(task)
        else:
            raise ValueError(f"Unsupported modality: {task.modality}")

    # -----------------------------------------------------------------
    # Image generation  (Gemini 3 Pro Image Preview)
    # -----------------------------------------------------------------
    def _generate_image(self, task: GenerationTask) -> Dict[str, Any]:
        """
        Generate an image using Google Gemini and return raw bytes with
        C2PA content credentials intact.
        """
        if self.client is None:
            raise RuntimeError("Google Gemini client not initialized")

        try:
            from google.genai import types
        except ImportError:
            raise RuntimeError(
                "google-genai package not installed. Run: pip install google-genai"
            )

        try:
            params = task.parameters or {}
            model = params.get("model", DEFAULT_IMAGE_MODEL)
            aspect_ratio = params.get("aspect_ratio")

            bt.logging.info(
                f"Generating image with Google Gemini: "
                f"model={model}, prompt={task.prompt[:60]}..."
            )

            # Build image config if aspect ratio requested
            image_config = None
            if aspect_ratio and aspect_ratio in VALID_ASPECT_RATIOS:
                image_config = types.ImageConfig(aspect_ratio=aspect_ratio)

            config_kwargs: Dict[str, Any] = {
                "response_modalities": ["IMAGE"],
            }
            if image_config is not None:
                config_kwargs["image_config"] = image_config

            start_time = time.time()

            response = self.client.models.generate_content(
                model=model,
                contents=[task.prompt],
                config=types.GenerateContentConfig(**config_kwargs),
            )

            gen_time = time.time() - start_time

            # Extract raw image bytes -- preserve C2PA metadata
            for part in response.parts:
                if part.inline_data is not None:
                    image_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type or "image/png"

                    bt.logging.success(
                        f"Google Gemini generated {len(image_data)} bytes "
                        f"(mime={mime_type}) with C2PA metadata in {gen_time:.2f}s"
                    )

                    return {
                        "data": image_data,
                        "metadata": {
                            "model": model,
                            "provider": "google",
                            "mime_type": mime_type,
                            "generation_time": gen_time,
                        },
                    }

            # No image part found in response
            text_parts = [p.text for p in response.parts if p.text]
            raise RuntimeError(
                f"Gemini returned no image data. "
                f"Text response: {' '.join(text_parts)[:200]}"
            )

        except Exception as e:
            bt.logging.error(f"Google image generation failed: {e}")
            raise

    # -----------------------------------------------------------------
    # Video generation  (Veo via Vertex AI)
    # -----------------------------------------------------------------
    def _generate_video(self, task: GenerationTask) -> Dict[str, Any]:
        """
        Generate a video using Google Veo on Vertex AI.

        Requires GOOGLE_CLOUD_PROJECT and GOOGLE_APPLICATION_CREDENTIALS
        environment variables to be set.
        """
        if not self._veo_available or self.veo_client is None:
            raise RuntimeError(
                "Veo video generation requires Vertex AI credentials. "
                "Set GOOGLE_CLOUD_PROJECT and GOOGLE_APPLICATION_CREDENTIALS."
            )

        try:
            from google.genai import types
        except ImportError:
            raise RuntimeError(
                "google-genai package not installed. Run: pip install google-genai"
            )

        params: Dict[str, Any] = task.parameters or {}
        model: str = params.get("model", DEFAULT_VIDEO_MODEL)
        aspect_ratio: str = params.get("aspect_ratio", "16:9")
        duration_seconds: int = int(params.get("duration", params.get("seconds", 4)))
        poll_interval: float = float(params.get("poll_interval", 10.0))
        timeout: float = float(params.get("timeout", 600.0))

        bt.logging.info(
            f"Veo video generation requested: model={model}, "
            f"aspect_ratio={aspect_ratio}, duration={duration_seconds}s, "
            f"prompt={task.prompt[:60]!r}"
        )

        try:
            start_time = time.monotonic()

            operation = self.veo_client.models.generate_videos(
                model=model,
                prompt=task.prompt,
                config=types.GenerateVideosConfig(
                    aspect_ratio=aspect_ratio,
                    number_of_videos=1,
                ),
            )

            bt.logging.info("Veo job submitted, polling for completion...")

            # Poll until complete or timeout
            while not operation.done:
                elapsed = time.monotonic() - start_time
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Veo video generation timed out after {timeout}s"
                    )
                bt.logging.info(
                    f"Veo job in progress... elapsed={elapsed:.1f}s"
                )
                time.sleep(poll_interval)
                operation = self.veo_client.operations.get(operation)

            # Retrieve result
            result = operation.result
            if not result or not result.generated_videos:
                raise RuntimeError("Veo returned no generated videos")

            video = result.generated_videos[0]
            video_uri = video.video.uri

            bt.logging.info(f"Veo video ready at URI: {video_uri}")

            # Download the video bytes from GCS
            import requests as _requests
            video_response = _requests.get(video_uri, timeout=120)
            video_response.raise_for_status()
            video_bytes = video_response.content

            gen_time = time.monotonic() - start_time

            bt.logging.success(
                f"Downloaded Veo video: {len(video_bytes)} bytes in {gen_time:.1f}s"
            )

            return {
                "data": video_bytes,
                "metadata": {
                    "model": model,
                    "provider": "google",
                    "mime_type": "video/mp4",
                    "aspect_ratio": aspect_ratio,
                    "duration_seconds": duration_seconds,
                    "generation_time": gen_time,
                    "video_uri": video_uri,
                },
            }

        except Exception as e:
            bt.logging.error(f"Google Veo video generation failed: {e}")
            raise

    # -----------------------------------------------------------------
    # Service info
    # -----------------------------------------------------------------
    def get_service_info(self) -> Dict[str, Any]:
        """Return information about this service."""
        return {
            "name": "Google",
            "type": "api",
            "provider": "google",
            "available": self.is_available(),
            "supported_tasks": self.get_supported_tasks(),
            "veo_available": self._veo_available,
            "default_image_model": DEFAULT_IMAGE_MODEL,
            "default_video_model": DEFAULT_VIDEO_MODEL,
        }
