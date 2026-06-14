import os
import time
import base64
import io
from typing import Dict, Any, Optional, List

import requests
import bittensor as bt
from PIL import Image

from .base_service import BaseGenerationService, CheckpointFn
from ..task_manager import GenerationTask

CHECKPOINT_KIND_OPENROUTER_VIDEO = "openrouter_video"

VIDEO_API_BASE = "https://openrouter.ai/api/v1/videos"

# All OpenRouter video models and their C2PA status.
#
# Only C2PA-capable models are ENABLED below. Miners pay for each generation
# and validators reject unsigned content, so disabled models would waste money.
#
# Disabled (no C2PA manifest produced):
#   kwaivgi/kling-v3.0-pro, kwaivgi/kling-v3.0-std, kwaivgi/kling-video-o1
#   alibaba/wan-2.6, alibaba/wan-2.7
#   minimax/hailuo-2.3
#   bytedance/seedance-1-5-pro  — no JUMBF data in output, confirmed via live test
#
# Disabled (broken cert chain — can't be fixed with trust anchors):
#   openai/sora-2-pro  — self-signed end-entity cert (CA=false), c2pa-rs rejects it
#
# Enabled (produces verifiable C2PA content):
#   google/veo-3.1, google/veo-3.1-fast, google/veo-3.1-lite  — Google C2PA Root CA G3
#   bytedance/seedance-2.0, bytedance/seedance-2.0-fast        — GlobalSign R45 chain (Byteplus Pte. Ltd.)
VIDEO_MODELS: Dict[str, Dict[str, Any]] = {
    "google/veo-3.1": {
        "durations": [4, 6, 8],
        "ratios": ["16:9", "9:16"],
        "resolutions": ["720p", "1080p", "4K"],
        "generate_audio": True,
        "supports_seed": True,
        "frame_images": ["first_frame", "last_frame"],
    },
    "google/veo-3.1-fast": {
        "durations": [4, 6, 8],
        "ratios": ["16:9", "9:16"],
        "resolutions": ["720p", "1080p", "4K"],
        "generate_audio": True,
        "supports_seed": True,
        "frame_images": ["first_frame", "last_frame"],
    },
    "google/veo-3.1-lite": {
        "durations": [4, 6, 8],
        "ratios": ["16:9", "9:16"],
        "resolutions": ["720p", "1080p"],
        "generate_audio": True,
        "supports_seed": True,
        "frame_images": ["first_frame", "last_frame"],
    },
    "bytedance/seedance-2.0": {
        "durations": list(range(4, 16)),
        "ratios": ["1:1", "3:4", "9:16", "4:3", "16:9", "21:9", "9:21"],
        "resolutions": ["480p", "720p", "1080p"],
        "generate_audio": True,
        "supports_seed": True,
        "frame_images": ["first_frame", "last_frame"],
    },
    "bytedance/seedance-2.0-fast": {
        "durations": list(range(4, 16)),
        "ratios": ["1:1", "3:4", "9:16", "4:3", "16:9", "21:9", "9:21"],
        "resolutions": ["480p", "720p"],
        "generate_audio": True,
        "supports_seed": True,
        "frame_images": ["first_frame", "last_frame"],
    },
}

DEFAULT_VIDEO_MODEL = "google/veo-3.1-lite"

RESOLUTION_PRIORITY = ["720p", "480p", "1080p", "1K", "2K", "4K"]


def _nearest_duration(requested: int, allowed: List[int]) -> int:
    return min(allowed, key=lambda x: abs(x - requested))


def _nearest_resolution(requested: str, allowed: List[str]) -> str:
    try:
        idx = RESOLUTION_PRIORITY.index(requested)
    except ValueError:
        idx = 2  # default to 720p-ish
    for r in reversed(RESOLUTION_PRIORITY[:idx + 1]):
        if r in allowed:
            return r
    return allowed[0]


def canonical_video_model(model: Any) -> str:
    """Resolve parameters.model to a known OpenRouter video model ID (case-insensitive)."""
    if model is None or (isinstance(model, str) and not model.strip()):
        return DEFAULT_VIDEO_MODEL
    raw = str(model).strip()
    if raw in VIDEO_MODELS:
        return raw
    lower = raw.lower()
    for mid in VIDEO_MODELS:
        if mid.lower() == lower:
            return mid
    accepted = ", ".join(sorted(VIDEO_MODELS.keys()))
    raise ValueError(
        f"Unsupported OpenRouter video model {raw!r}. Accepted values: {accepted}"
    )


class OpenRouterService(BaseGenerationService):
    """
    OpenRouter API service for image and video generation.
    
    - Image: via /api/v1/chat/completions with Gemini Flash image models.
    - Video: via /api/v1/videos (separate endpoints).
      Only C2PA-capable models are enabled (Google Veo, ByteDance Seedance).
      Kling, Wan, Hailuo, and Sora are excluded — they either don't sign
      content or use broken cert chains that validators will reject.
    """
    
    def __init__(self, config: Any = None):
        super().__init__(config)
        self.api_key = os.getenv('OPEN_ROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.video_base_url = VIDEO_API_BASE
        
        self.default_model = "google/gemini-3-pro-image-preview"
        self.default_video_model = DEFAULT_VIDEO_MODEL
        
        self.timeout = 60.0
        self.max_retries = 3
        self.download_timeout = 600
        
        if self.api_key and self.api_key.strip():
            bt.logging.info("OpenRouter service initialized with API key")
        else:
            bt.logging.warning(f"OpenRouter API key not found or empty. Value: {repr(self.api_key)}. Set OPEN_ROUTER_API_KEY environment variable.")
    
    def is_available(self) -> bool:
        return self.api_key is not None and self.api_key.strip() != ""
    
    def supports_modality(self, modality: str) -> bool:
        return modality in {"image", "video"}
    
    def get_supported_tasks(self) -> Dict[str, list]:
        return {
            "image": ["image_generation"],
            "video": ["text_to_video"],
        }
    
    def get_api_key_requirements(self) -> Dict[str, str]:
        return {
            "OPEN_ROUTER_API_KEY": "OpenRouter API key for multi-model image & video generation"
        }
    
    async def process(self, task: GenerationTask) -> Optional[Dict[str, Any]]:
        """Async: handles image generation via chat completions API."""
        if not self.is_available():
            bt.logging.error("OpenRouter service not available")
            raise RuntimeError("OpenRouter service not available")
        
        try:
            prompt = task.prompt
            modality = task.modality
            parameters = task.parameters or {}
            
            bt.logging.info(f"OpenRouter processing task: modality={modality}")
            bt.logging.debug(f"Task prompt: {prompt[:100]}...")
            bt.logging.debug(f"Task parameters: {parameters}")
            
            if modality != "image":
                raise ValueError(f"OpenRouter async process() only supports images, got {modality}")
            
            model = parameters.get('model', self.default_model)
            bt.logging.info(f"Generating image with OpenRouter model: {model}")
            
            api_result = await self._generate_image(prompt, model)
            
            if api_result is None:
                bt.logging.error("OpenRouter API returned None")
                raise RuntimeError("OpenRouter API returned None")
            
            bt.logging.debug(f"OpenRouter API result keys: {list(api_result.keys())}")
            
            image_data = api_result.get("raw_binary")
            pil_image = api_result.get("image")
            
            if image_data is None:
                bt.logging.error("No raw binary data in OpenRouter API result")
                raise ValueError("No raw binary data in OpenRouter API result")
            
            bt.logging.info(
                f"OpenRouterService returning: {len(image_data)} bytes, "
                f"magic={image_data[:16].hex()}"
            )
            
            result = {
                "data": image_data,
                "metadata": {
                    "model": model,
                    "provider": "openrouter",
                    "format": api_result.get("format", "unknown"),
                    "generation_time": api_result.get('gen_duration', 0),
                    "width": pil_image.width if pil_image else 0,
                    "height": pil_image.height if pil_image else 0,
                }
            }
            
            bt.logging.success(f"OpenRouter generation completed in {api_result.get('gen_duration', 0):.2f}s")
            return result
            
        except Exception as e:
            bt.logging.error(f"Error in OpenRouter processing: {e}")
            bt.logging.error(f"OpenRouter processing failed for task {task.task_id}: {type(e).__name__}: {e}")
            raise
    
    def process_with_checkpoint(
        self, task: GenerationTask, on_checkpoint: CheckpointFn = None
    ) -> Dict[str, Any]:
        """Sync: handles video generation via /videos endpoints with checkpoint support."""
        if not self.is_available():
            bt.logging.error("OpenRouter service not available")
            raise RuntimeError("OpenRouter service not available")
        
        if task.modality != "video":
            raise ValueError(
                f"OpenRouter process_with_checkpoint only supports video, got {task.modality}"
            )
        
        return self._generate_video(task, on_checkpoint)
    
    # ────────────────── Image Generation ──────────────────
    
    async def _generate_image(self, prompt: str, model: str) -> Optional[Dict[str, Any]]:
        """
        Generate an image using OpenRouter API.
        
        Based on the nano_banana.py implementation but adapted for async operation.
        
        Args:
            prompt: The text prompt for image generation
            model: The model to use for generation
            
        Returns:
            Dict with generation results or None on failure
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "modalities": ["image", "text"]
        }
        
        start_time = time.time()
        
        try:
            # Make API request
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_text = response.text if response.text else "Unknown error"
                bt.logging.error(f"OpenRouter API error: {response.status_code} - {error_text}")
                raise Exception(f"OpenRouter API returned {response.status_code}: {error_text}")

            # Safely parse JSON response
            try:
                result = response.json()
                if result is None:
                    bt.logging.error("OpenRouter API returned null JSON")
                    raise ValueError("OpenRouter API returned null JSON")
                if not isinstance(result, dict):
                    bt.logging.error(f"OpenRouter API returned non-dict JSON: {type(result)}")
                    raise ValueError(f"OpenRouter API returned non-dict JSON: {type(result)}")
            except ValueError as e:
                bt.logging.error(f"OpenRouter API returned invalid JSON: {e}")
                raise ValueError(f"OpenRouter API returned invalid JSON: {e}")

            # Parse response (following nano_banana pattern)
            if 'choices' not in result or not result['choices']:
                bt.logging.error("OpenRouter response missing choices")
                bt.logging.debug(f"OpenRouter response: {result}")
                raise ValueError("OpenRouter response missing choices")

            choice = result['choices'][0]
            bt.logging.debug(f"OpenRouter choice keys: {list(choice.keys())}")

            if 'message' not in choice or 'images' not in choice['message']:
                bt.logging.error("OpenRouter response missing images in message")
                bt.logging.debug(f"OpenRouter message keys: {list(choice.get('message', {}).keys())}")
                raise ValueError("OpenRouter response missing images in message")

            message = choice['message']
            bt.logging.debug(f"OpenRouter message keys: {list(message.keys())}")

            images = message['images']
            if not images:
                bt.logging.error("OpenRouter response contains no images")
                raise ValueError("OpenRouter response contains no images")

            bt.logging.debug(f"OpenRouter image[0] keys: {list(images[0].keys())}")

            # Process the first image
            image_url = images[0]['image_url']['url']

            # Handle base64 data URL format: "data:image/<fmt>;base64,<data>"
            if ',' not in image_url:
                bt.logging.error("Invalid image URL format from OpenRouter")
                raise ValueError("Invalid image URL format from OpenRouter")

            # Extract format from data URL (e.g., "data:image/png;base64,...")
            header_part = image_url.split(',')[0]
            image_format = "PNG"
            if "image/" in header_part:
                fmt = header_part.split("image/")[1].split(";")[0].upper()
                if fmt in ["PNG", "JPEG", "JPG", "WEBP"]:
                    image_format = fmt

            bt.logging.debug(f"OpenRouter image data URL header: {header_part}")
            bt.logging.debug(f"OpenRouter detected format: {image_format}")
                
            base64_data = image_url.split(',')[1]
            image_binary = base64.b64decode(base64_data)
            
            bt.logging.info(f"OpenRouter raw binary size: {len(image_binary)} bytes")
            bt.logging.debug(f"OpenRouter binary magic bytes: {image_binary[:16].hex()}")
            
            # Open with PIL just to get dimensions, but preserve raw bytes for C2PA
            pil_image = Image.open(io.BytesIO(image_binary))

            gen_time = time.time() - start_time

            output = {
                "image": pil_image,
                "raw_binary": image_binary,  # Preserve original bytes with C2PA
                "format": image_format,
                "modality": "image",
                "media_type": "synthetic",
                "prompt": prompt,
                "model_name": model,
                "time": time.time(),
                "gen_duration": gen_time,
                "gen_args": {
                    "provider": "openrouter",
                    "model": model,
                    "modalities": ["image", "text"],
                    "api_url": self.base_url,
                },
            }

            return output
            
        except requests.exceptions.Timeout:
            bt.logging.error(f"OpenRouter API timeout after {self.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"OpenRouter API request failed: {e}")
            raise
        except Exception as e:
            bt.logging.error(f"OpenRouter image processing failed: {e}")
            raise
    
    # ────────────────── Video Generation ──────────────────
    
    def _generate_video(
        self, task: GenerationTask, on_checkpoint: CheckpointFn
    ) -> Dict[str, Any]:
        """Generate a video via OpenRouter /videos endpoints with polling and checkpointing."""
        oc = on_checkpoint or (lambda _: None)
        params = task.parameters or {}
        model = canonical_video_model(params.get("model", self.default_video_model))
        
        poll_interval = float(params.get("poll_interval", 10.0))
        poll_timeout = float(params.get("timeout", 2400.0))
        
        resume = (
            task.checkpoint is not None
            and task.checkpoint.get("kind") == CHECKPOINT_KIND_OPENROUTER_VIDEO
            and task.checkpoint.get("job_id")
        )
        
        if resume:
            bt.logging.info(
                f"Resuming OpenRouter video job_id={task.checkpoint.get('job_id')} "
                f"(miner restart recovery)"
            )
            return self._resume_video_checkpoint(task, oc, poll_interval, poll_timeout)
        
        create_kwargs = self._build_video_create_kwargs(task.prompt, params, model)
        bt.logging.info(f"OpenRouter text-to-video: model={model}, prompt={task.prompt[:60]!r}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        start_time = time.time()
        
        # Phase 1: Submit
        resp = requests.post(
            self.video_base_url,
            headers=headers,
            json=create_kwargs,
            timeout=60,
        )
        if resp.status_code != 202:
            detail = resp.text
            bt.logging.error(f"OpenRouter video create failed: {resp.status_code} - {detail}")
            raise RuntimeError(f"OpenRouter video create failed: {resp.status_code} - {detail}")
        
        job = resp.json()
        job_id = job["id"]
        status = job["status"]
        polling_url = job.get("polling_url", "")
        
        bt.logging.info(f"OpenRouter video job created: id={job_id}, status={status}")
        
        # Save checkpoint
        oc({
            "kind": CHECKPOINT_KIND_OPENROUTER_VIDEO,
            "job_id": job_id,
            "model": model,
            "create_kwargs": create_kwargs,
        })
        
        # Phase 2: Poll
        try:
            result = self._poll_video_job(job_id, headers, poll_interval, poll_timeout, start_time)
        finally:
            oc(None)
        
        video_bytes = self._download_video(job_id, headers)
        
        elapsed = time.time() - start_time
        bt.logging.success(f"OpenRouter video downloaded: {len(video_bytes)} bytes in {elapsed:.1f}s")
        
        return {
            "data": video_bytes,
            "metadata": {
                "model": model,
                "provider": "openrouter",
                "mime_type": "video/mp4",
                "job_id": job_id,
                "generation_time": elapsed,
            },
        }
    
    def _resume_video_checkpoint(
        self,
        task: GenerationTask,
        oc,
        poll_interval: float,
        poll_timeout: float,
    ) -> Dict[str, Any]:
        """Resume polling an existing OpenRouter video job after miner restart."""
        cp = task.checkpoint
        job_id = cp["job_id"]
        model = cp.get("model", task.parameters.get("model", self.default_video_model))
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        bt.logging.info(f"Resuming OpenRouter video job_id={job_id}")
        start_time = time.time()
        
        try:
            result = self._poll_video_job(job_id, headers, poll_interval, poll_timeout, start_time)
        finally:
            oc(None)
        
        video_bytes = self._download_video(job_id, headers)
        
        elapsed = time.time() - start_time
        bt.logging.success(f"OpenRouter video (resumed) downloaded: {len(video_bytes)} bytes in {elapsed:.1f}s")
        
        return {
            "data": video_bytes,
            "metadata": {
                "model": model,
                "provider": "openrouter",
                "mime_type": "video/mp4",
                "job_id": job_id,
                "generation_time": elapsed,
                "resumed": True,
            },
        }
    
    def _poll_video_job(
        self,
        job_id: str,
        headers: Dict[str, str],
        poll_interval: float,
        poll_timeout: float,
        start_time: float,
    ) -> Dict[str, Any]:
        """Poll /videos/{job_id} until completed, failed, or timeout."""
        while True:
            elapsed = time.time() - start_time
            if elapsed > poll_timeout:
                raise TimeoutError(
                    f"OpenRouter video generation timed out after {poll_timeout}s (job_id={job_id})"
                )
            
            resp = requests.get(f"{self.video_base_url}/{job_id}", headers=headers, timeout=15)
            if resp.status_code != 200:
                bt.logging.error(f"OpenRouter poll error: {resp.status_code} - {resp.text}")
                time.sleep(poll_interval)
                continue
            
            job = resp.json()
            status = job.get("status", "unknown")
            error = job.get("error", "")
            
            bt.logging.info(f"OpenRouter video job {job_id}: status={status}, elapsed={elapsed:.1f}s")
            
            if status == "completed":
                return job
            elif status in ("failed", "cancelled", "expired"):
                detail = error or status
                raise RuntimeError(f"OpenRouter video generation {status}: {detail}")
            elif status in ("pending", "in_progress"):
                time.sleep(poll_interval)
                continue
            else:
                bt.logging.warning(f"Unknown OpenRouter video status: {status}")
                time.sleep(poll_interval)
    
    def _download_video(self, job_id: str, headers: Dict[str, str]) -> bytes:
        """Download generated video content from /videos/{job_id}/content."""
        bt.logging.info(f"Downloading OpenRouter video content for job_id={job_id}")
        resp = requests.get(
            f"{self.video_base_url}/{job_id}/content",
            headers=headers,
            timeout=self.download_timeout,
        )
        resp.raise_for_status()
        return resp.content
    
    def _build_video_create_kwargs(
        self, prompt: str, params: Dict[str, Any], model: str
    ) -> Dict[str, Any]:
        """Map task parameters to OpenRouter /videos request body."""
        model_info = VIDEO_MODELS.get(model, {})
        allowed_durations: List[int] = model_info.get("durations", [4, 6, 8])
        allowed_ratios: List[str] = model_info.get("ratios", ["16:9"])
        allowed_resolutions: List[str] = model_info.get("resolutions", ["720p"])
        supports_seed: bool = model_info.get("supports_seed", False)
        generate_audio_default: bool = model_info.get("generate_audio", False)
        
        # Duration
        dur_raw = int(params.get("duration", allowed_durations[0]))
        duration = _nearest_duration(dur_raw, allowed_durations)
        
        # Aspect ratio
        ratio = params.get("aspect_ratio", params.get("ratio", allowed_ratios[0]))
        if isinstance(ratio, str):
            ratio = ratio.strip()
        if ratio not in allowed_ratios:
            bt.logging.warning(f"Aspect ratio {ratio!r} not supported by {model}; using {allowed_ratios[0]}")
            ratio = allowed_ratios[0]
        
        # Resolution
        resolution = params.get("resolution", allowed_resolutions[0])
        if resolution not in allowed_resolutions:
            resolution = _nearest_resolution(resolution, allowed_resolutions)
            bt.logging.warning(f"Resolution adjusted to {resolution} for {model}")
        
        kwargs: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": ratio,
            "resolution": resolution,
        }
        
        # Audio — only include if the model supports toggling it
        # Some models (e.g. Sora) reject generate_audio altogether
        audio_settable = model_info.get("audio_settable", generate_audio_default)
        if "generate_audio" in params and audio_settable:
            kwargs["generate_audio"] = bool(params["generate_audio"])
        
        # Seed
        if supports_seed and "seed" in params:
            kwargs["seed"] = int(params["seed"])
        
        # Size override
        if "size" in params:
            kwargs["size"] = params["size"]
        
        return kwargs
    
    # ────────────────── Service Info ──────────────────
    
    def get_service_info(self) -> Dict[str, Any]:
        """Return information about this service."""
        return {
            "name": "OpenRouter",
            "type": "api",
            "provider": "openrouter.ai",
            "available": self.is_available(),
            "supported_tasks": self.get_supported_tasks(),
            "default_model": self.default_model,
            "default_video_model": self.default_video_model,
            "video_models": sorted(VIDEO_MODELS.keys()),
            "base_url": self.base_url,
            "video_base_url": self.video_base_url,
            "config": {
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
        }
