import os
import time
import requests
import bittensor as bt
from typing import Dict, Any, Optional

from .base_service import BaseGenerationService
from ..task_manager import GenerationTask


class FalAIService(BaseGenerationService):
    """
    Fal.ai API service for media generation.
    
    Supports:
    - Images via FLUX.1 [dev] (fal-ai/flux/dev)
    - Video via Kling (fal-ai/kling-video/v1/standard/text-to-video)
    """
    
    def __init__(self, config: Any = None):
        super().__init__(config)
        self.api_key = os.getenv("FAL_KEY")
        self.base_url = "https://queue.fal.run"
        
        # Default models
        self.image_model = "fal-ai/flux/dev"
        self.video_model = "fal-ai/kling-video/v1/standard/text-to-video"
        
        if self.api_key:
            bt.logging.info("FalAIService initialized with API key")
        else:
            bt.logging.warning("FAL_KEY not found. Fal.ai service will not be available.")

    def is_available(self) -> bool:
        return self.api_key is not None and self.api_key.strip() != ""

    def supports_modality(self, modality: str) -> bool:
        return modality in {"image", "video"}

    def get_supported_tasks(self) -> Dict[str, list]:
        return {
            "image": ["image_generation"],
            "video": ["video_generation"],
        }

    def get_api_key_requirements(self) -> Dict[str, str]:
        return {"FAL_KEY": "Fal.ai API key for image and video generation"}

    def process(self, task: GenerationTask) -> Dict[str, Any]:
        if task.modality == "image":
            return self._generate_image(task)
        elif task.modality == "video":
            return self._generate_video(task)
        else:
            raise ValueError(f"Unsupported modality: {task.modality}")

    def _generate_image(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate an image using Fal.ai."""
        params = task.parameters or {}
        model = params.get("model", self.image_model)
        
        # Map common parameters to Fal.ai specific ones if needed
        # FLUX.1 [dev] supports: prompt, image_size, num_inference_steps, seed, guidance_scale, etc.
        payload = {
            "prompt": task.prompt,
            "image_size": params.get("size", "landscape_4_3"), # default to landscape
            "num_inference_steps": params.get("steps", 28),
            "seed": params.get("seed"),
            "guidance_scale": params.get("guidance_scale", 3.5),
            "num_images": 1,
            "enable_safety_checker": params.get("safety_checker", True),
            "sync_mode": False # Use queue for reliability
        }
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        return self._run_fal_request(model, payload, "image")

    def _generate_video(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate a video using Fal.ai."""
        params = task.parameters or {}
        model = params.get("model", self.video_model)
        
        # Kling supports: prompt, duration, aspect_ratio
        payload = {
            "prompt": task.prompt,
            "duration": str(params.get("duration", "5")), # "5" or "10"
            "aspect_ratio": params.get("aspect_ratio", "16:9"),
        }
        
        return self._run_fal_request(model, payload, "video")

    def _run_fal_request(self, model: str, payload: Dict[str, Any], modality: str) -> Dict[str, Any]:
        """Execute the request against Fal.ai queue and poll for result."""
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        
        url = f"{self.base_url}/{model}"
        
        bt.logging.info(f"Fal.ai submitting job to {model}...")
        
        try:
            # Submit job
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            request_id = data.get("request_id")
            if not request_id:
                # Some endpoints might return result immediately if sync_mode=True, 
                # but we forced sync_mode=False or are using queue.
                # If we get immediate result (unlikely with queue URL), handle it.
                if "images" in data or "video" in data:
                    return self._process_completed_response(data, model, modality)
                raise RuntimeError(f"No request_id returned from Fal.ai: {data}")
                
            bt.logging.info(f"Fal.ai job submitted. Request ID: {request_id}")
            
            # Poll for status
            start_time = time.time()
            timeout = 600 # 10 minutes max
            poll_interval = 2.0
            
            while time.time() - start_time < timeout:
                status_url = f"{self.base_url}/requests/{request_id}/status"
                status_res = requests.get(status_url, headers=headers)
                status_res.raise_for_status()
                status_data = status_res.json()
                
                status = status_data.get("status")
                
                if status == "COMPLETED":
                    # Fetch final result
                    result_url = f"{self.base_url}/requests/{request_id}"
                    result_res = requests.get(result_url, headers=headers)
                    result_res.raise_for_status()
                    result_data = result_res.json()
                    
                    return self._process_completed_response(result_data, model, modality)
                
                elif status == "FAILED":
                    error = status_data.get("error", "Unknown error")
                    raise RuntimeError(f"Fal.ai job failed: {error}")
                
                elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                    time.sleep(poll_interval)
                    poll_interval = min(poll_interval * 1.5, 10) # Exponential backoff cap at 10s
                    
                else:
                    bt.logging.warning(f"Unknown Fal.ai status: {status}")
                    time.sleep(poll_interval)
            
            raise TimeoutError(f"Fal.ai job timed out after {timeout}s")

        except Exception as e:
            bt.logging.error(f"Fal.ai request failed: {e}")
            raise

    def _process_completed_response(self, data: Dict[str, Any], model: str, modality: str) -> Dict[str, Any]:
        """Download and format the result."""
        
        media_url = None
        if modality == "image":
            # FLUX returns 'images': [{'url': ..., ...}]
            if "images" in data and len(data["images"]) > 0:
                media_url = data["images"][0]["url"]
        elif modality == "video":
            # Kling returns 'video': {'url': ...}
            if "video" in data and "url" in data["video"]:
                media_url = data["video"]["url"]
        
        if not media_url:
            raise RuntimeError(f"No media URL found in Fal.ai response: {data}")
            
        bt.logging.info(f"Downloading media from {media_url}...")
        media_res = requests.get(media_url)
        media_res.raise_for_status()
        media_bytes = media_res.content
        
        return {
            "data": media_bytes,
            "metadata": {
                "model": model,
                "provider": "fal.ai",
                "source_url": media_url,
                "size": len(media_bytes)
            }
        }
