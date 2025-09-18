import os
import time
import base64
import io
from typing import Dict, Any, Optional

import requests
import bittensor as bt
from PIL import Image

from .base_service import BaseGenerationService
from ..task_manager import GenerationTask


class OpenRouterService(BaseGenerationService):
    """
    OpenRouter API service for image generation using various models.
    
    This service integrates with OpenRouter.ai to provide access to multiple
    image generation models including Google Gemini Flash Image Preview.
    
    Based on the nano_banana implementation pattern but adapted for the
    new modular service architecture.
    """
    
    def __init__(self, config: Any = None):
        super().__init__(config)
        self.api_key = os.getenv('OPEN_ROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Default model from nano_banana
        self.default_model = "google/gemini-2.5-flash-image-preview"
        
        # Timeout and retry settings
        self.timeout = 60.0
        self.max_retries = 3
        
        if self.api_key:
            bt.logging.info("OpenRouter service initialized with API key")
        else:
            bt.logging.warning("OpenRouter API key not found. Set OPEN_ROUTER_API_KEY environment variable.")
    
    def is_available(self) -> bool:
        """Check if OpenRouter service is available."""
        return self.api_key is not None
    
    def supports_modality(self, modality: str) -> bool:
        """Check if this service supports the given modality."""
        # OpenRouter supports image generation via various models
        return modality == "image"
    
    def get_supported_tasks(self) -> Dict[str, list]:
        """Return supported tasks by modality."""
        return {
            "image": ["image_generation"]
        }
    
    def get_api_key_requirements(self) -> Dict[str, str]:
        """Return OpenRouter API key requirements."""
        return {
            "OPEN_ROUTER_API_KEY": "OpenRouter API key for multi-model image generation"
        }
    
    async def process(self, task: GenerationTask) -> Optional[Dict[str, Any]]:
        """
        Process a generation task using OpenRouter API.
        
        Args:
            task: The generation task to process
            
        Returns:
            Dict with generation results or None on failure
        """
        if not self.is_available():
            bt.logging.error("OpenRouter service not available")
            raise RuntimeError("OpenRouter service not available")
        
        try:
            # Extract task parameters with enhanced logging
            prompt = task.prompt
            modality = task.modality
            parameters = task.parameters or {}
            
            bt.logging.info(f"OpenRouter processing task: modality={modality}")
            bt.logging.debug(f"Task prompt: {prompt[:100]}...")
            bt.logging.debug(f"Task parameters: {parameters}")
            
            # Validate task
            if not self.supports_modality(modality):
                bt.logging.error(f"OpenRouter service doesn't support modality {modality}")
                raise ValueError(f"OpenRouter service doesn't support modality {modality}")
            
            # Get model from parameters or use default
            model = parameters.get('model', self.default_model)
            
            bt.logging.info(f"Generating image with OpenRouter model: {model}")
            
            # Call OpenRouter API
            api_result = await self._generate_image(prompt, model)
            
            if api_result is None:
                bt.logging.error("OpenRouter API returned None")
                raise RuntimeError("OpenRouter API returned None")
            
            bt.logging.debug(f"OpenRouter API result keys: {list(api_result.keys())}")
            
            # Convert PIL image to bytes for miner compatibility
            pil_image = api_result.get("image")
            if pil_image is None:
                bt.logging.error("No image in OpenRouter API result")
                raise ValueError("No image in OpenRouter API result")
            
            # Convert PIL image to PNG bytes
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            image_data = img_buffer.getvalue()
            
            bt.logging.info(f"Converted image to {len(image_data)} bytes")
            
            # Return in the format expected by miner
            result = {
                "data": image_data,  # Binary data for miner compatibility
                "metadata": {
                    "model": model,
                    "provider": "openrouter",
                    "format": "PNG",
                    "generation_time": api_result.get('gen_duration', 0),
                    "width": pil_image.width,
                    "height": pil_image.height,
                }
            }
            
            bt.logging.success(f"OpenRouter generation completed in {api_result.get('gen_duration', 0):.2f}s")
            return result
            
        except Exception as e:
            bt.logging.error(f"Error in OpenRouter processing: {e}")
            bt.logging.error(f"OpenRouter processing failed for task {task.task_id}: {type(e).__name__}: {e}")
            raise  # Re-raise instead of returning None
    
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
            if 'message' not in choice or 'images' not in choice['message']:
                bt.logging.error("OpenRouter response missing images in message")
                raise ValueError("OpenRouter response missing images in message")

            images = choice['message']['images']
            if not images:
                bt.logging.error("OpenRouter response contains no images")
                raise ValueError("OpenRouter response contains no images")

            # Process the first image
            image_url = images[0]['image_url']['url']
            
            # Handle base64 data URL format: "data:image/<fmt>;base64,<data>"
            if ',' not in image_url:
                bt.logging.error("Invalid image URL format from OpenRouter")
                raise ValueError("Invalid image URL format from OpenRouter")
                
            base64_data = image_url.split(',')[1]
            image_binary = base64.b64decode(base64_data)
            pil_image = Image.open(io.BytesIO(image_binary))

            gen_time = time.time() - start_time

            # Return in the expected format
            output = {
                "image": pil_image,
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
    
    def get_service_info(self) -> Dict[str, Any]:
        """Return information about this service."""
        return {
            "name": "OpenRouter",
            "type": "api",
            "provider": "openrouter.ai",
            "available": self.is_available(),
            "supported_tasks": self.get_supported_tasks(),
            "default_model": self.default_model,
            "base_url": self.base_url,
            "config": {
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
        }
