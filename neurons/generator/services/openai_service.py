import os
import time
from typing import Dict, Any, Optional

import bittensor as bt

from .base_service import BaseGenerationService
from ..task_manager import GenerationTask


class OpenAIService(BaseGenerationService):
    """
    OpenAI API service for image generation using DALL-E.
    
    This demonstrates how to implement a 3rd party API service that:
    1. Forwards requests to external APIs
    2. Returns direct URLs instead of binary data
    3. Handles API-specific parameters and errors
    """
    
    def __init__(self, config: Any = None):
        super().__init__(config)
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = None
        
        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                bt.logging.info("OpenAI client initialized")
            except ImportError:
                bt.logging.warning("OpenAI package not installed. Run: pip install openai")
            except Exception as e:
                bt.logging.error(f"Failed to initialize OpenAI client: {e}")
    
    def is_available(self) -> bool:
        """Check if OpenAI service is available."""
        return self.api_key is not None and self.client is not None
    
    def supports_task(self, task_type: str, modality: str) -> bool:
        """Check if this service supports the task type and modality."""
        # OpenAI DALL-E only supports image generation and modification
        supported_tasks = {
            "image": ["image_generation", "image_modification"]
        }
        return modality in supported_tasks and task_type in supported_tasks[modality]
    
    def get_supported_tasks(self) -> Dict[str, list]:
        """Return supported tasks by modality."""
        return {
            "image": ["image_generation", "image_modification"],
            "video": []  # OpenAI doesn't support video generation yet
        }
    
    def get_api_key_requirements(self) -> Dict[str, str]:
        """Return OpenAI API key requirements."""
        return {
            "OPENAI_API_KEY": "OpenAI API key for DALL-E image generation"
        }
    
    def process(self, task: GenerationTask) -> Dict[str, Any]:
        """Process a task using OpenAI API."""
        if task.task_type == "image_generation":
            return self._generate_image(task)
        elif task.task_type == "image_modification":
            return self._modify_image(task)
        else:
            raise ValueError(f"Unsupported task type: {task.task_type}")
    
    def _generate_image(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate an image using DALL-E."""
        try:
            bt.logging.info(f"Generating image with DALL-E: {task.prompt[:50]}...")
            
            # Extract parameters
            width = task.parameters.get("width", 1024)
            height = task.parameters.get("height", 1024)
            quality = task.parameters.get("quality", "standard")
            
            # Ensure size is supported by DALL-E
            size = self._get_valid_size(width, height)
            
            # Generate image
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=task.prompt,
                size=size,
                quality=quality,
                n=1,
            )
            
            image_url = response.data[0].url
            bt.logging.success(f"Image generated successfully: {image_url}")
            
            return {
                "url": image_url,
                "metadata": {
                    "model": "dall-e-3",
                    "size": size,
                    "quality": quality,
                    "provider": "openai"
                }
            }
            
        except Exception as e:
            bt.logging.error(f"OpenAI image generation failed: {e}")
            raise
    
    def _modify_image(self, task: GenerationTask) -> Dict[str, Any]:
        """Modify an image using DALL-E (placeholder implementation)."""
        try:
            # Note: This is a simplified implementation
            # Real image modification would use OpenAI's image editing API
            # which requires uploading the original image and a mask
            
            bt.logging.info(f"Modifying image with DALL-E: {task.prompt[:50]}...")
            
            # For now, we'll use generation with a modified prompt
            # In a real implementation, you'd use the image editing endpoint
            modification_prompt = f"Modify this image: {task.prompt}"
            
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=modification_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            bt.logging.success(f"Image modification completed: {image_url}")
            
            return {
                "url": image_url,
                "metadata": {
                    "model": "dall-e-3",
                    "operation": "modification",
                    "provider": "openai"
                }
            }
            
        except Exception as e:
            bt.logging.error(f"OpenAI image modification failed: {e}")
            raise
    
    def _get_valid_size(self, width: int, height: int) -> str:
        """Convert width/height to valid DALL-E size."""
        # DALL-E 3 supports: 1024x1024, 1024x1792, 1792x1024
        if width == height:
            return "1024x1024"
        elif width > height:
            return "1792x1024"
        else:
            return "1024x1792"
