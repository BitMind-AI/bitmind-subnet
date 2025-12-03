import os
import time
from typing import Dict, Any, Optional

import bittensor as bt
import requests

from .base_service import BaseGenerationService
from ..task_manager import GenerationTask


class OpenAIService(BaseGenerationService):
    """
    OpenAI API service for image generation using DALL-E.
    
    DALL-E 3 embeds C2PA content credentials in generated images,
    so we download the actual image bytes to preserve this metadata.
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
        return (self.api_key is not None and 
                self.api_key.strip() != "" and 
                self.client is not None)
    
    def supports_modality(self, modality: str) -> bool:
        """Check if this service supports the given modality."""
        # OpenAI DALL-E only supports image generation
        return modality == "image"
    
    def get_supported_tasks(self) -> Dict[str, list]:
        """Return supported tasks by modality."""
        return {
            "image": ["image_generation"],
            "video": []  # OpenAI doesn't support video generation yet
        }
    
    def get_api_key_requirements(self) -> Dict[str, str]:
        """Return OpenAI API key requirements."""
        return {
            "OPENAI_API_KEY": "OpenAI API key for DALL-E image generation"
        }
    
    def process(self, task: GenerationTask) -> Dict[str, Any]:
        """Process a task using OpenAI API."""
        if task.modality == "image":
            return self._generate_image(task)
        else:
            raise ValueError(f"Unsupported modality: {task.modality}")
    
    def _generate_image(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate an image using DALL-E and download the bytes with C2PA metadata."""
        try:
            bt.logging.info(f"Generating image with DALL-E: {task.prompt[:50]}...")
            
            params = task.parameters or {}
            width = params.get("width", 1024)
            height = params.get("height", 1024)
            quality = params.get("quality", "standard")
            
            size = self._get_valid_size(width, height)
            
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=task.prompt,
                size=size,
                quality=quality,
                n=1,
            )
            
            image_url = response.data[0].url
            bt.logging.info(f"DALL-E generated image, downloading from URL...")
            
            img_response = requests.get(image_url, timeout=60)
            img_response.raise_for_status()
            image_data = img_response.content
            
            bt.logging.success(f"Downloaded {len(image_data)} bytes with C2PA metadata")
            
            return {
                "data": image_data,
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
    
    
    def _get_valid_size(self, width: int, height: int) -> str:
        """Convert width/height to valid DALL-E size."""
        # DALL-E 3 supports: 1024x1024, 1024x1792, 1792x1024
        if width == height:
            return "1024x1024"
        elif width > height:
            return "1792x1024"
        else:
            return "1024x1792"
