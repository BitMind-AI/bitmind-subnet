import os
import io
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

import bittensor as bt
import numpy as np
from PIL import Image

from .base_service import BaseGenerationService
from ..task_manager import GenerationTask


class LocalService(BaseGenerationService):
    """
    Local model service for running open source models.
    
    This demonstrates how to implement a local service that:
    1. Loads and runs models locally
    2. Returns binary data instead of URLs
    3. Handles model loading and GPU management
    4. Can be extended with different model types
    """
    
    def __init__(self, config: Any = None):
        super().__init__(config)
        self.device = self._get_device()
        self.image_model = None
        self.video_model = None
        self.models_loaded = False
        
        # Try to load models
        try:
            self._load_models()
        except Exception as e:
            bt.logging.warning(f"Failed to load local models: {e}")
    
    def _get_device(self) -> str:
        """Determine the best device to use."""
        if hasattr(self.config, 'device') and self.config.device:
            return self.config.device
        
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    
    def _load_models(self):
        """Load local models."""
        bt.logging.info("Loading local generation models...")
        
        # Try to load Stable Diffusion for image generation
        try:
            self._load_image_model()
        except Exception as e:
            bt.logging.warning(f"Failed to load image model: {e}")
        
        # Try to load video model (placeholder)
        try:
            self._load_video_model()
        except Exception as e:
            bt.logging.warning(f"Failed to load video model: {e}")
        
        self.models_loaded = True
    
    def _load_image_model(self):
        """Load Stable Diffusion model for image generation."""
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            model_id = "runwayml/stable-diffusion-v1-5"
            bt.logging.info(f"Loading Stable Diffusion model: {model_id}")
            
            self.image_model = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # Disable safety checker for faster inference
                requires_safety_checker=False,
            )
            
            if self.device == "cuda":
                self.image_model = self.image_model.to("cuda")
            
            bt.logging.success("✅ Stable Diffusion model loaded")
            
        except ImportError:
            bt.logging.warning("Diffusers package not installed. Run: pip install diffusers torch")
            raise
        except Exception as e:
            bt.logging.error(f"Failed to load Stable Diffusion: {e}")
            raise
    
    def _load_video_model(self):
        """Load Wan2.2 video generation model."""
        try:
            from diffusers import WanPipeline, AutoencoderKLWan
            import torch
            
            model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
            bt.logging.info(f"Loading Wan2.2 video model: {model_id}")
            
            # Load VAE with float32 for better stability
            vae = AutoencoderKLWan.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=torch.float32
            )
            
            # Load main pipeline
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.video_model = WanPipeline.from_pretrained(
                model_id,
                vae=vae,
                torch_dtype=dtype
            )
            
            if self.device == "cuda":
                self.video_model = self.video_model.to(self.device)
            
            bt.logging.success("✅ Wan2.2 video model loaded")
            
        except ImportError:
            bt.logging.warning("Required packages not installed. Run: pip install diffusers torch")
            self.video_model = None
        except Exception as e:
            bt.logging.error(f"Failed to load Wan2.2 video model: {e}")
            self.video_model = None
    
    def is_available(self) -> bool:
        """Check if local service is available."""
        return self.models_loaded and (self.image_model is not None or self.video_model is not None)
    
    def supports_task(self, task_type: str, modality: str) -> bool:
        """Check if this service supports the task type and modality."""
        if modality == "image" and self.image_model is not None:
            return task_type in ["image_generation", "image_modification"]
        elif modality == "video" and self.video_model is not None:
            return task_type in ["video_generation", "video_modification"]
        return False
    
    def get_supported_tasks(self) -> Dict[str, list]:
        """Return supported tasks by modality."""
        tasks = {}
        
        if self.image_model is not None:
            tasks["image"] = ["image_generation", "image_modification"]
        else:
            tasks["image"] = []
            
        if self.video_model is not None:
            tasks["video"] = ["video_generation", "video_modification"]
        else:
            tasks["video"] = []
            
        return tasks
    
    def get_api_key_requirements(self) -> Dict[str, str]:
        """Return API key requirements for local service."""
        return {
            "HUGGINGFACE_HUB_TOKEN": "Hugging Face token for model downloads (optional but recommended)"
        }
    
    def process(self, task: GenerationTask) -> Dict[str, Any]:
        """Process a task using local models."""
        if task.modality == "image":
            return self._process_image_task(task)
        elif task.modality == "video":
            return self._process_video_task(task)
        else:
            raise ValueError(f"Unsupported modality: {task.modality}")
    
    def _process_image_task(self, task: GenerationTask) -> Dict[str, Any]:
        """Process an image task."""
        if task.task_type == "image_generation":
            return self._generate_image_local(task)
        elif task.task_type == "image_modification":
            return self._modify_image_local(task)
        else:
            raise ValueError(f"Unsupported image task: {task.task_type}")
    
    def _generate_image_local(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate an image using local Stable Diffusion."""
        try:
            bt.logging.info(f"Generating image locally: {task.prompt[:50]}...")
            
            # Extract parameters
            width = task.parameters.get("width", 512)
            height = task.parameters.get("height", 512)
            num_inference_steps = task.parameters.get("steps", 20)
            guidance_scale = task.parameters.get("guidance_scale", 7.5)
            
            # Generate image
            with bt.logging.debug("Running Stable Diffusion inference..."):
                image = self.image_model(
                    prompt=task.prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]
            
            # Convert to bytes
            img_bytes = self._pil_to_bytes(image)
            
            bt.logging.success("Image generated successfully with local model")
            
            return {
                "data": img_bytes,
                "metadata": {
                    "model": "stable-diffusion-v1-5",
                    "width": width,
                    "height": height,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "provider": "local"
                }
            }
            
        except Exception as e:
            bt.logging.error(f"Local image generation failed: {e}")
            raise
    
    def _modify_image_local(self, task: GenerationTask) -> Dict[str, Any]:
        """Modify an image using local models."""
        try:
            bt.logging.info(f"Modifying image locally: {task.prompt[:50]}...")
            
            # For this example, we'll use img2img if available
            # In a real implementation, you might use:
            # - StableDiffusionImg2ImgPipeline
            # - StableDiffusionInpaintPipeline
            # - ControlNet
            
            # Convert input data to PIL Image
            if task.input_data:
                input_image = Image.open(io.BytesIO(task.input_data))
            else:
                raise ValueError("No input image provided for modification")
            
            # For this example, we'll just generate a new image
            # In practice, you'd use img2img with the input image
            return self._generate_image_local(task)
            
        except Exception as e:
            bt.logging.error(f"Local image modification failed: {e}")
            raise
    
    def _process_video_task(self, task: GenerationTask) -> Dict[str, Any]:
        """Process a video task using Wan2.2."""
        if task.task_type == "video_generation":
            return self._generate_video_local(task)
        elif task.task_type == "video_modification":
            return self._modify_video_local(task)
        else:
            raise ValueError(f"Unsupported video task: {task.task_type}")
    
    def _generate_video_local(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate a video using local Wan2.2 model."""
        try:
            bt.logging.info(f"Generating video locally: {task.prompt[:50]}...")
            
            if self.video_model is None:
                raise ValueError("Video model not loaded")
            
            height = task.parameters.get("height", 720)
            width = task.parameters.get("width", 1280)
            num_frames = task.parameters.get("num_frames", 81)
            guidance_scale = task.parameters.get("guidance_scale", 4.0)
            guidance_scale_2 = task.parameters.get("guidance_scale_2", 3.0)
            num_inference_steps = task.parameters.get("num_inference_steps", 40)
            fps = task.parameters.get("fps", 16)
            
            negative_prompt = task.parameters.get(
                "negative_prompt",
            )
            
            # Generate video
            with bt.logging.debug("Running Wan2.2 video generation..."):
                output = self.video_model(
                    prompt=task.prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    guidance_scale_2=guidance_scale_2,
                    num_inference_steps=num_inference_steps,
                ).frames[0]
            
            # Convert video frames to bytes
            video_bytes = self._frames_to_video_bytes(output, fps=fps)
            
            bt.logging.success("Video generated successfully with Wan2.2 model")
            
            return {
                "data": video_bytes,
                "metadata": {
                    "model": "Wan2.2-T2V-A14B",
                    "width": width,
                    "height": height,
                    "num_frames": num_frames,
                    "fps": fps,
                    "guidance_scale": guidance_scale,
                    "guidance_scale_2": guidance_scale_2,
                    "num_inference_steps": num_inference_steps,
                    "provider": "local"
                }
            }
            
        except Exception as e:
            bt.logging.error(f"Local video generation failed: {e}")
            raise
    
    def _modify_video_local(self, task: GenerationTask) -> Dict[str, Any]:
        """Modify a video using local models."""
        try:
            bt.logging.info(f"Modifying video locally: {task.prompt[:50]}...")
            
            # For video modification, we could implement:
            # - Video-to-video generation with input video as conditioning
            # - Frame-by-frame modification
            # - Video inpainting
            
            # For now, we'll use the generation pipeline
            # In practice, you'd implement proper video modification
            if task.input_data:
                bt.logging.info("Input video provided - using generation pipeline for now")
            
            # Use generation pipeline with modified prompt
            modification_prompt = f"Modified version: {task.prompt}"
            modified_task = GenerationTask(
                task_id=task.task_id,
                task_type="video_generation",
                modality=task.modality,
                prompt=modification_prompt,
                parameters=task.parameters,
                webhook_url=task.webhook_url,
                signed_by=task.signed_by,
                input_data=task.input_data
            )
            
            return self._generate_video_local(modified_task)
            
        except Exception as e:
            bt.logging.error(f"Local video modification failed: {e}")
            raise
    
    def _frames_to_video_bytes(self, frames, fps: int = 16) -> bytes:
        """Convert video frames to MP4 bytes."""
        try:
            from diffusers.utils import export_to_video
            
            # Create temporary file for video export
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            try:
                # Export frames to temporary video file
                export_to_video(frames, temp_path, fps=fps)
                
                # Read video file as bytes
                with open(temp_path, 'rb') as f:
                    video_bytes = f.read()
                
                return video_bytes
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            bt.logging.error(f"Failed to convert frames to video bytes: {e}")
            raise
    
    def _pil_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Convert PIL Image to bytes."""
        img_buffer = io.BytesIO()
        image.save(img_buffer, format=format)
        return img_buffer.getvalue()
