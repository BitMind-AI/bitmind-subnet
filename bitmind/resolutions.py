import numpy as np
from typing import Tuple

from bitmind.types import MediaType


"""Canonical real image resolutions derived from dataset analysis.

Images were grouped by aspect ratio, then dimensions were binned to the nearest
128 pixels to reduce noise from minor variations. For each aspect ratio, we kept
1 bin per 0.5% of dataset frequency (no minimum), resulting in a concise and
representative set of (width, height) tuples for use in the ResolutionSampler.

Synthetic image resolutions are based on common default sizes used by popular
generative models (e.g., Stable Diffusion, Midjourney, GPT-4V, etc.).
"""


class ResolutionSampler:
    """Samples image resolutions from realistic distributions based on media type."""

    def __init__(self) -> None:
        """Initialize the ResolutionSampler with real and synthetic resolution data."""
        self.real_resolutions = [
            (256, 256),
            (384, 256),
            (384, 384),
            (384, 640),
            (512, 256),
            (512, 384),
            (512, 512),
            (512, 640),
            (512, 768),
            (512, 896),
            (640, 384),
            (640, 512),
            (640, 640),
            (640, 768),
            (640, 896),
            (640, 1024),
            (768, 512),
            (768, 640),
            (768, 768),
            (896, 512),
            (896, 640),
            (896, 768),
            (896, 896),
            (1024, 512),
            (1024, 640),
            (1024, 768),
            (1536, 1152),
            (2048, 2048),
            (2432, 2432),
            (2560, 1920),
        ]
        # Assign equal weights to all real resolutions
        self.real_weights = np.ones(len(self.real_resolutions)) / len(self.real_resolutions)
        
        # Synthetic (generated) resolutions and weights based on popular model defaults
        self.synthetic_resolutions = [
            (1024, 1024),   # GPT-4V, SDXL, IF, etc.
            (2048, 2048),   # Midjourney default
            (512, 512),
            (768, 768),
            (640, 640),
            (896, 896),
            (800, 800),
            (960, 960),
            (512, 768),
            (768, 512),
            (512, 1024),
            (1024, 512),
            (512, 896),
            (896, 512),
            (720, 1280),
            (1280, 720),
            (832, 1104),
            (1104, 832),
            (480, 832),
            (480, 848),
            (720, 720),
            (854, 480),
            (1024, 576),
            (1920, 1080),   # 1080p
            (3840, 2160),   # 4K
            (2560, 1440),   # QHD
        ]
        self.synthetic_weights = np.array([
            0.16,   # 1024x1024 (GPT-4V, SDXL, etc.)
            0.13,   # 2048x2048 (Midjourney)
            0.12,   # 512x512
            0.11,   # 768x768
            0.05,   # 640x640
            0.04,   # 896x896
            0.01,   # 1536x1536
            0.01,   # 800x800
            0.01,   # 960x960
            0.07,   # 512x768
            0.07,   # 768x512
            0.03,   # 512x1024
            0.03,   # 1024x512
            0.01,   # 512x896
            0.01,   # 896x512
            0.02,   # 720x1280
            0.02,   # 1280x720
            0.01,   # 832x1104
            0.01,   # 1104x832
            0.01,   # 480x832
            0.01,   # 480x848
            0.01,   # 720x720
            0.01,   # 854x480
            0.01,   # 1024x576
            0.01,   # 1920x1080
            0.005,  # 3840x2160
            0.005,  # 2560x1440
        ])
        self.synthetic_weights = self.synthetic_weights / self.synthetic_weights.sum()

    def sample_resolution(self, media_type: MediaType) -> Tuple[int, int]:
        """Sample a resolution based on the media type.
        
        Args:
            media_type: The type of media (REAL or SYNTHETIC).
            
        Returns:
            A tuple of (width, height) representing the sampled resolution.
        """
        if media_type == MediaType.REAL:
            idx = np.random.choice(len(self.real_resolutions), p=self.real_weights)
            return self.real_resolutions[idx]
        idx = np.random.choice(len(self.synthetic_resolutions), p=self.synthetic_weights)
        return self.synthetic_resolutions[idx]

    def sample_cross_domain_resolution(self, media_type: MediaType) -> Tuple[int, int]:
        """Sample a resolution from the opposite domain of the media type.
        
        Args:
            media_type: The type of media (REAL or SYNTHETIC).
            
        Returns:
            A tuple of (width, height) representing the sampled resolution.
        """
        if media_type == MediaType.REAL:
            idx = np.random.choice(len(self.synthetic_resolutions), p=self.synthetic_weights)
            return self.synthetic_resolutions[idx]
        idx = np.random.choice(len(self.real_resolutions), p=self.real_weights)
        return self.real_resolutions[idx]