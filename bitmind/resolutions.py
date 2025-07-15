import numpy as np
from typing import Tuple

from bitmind.types import MediaType


"""Canonical real image resolutions derived from analysis of a real image dataset.

Images were grouped by aspect ratio, then each image's resolution was
snapped to the most common value within ±8 pixels for that aspect ratio.
For each aspect ratio, we kept 1 bin per 0.5% of dataset frequency (no
minimum), resulting in a concise and representative set of (width,
height) tuples for use in the ResolutionSampler.

Synthetic image resolutions are based on common default sizes used by
popular generative models (e.g., Stable Diffusion, Midjourney, GPT-4V,
etc.).
"""


class ResolutionSampler:
    """Samples image resolutions from realistic distributions based on media type."""

    def __init__(self) -> None:
        """Initialize the ResolutionSampler with real and synthetic resolution data."""
        self.real_resolutions = [
            (348, 348),
            (400, 300),
            (400, 320),
            (400, 400),
            (400, 600),
            (460, 345),
            (480, 270),
            (480, 360),
            (480, 640),
            (500, 333),
            (500, 375),
            (500, 500),
            (500, 750),
            (564, 846),
            (600, 400),
            (600, 450),
            (600, 600),
            (600, 800),
            (600, 900),
            (630, 420),
            (640, 360),
            (640, 480),
            (640, 640),
            (640, 960),
            (660, 440),
            (700, 525),
            (700, 700),
            (720, 480),
            (720, 540),
            (736, 552),
            (750, 500),
            (750, 750),
            (800, 400),
            (800, 450),
            (800, 500),
            (800, 533),
            (800, 534),
            (800, 600),
            (800, 800),
            (900, 600),
            (960, 640),
            (960, 720),
            (1000, 600),
            (1000, 667),
            (1000, 750),
            (1023, 682),
            (1024, 576),
            (1024, 683),
            (1080, 720),
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
        ]
        self.synthetic_weights = np.array([
            0.15,   # 1024x1024 (DALL-E default, SD default) - most common
            0.08,   # 2048x2048 (SD Core 1.5MP equivalent) - high quality
            0.08,   # 512x512 - DALL-E 2 option
            0.06,   # 768x768 - SDXL base resolution
            0.06,   # 640x640 - common smaller size
            0.05,   # 896x896 - SDXL intermediate
            0.05,   # 800x800 - square variant
            0.05,   # 960x960 - larger square
            0.05,   # 512x768 - portrait
            0.05,   # 768x512 - landscape
            0.05,   # 512x1024 - tall portrait
            0.05,   # 1024x512 - wide landscape
            0.04,   # 1536x1024 - DALL-E landscape
            0.04,   # 1024x1536 - DALL-E portrait
            0.03,   # 1792x1024 - DALL-E 3 landscape
            0.03,   # 1024x1792 - DALL-E 3 portrait
            0.03,   # 832x1104 - SDXL portrait
            0.03,   # 1104x832 - SDXL landscape
            0.02,   # 480x832 - smaller portrait
            0.02,   # 480x848 - smaller portrait variant
            0.02,   # 720x720 - smaller square
            0.02,   # 854x480 - mobile landscape
            0.02,   # 1024x576 - widescreen
            0.02,   # 1920x1080 - HD video
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
            idx = np.random.choice(
                len(self.real_resolutions), p=self.real_weights
            )
            return self.real_resolutions[idx]
        idx = np.random.choice(
            len(self.synthetic_resolutions), p=self.synthetic_weights
        )
        return self.synthetic_resolutions[idx]

    def sample_cross_domain_resolution(self, media_type: MediaType) -> Tuple[int, int]:
        """Sample a resolution from the opposite domain of the media type.

        Args:
            media_type: The type of media (REAL or SYNTHETIC).

        Returns:
            A tuple of (width, height) representing the sampled resolution.
        """
        if media_type == MediaType.REAL:
            idx = np.random.choice(
                len(self.synthetic_resolutions), p=self.synthetic_weights
            )
            return self.synthetic_resolutions[idx]
        idx = np.random.choice(
            len(self.real_resolutions), p=self.real_weights
        )
        return self.real_resolutions[idx]

    def closest_resolution_within_scale(self, original_size, candidate_resolutions, max_scale=2.0):
        """
        Find the closest resolution in candidate_resolutions to original_size that does not require upscaling by more than max_scale.

        Args:
            original_size: tuple (width, height) of the original image
            candidate_resolutions: list of (width, height) tuples
            max_scale: maximum allowed upscaling factor

        Returns:
            (width, height) tuple of the closest valid resolution, or None if none found
        """
        orig_w, orig_h = original_size
        best = None
        best_dist = float('inf')
        for w, h in candidate_resolutions:
            scale = max(w / orig_w, h / orig_h)
            if scale <= max_scale:
                dist = (w - orig_w) ** 2 + (h - orig_h) ** 2
                if dist < best_dist:
                    best = (w, h)
                    best_dist = dist
        return best