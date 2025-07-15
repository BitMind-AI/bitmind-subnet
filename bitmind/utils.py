import traceback
import bittensor as bt
import functools
import json
import os
import numpy as np
from typing import Any, Dict, Union, List, Optional, Tuple
from enum import Enum

from bitmind.types import MediaType


def print_info(metagraph, hotkey, block, isMiner=True):
    uid = metagraph.hotkeys.index(hotkey)
    log = f"UID:{uid} | Block:{block} | Consensus:{metagraph.C[uid]} | "
    if isMiner:
        bt.logging.info(
            log
            + f"Stake:{metagraph.S[uid]} | Trust:{metagraph.T[uid]} | Incentive:{metagraph.I[uid]} | Emission:{metagraph.E[uid]}"
        )
        return
    bt.logging.info(log + f"VTrust:{metagraph.Tv[uid]} | ")


def fail_with_none(message: str = ""):
    def outer(func):
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                bt.logging.error(message)
                bt.logging.error(str(e))
                bt.logging.error(traceback.format_exc())
                return None

        return inner

    return outer


def on_block_interval(interval_attr_name):
    """
    Decorator for methods that should only execute at specific block intervals.

    Args:
        interval_attr_name: String name of the config attribute that specifies the interval
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, block, *args, **kwargs):
            interval = getattr(self.config, interval_attr_name)
            if interval is None:
                bt.logging.error(f"No interval found for {interval_attr_name}")
            if (
                block == 0 or block % interval == 0
            ):  # Allow execution on block 0 for initialization
                return await func(self, block, *args, **kwargs)
            return None

        return wrapper

    return decorator


class ExitContext:
    """
    Using this as a class lets us pass this to other threads
    """

    isExiting: bool = False

    def startExit(self, *_):
        if self.isExiting:
            exit()
        self.isExiting = True

    def __bool__(self):
        return self.isExiting


def get_metadata(media_path):
    """Get metadata for a media file if it exists."""
    base_path = os.path.splitext(media_path)[0]
    json_path = f"{base_path}.json"

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            bt.logging.error(f"Warning: Could not parse JSON file: {json_path}")
            return {}
    return {}


def get_file_modality(filepath: str) -> str:
    """
    Determine the type of media file based on its extension.

    Args:
        filepath: Path to the media file

    Returns:
        "image", "video", or "file" based on the file extension
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]:
        return "image"
    elif ext in [".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv"]:
        return "video"
    else:
        return "file"


def prepare_for_logging(obj: Any) -> Any:
    """
    Prepare an object for JSON serialization by converting numpy types and other
    non-serializable objects to JSON-compatible types.
    
    Args:
        obj: The object to prepare for logging
        
    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {k: prepare_for_logging(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [prepare_for_logging(item) for item in obj]
    else:
        # For any other type, try to convert to string
        try:
            return str(obj)
        except:
            return f"<non-serializable: {type(obj).__name__}>"


def sample_resolution(
    media_type: MediaType,
    cross_domain: bool = True,
    original_size: Optional[Tuple] = None,
    max_scale: float = 2.0,
    order="WH",
) -> Tuple[int, int]:
    """Sample a resolution based on the media type.

    Args:
        media_type: The type of media (REAL or SYNTHETIC).

    Returns:
        A tuple of (width, height) representing the sampled resolution.
    """
    if cross_domain:
        if media_type == MediaType.REAL:
            media_type = MediaType.SYNTHETIC
        else:
            media_type = MediaType.REAL

    candidates, weights = get_resolutions_and_weights(media_type)

    if original_size is not None and max_scale is not None:
        idxs = candidate_idxs_within_scale(original_size, candidates, max_scale)
        candidates = [res for i, res in enumerate(candidates) if i in idxs]
        weights = np.array([w for i, w in enumerate(weights) if i in idxs])
        weights = weights / weights.sum()

    idx = np.random.choice(
        len(candidates), p=weights
    )
    if order == "HW":
        return candidates[idx][::-1]
    return candidates[idx]


def candidate_idxs_within_scale(original_size, candidate_resolutions, max_scale=2.0):
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
    candidate_idxs = []
    for i, (w, h) in enumerate(candidate_resolutions):
        scale = max(w / orig_w, h / orig_h)
        if scale <= max_scale:
            candidate_idxs.append(i)
    return candidate_idxs


def get_resolutions_and_weights(media_type: MediaType):
    if media_type == MediaType.REAL:
        return get_real_resolutions_and_weights()
    else:
        return get_synthetic_resolutions_and_weights()


def get_real_resolutions_and_weights():
    """Canonical real image resolutions derived from analysis of a real image dataset.

    Images were grouped by aspect ratio, then each image's resolution was
    snapped to the most common value within ±8 pixels for that aspect ratio.
    For each aspect ratio, we kept 1 bin per 0.5% of dataset frequency (no
    minimum), resulting in a concise and representative set of (width,
    height) tuples for use in the ResolutionSampler."""
    real_resolutions = [
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
    real_weights = np.ones(len(real_resolutions)) / len(real_resolutions)
    return real_resolutions, real_weights


def get_synthetic_resolutions_and_weights():
    """
    Synthetic image resolutions are based on common default sizes used by
    popular generative models (e.g., Stable Diffusion, Midjourney, GPT-4V,
    etc.)."""
    synthetic_resolutions = [
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
    # Create exponentially decaying weights for synthetic resolutions
    # First few resolutions (1024x1024, 2048x2048, etc) get higher weights
    #decay_rate = 0.85  # Controls how quickly weights drop off
    #raw_weights = np.power(decay_rate, np.arange(len(self.synthetic_resolutions)))
    #self.synthetic_weights = raw_weights / raw_weights.sum()  # Normalize to sum to 1
    synthetic_weights = np.array([
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
    synthetic_weights = synthetic_weights / synthetic_weights.sum()
    return synthetic_resolutions, synthetic_weights