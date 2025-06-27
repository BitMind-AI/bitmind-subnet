import traceback
import bittensor as bt
import functools
import json
import os
import numpy as np
from typing import Any, Dict, Union, List
from enum import Enum


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
