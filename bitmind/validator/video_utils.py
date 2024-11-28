import tempfile
from pathlib import Path
from typing import Optional, BinaryIO

import bittensor as bt
import ffmpeg
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image

from .cache.util import seconds_to_str


def video_to_pil(video_path: str | Path) -> list[Image.Image]:
   """
   Load video as a list of PIL images.
   Args:
       video_path: Path to video file
   Returns:
       List of PIL Image objects
   """
   clip = VideoFileClip(str(video_path))
   frames = [Image.fromarray(np.array(frame)) for frame in clip.iter_frames()]
   clip.close()
   return frames


def clip_video(video_path: str, start: int, num_seconds: int) -> Optional[BinaryIO]:
    temp_fileobj = tempfile.NamedTemporaryFile(suffix=".mp4")
    try:
        (
            ffmpeg
            .input(video_path, ss=seconds_to_str(start), t=str(num_seconds))
            .output(temp_fileobj.name, vf='fps=1')
            .overwrite_output()
            .run(capture_stderr=True) 
        )
        return temp_fileobj
    except ffmpeg.Error as e:
        bt.logging.error(f"FFmpeg error: {e.stderr.decode()}") 
        raise

def get_video_duration(filename: str) -> int:
    metadata = ffmpeg.probe(filename)
    video_stream = next((stream for stream in metadata['streams'] if stream['codec_type'] == 'video'), None)
    duration = int(float(video_stream['duration']))
    return duration

def copy_audio(video_path: str) -> BinaryIO:
    temp_audiofile = tempfile.NamedTemporaryFile(suffix=".aac")
    (
        ffmpeg
        .input(video_path)
        .output(temp_audiofile.name, vn=None, acodec='copy')
        .overwrite_output()
        .run(quiet=True)
    )
    return temp_audiofile