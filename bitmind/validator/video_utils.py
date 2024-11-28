import tempfile
from pathlib import Path
from typing import Optional, BinaryIO, List, Union

import bittensor as bt
import ffmpeg
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image

from .cache.util import seconds_to_str


def video_to_pil(video_path: Union[str, Path]) -> List[Image.Image]:
    """Load video file and convert it to a list of PIL images.

    Args:
        video_path: Path to the input video file.

    Returns:
        List of PIL Image objects representing each frame of the video.
    """
    clip = VideoFileClip(str(video_path))
    frames = [Image.fromarray(np.array(frame)) for frame in clip.iter_frames()]
    clip.close()
    return frames


def clip_video(
    video_path: str,
    start: int,
    num_seconds: int
) -> Optional[BinaryIO]:
    """Extract a clip from a video file.

    Args:
        video_path: Path to the input video file.
        start: Start time in seconds.
        num_seconds: Duration of the clip in seconds.

    Returns:
        A temporary file object containing the clipped video,
        or None if the operation fails.

    Raises:
        ffmpeg.Error: If FFmpeg encounters an error during processing.
    """
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
    """Get the duration of a video file in seconds.

    Args:
        filename: Path to the video file.

    Returns:
        Duration of the video in seconds.

    Raises:
        KeyError: If video stream information cannot be found.
    """
    metadata = ffmpeg.probe(filename)
    video_stream = next(
        (stream for stream in metadata['streams']
         if stream['codec_type'] == 'video'),
        None
    )
    if not video_stream:
        raise KeyError("No video stream found in the file")
    return int(float(video_stream['duration']))


def copy_audio(video_path: str) -> BinaryIO:
    """Extract the audio stream from a video file.

    Args:
        video_path: Path to the input video file.

    Returns:
        A temporary file object containing the extracted audio stream.

    Raises:
        ffmpeg.Error: If FFmpeg encounters an error during processing.
    """
    temp_audiofile = tempfile.NamedTemporaryFile(suffix=".aac")
    (
        ffmpeg
        .input(video_path)
        .output(temp_audiofile.name, vn=None, acodec='copy')
        .overwrite_output()
        .run(quiet=True)
    )
    return temp_audiofile