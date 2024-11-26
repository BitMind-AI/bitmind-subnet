import asyncio
import json
import os
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from typing import List, Optional, Callable, Union, BinaryIO

import aiohttp
import aiofiles
import numpy as np
import requests
from pydantic import BaseModel
from yt_dlp import YoutubeDL

import bittensor as bt
import ffmpeg



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