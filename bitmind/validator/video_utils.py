from typing import Optional, BinaryIO, List
from pydantic import BaseModel
from yt_dlp import YoutubeDL
import bittensor as bt
import numpy as np
import tempfile
import requests
import ffmpeg
import random
import time
import re
import json
import os

VIDEO_DOWNLOAD_LENGTH = 60


def get_description(yt: YoutubeDL, video_path: str) -> str:
    description = yt.title
    if yt.description:
        description += f"\n\n{yt.description}"
    return description


class VideoMetadata(BaseModel):
    """
    A model class representing YouTube video metadata.
    """
    video_id: str
    video_path: str
    description: str
    views: int
    length: int
    start_time: int
    end_time: int

    def __repr_args__(self):
        parent_args = super().__repr_args__()
        return (
            [(a, v) for a, v in parent_args]
        )


def search_and_download_youtube_videos(query: str, num_videos: int) -> List[VideoMetadata]:
    """
    Search YouTube for videos matching the given query and return a list of VideoMetadata objects.

    Args:
        query (str): The query to search for.
        num_videos (int, optional): The number of videos to return.

    Returns:
        List[VideoMetadata]: A list of VideoMetadata objects representing the search results.
    """
    # fetch more videos than we need
    results = search_videos(query, max_results=int(num_videos * 1.5))
    video_metas = []
    #try:
    # take the first N that we need
    for result in results:
        start = time.time()
        max_start = np.clip(result.length - VIDEO_DOWNLOAD_LENGTH, result.length, None)
        video_start = random.choice(list(range(0, max_start)))
        download_path = download_youtube_video(
            result.video_id,
            start=video_start,
            end=min(result.length, video_start + VIDEO_DOWNLOAD_LENGTH)  # download a random 5 minutes
        )
        if download_path:
            try:
                result.length = get_video_duration(download_path.name)
                bt.logging.info(f"Downloaded video {result.video_id} ({result.length}) in {time.time() - start} seconds")
                description = get_description(result, download_path)
                video_metas.append(VideoMetadata(
                    video_id=result.video_id,
                    video_path=download_path,
                    description=description,
                    views=result.views,
                    length=result.length,
                    start_time=start,
                    end_time=end,
                ))
            finally:
                download_path.close()
        if len(video_metas) == num_videos:
            break

    #except Exception as e:
    #    bt.logging.error(f"Error searching for videos: {e}")

    return video_metas


def seconds_to_str(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def clip_video(video_path: str, start: int, num_seconds: int) -> Optional[BinaryIO]:
    temp_fileobj = tempfile.NamedTemporaryFile(suffix=".mp4")
    (
        ffmpeg
        .input(video_path, ss=seconds_to_str(start), t=str(num_seconds))
        .output(temp_fileobj.name, c="copy", vf='fps=1/')  # copy flag prevents decoding and re-encoding
        .overwrite_output()
        .run(quiet=True)
    )
    return temp_fileobj


def skip_live(info_dict):
    """
    function to skip downloading if it's a live video (yt_dlp doesn't respect the 20 minute 
    download limit for live videos), and we don't want to hang on an hour long stream
    """
    if info_dict.get("is_live"):
        return "Skipping live video"
    return None


class YoutubeResult(BaseModel):
    video_id: str
    title: str
    description: Optional[str]
    length: int
    views: int


def search_videos(query, max_results=8):
    videos = []
    ydl_opts = {
        "format": "worst",
        "dumpjson": True,
        "extract_flat": True,
        "quiet": True,
        "simulate": True,
        "match_filter": skip_live,
    }
    with YoutubeDL(ydl_opts) as ydl:
        try:
            search_query = f"ytsearch{max_results}:{query}"
            result = ydl.extract_info(search_query, download=False)
            if "entries" in result and result["entries"]:
                videos = [
                    YoutubeResult(
                        video_id=entry["id"],
                        title=entry["title"],
                        description=entry.get("description"),
                        length=(int(entry.get("duration")) if entry.get("duration") else VIDEO_DOWNLOAD_LENGTH),
                        views=(entry.get("view_count") if entry.get("view_count") else 0),
                    ) for entry in result["entries"]
                ]
        except Exception as e:
            bt.logging.warning(f"Error searching for videos: {e}")
            return []
    return videos


def get_video_duration(filename: str) -> int:
    metadata = ffmpeg.probe(filename)
    video_stream = next((stream for stream in metadata['streams'] if stream['codec_type'] == 'video'), None)
    duration = int(float(video_stream['duration']))
    return duration


class IPBlockedException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class FakeVideoException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def is_valid_youtube_id(youtube_id: str) -> bool:
    return youtube_id is not None and len(youtube_id) == 11


def download_youtube_video(
    video_id: str, start: Optional[int]=None, end: Optional[int]=None, proxy: Optional[str]=None
) -> Optional[BinaryIO]:
    if not is_valid_youtube_id(video_id):
        raise FakeVideoException(f"Invalid Youtube video ID: {video_id}")

    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    temp_fileobj = tempfile.NamedTemporaryFile(suffix=".mp4")
    ydl_opts = {
        "format": "worst",  # Download the worst quality
        "outtmpl": temp_fileobj.name,  # Set the output template to the temporary file"s name
        "overwrites": True,
        "quiet": True,
        "noprogress": True,
        "match_filter": skip_live,
    }

    if start is not None and end is not None:
        ydl_opts["download_ranges"] = lambda _, __: [{"start_time": start, "end_time": end}]

    if proxy is not None:
        ydl_opts["proxy"] = proxy

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Check if the file is empty (download failed)
        if os.stat(temp_fileobj.name).st_size == 0:
            print(f"Error downloading Youtube video: {temp_fileobj.name} is empty")
            temp_fileobj.close()
            return None

        return temp_fileobj
    except Exception as e:
        temp_fileobj.close()
        if (
            "Your IP is likely being blocked by Youtube" in str(e) or
            "Requested format is not available" in str(e)
        ):
            raise IPBlockedException(e)

        # Quick check to see if miner passed an "unplayable" (sign-in required, paid video, etc.).
        fake_video = False
        try:
            result = requests.get(video_url, proxies={"https": proxy})
            json_match = re.search(r"ytInitialPlayerResponse\s*=\s*(\{(?:.*?)\})\s*;\s*<", result.text)
            if json_match:
                player_info = json.loads(json_match.group(1))
                status = player_info.get('playabilityStatus', {}).get('status', 'ok')
                unacceptable_statuses = ('UNPLAYABLE',)
                if status in unacceptable_statuses or (status == 'ERROR' and player_info['playabilityStatus'].get('reason', '').lower() == 'video unavailable'):
                    if "sign in to confirm you’re not a bot" not in result.text.lower():
                        fake_video = True
                        print(f"Fake video submitted, youtube player status [{status}]: {player_info['playabilityStatus']}")
        except Exception as fake_check_exc:
            print(f"Error sanity checking playability: {fake_check_exc}")
        if fake_video:
            raise FakeVideoException("Unplayable video provided")
        if any(fake_vid_msg in str(e) for fake_vid_msg in ["Video unavailable", "is not a valid URL", "Incomplete YouTube ID", "Unsupported URL"]):
            raise FakeVideoException(e)
        print(f"Error downloading video: {e}")
        return None


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