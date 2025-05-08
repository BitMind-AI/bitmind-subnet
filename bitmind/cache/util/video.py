import json
import subprocess
from typing import Dict, Any, Optional

import ffmpeg


def get_video_duration(video_path: str) -> float:
    """Get the duration of a video file in seconds.

    Args:
        video_path: Path to the video file

    Returns:
        Duration in seconds

    Raises:
        Exception: If the duration cannot be determined
    """
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe["format"]["duration"])
        return duration
    except Exception as e:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    video_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            data = json.loads(result.stdout)
            duration = float(data["format"]["duration"])
            return duration
        except Exception as sub_e:
            raise Exception(f"Failed to get video duration: {e}, {sub_e}")


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Get comprehensive metadata from a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing metadata such as:
        - duration: Duration in seconds
        - fps: Frames per second
        - width: Video width in pixels
        - height: Video height in pixels
        - codec: Video codec
        - bitrate: Video bitrate
    """
    try:
        probe = ffmpeg.probe(video_path)

        format_info = probe["format"]

        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )

        audio_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "audio"),
            None,
        )

        metadata = {
            "duration": float(format_info.get("duration", 0)),
            "size_bytes": int(format_info.get("size", 0)),
            "bit_rate": (
                int(format_info.get("bit_rate", 0))
                if "bit_rate" in format_info
                else None
            ),
            "format": format_info.get("format_name"),
        }

        if video_stream:
            fps = _parse_frame_rate(video_stream)

            metadata.update(
                {
                    "fps": fps,
                    "width": int(video_stream.get("width", 0)),
                    "height": int(video_stream.get("height", 0)),
                    "codec": video_stream.get("codec_name"),
                    "pix_fmt": video_stream.get("pix_fmt"),
                    "has_video": True,
                }
            )
        else:
            metadata["has_video"] = False

        if audio_stream:
            metadata.update(
                {
                    "audio_codec": audio_stream.get("codec_name"),
                    "sample_rate": audio_stream.get("sample_rate"),
                    "channels": audio_stream.get("channels"),
                    "has_audio": True,
                }
            )
        else:
            metadata["has_audio"] = False

        return metadata

    except Exception as e:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration,size,bit_rate:stream=width,height,codec_name,codec_type,r_frame_rate,avg_frame_rate",
                    "-of",
                    "json",
                    video_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            data = json.loads(result.stdout)

            format_info = data.get("format", {})
            streams = data.get("streams", [])

            video_stream = next(
                (stream for stream in streams if stream.get("codec_type") == "video"),
                None,
            )
            audio_stream = next(
                (stream for stream in streams if stream.get("codec_type") == "audio"),
                None,
            )

            metadata = {
                "duration": float(format_info.get("duration", 0)),
                "size_bytes": int(format_info.get("size", 0)),
                "bit_rate": (
                    int(format_info.get("bit_rate", 0))
                    if "bit_rate" in format_info
                    else None
                ),
                "format": format_info.get("format_name"),
            }

            if video_stream:
                fps = _parse_frame_rate(video_stream)

                metadata.update(
                    {
                        "fps": fps,
                        "width": int(video_stream.get("width", 0)),
                        "height": int(video_stream.get("height", 0)),
                        "codec": video_stream.get("codec_name"),
                        "has_video": True,
                    }
                )
            else:
                metadata["has_video"] = False

            if audio_stream:
                metadata.update(
                    {
                        "audio_codec": audio_stream.get("codec_name"),
                        "sample_rate": audio_stream.get("sample_rate"),
                        "channels": audio_stream.get("channels"),
                        "has_audio": True,
                    }
                )
            else:
                metadata["has_audio"] = False

            return metadata

        except Exception as sub_e:
            return {
                "duration": 0,
                "has_video": False,
                "has_audio": False,
                "error": f"Failed to get video metadata: {e}, {sub_e}",
            }


def _parse_frame_rate(video_stream: Dict[str, Any]) -> Optional[float]:
    """Parse frame rate from video stream information."""
    fps = None
    if "r_frame_rate" in video_stream:
        try:
            num, den = video_stream["r_frame_rate"].split("/")
            fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            pass

    if fps is None and "avg_frame_rate" in video_stream:
        try:
            num, den = video_stream["avg_frame_rate"].split("/")
            if float(den) > 0:
                fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            pass

    return fps


def seconds_to_str(seconds):
    """Convert seconds to formatted time string (HH:MM:SS)."""
    seconds = int(float(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"
