import json
import subprocess
import math
import ffmpeg
from typing import Dict, Any, Optional, Union, Tuple


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


def get_video_metadata(video_path: str, max_fps: float = 30.0) -> Dict[str, Any]:
    """Get comprehensive metadata from a video file with sanity checks.

    Args:
        video_path: Path to the video file
        max_fps: Maximum reasonable FPS value (default: 60.0)

    Returns:
        Dictionary containing metadata with sanity-checked values
    """
    try:
        ffprobe_fields = (
            "format=duration,size,bit_rate,format_name:"
            "stream=width,height,codec_name,codec_type,"
            "r_frame_rate,avg_frame_rate,pix_fmt,sample_rate,channels"
        )
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                ffprobe_fields,
                "-of",
                "json",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,  # This will raise CalledProcessError if ffprobe fails
        )

        data = json.loads(result.stdout)

        # Extract basic format information
        format_info = data.get("format", {})
        streams = data.get("streams", [])

        # Find video and audio streams
        video_stream = next(
            (s for s in streams if s.get("codec_type") == "video"), None
        )
        audio_stream = next(
            (s for s in streams if s.get("codec_type") == "audio"), None
        )

        # Build base metadata
        metadata = {
            "duration": float(format_info.get("duration", 0)),
            "size_bytes": int(format_info.get("size", 0)),
            "bit_rate": (
                int(format_info.get("bit_rate", 0))
                if "bit_rate" in format_info
                else None
            ),
            "format": format_info.get("format_name"),
            "has_video": video_stream is not None,
            "has_audio": audio_stream is not None,
        }

        # Add video stream details if present
        if video_stream:
            fps, fps_corrected, original_fps = _get_sanitized_fps(video_stream, max_fps)

            metadata.update(
                {
                    "fps": fps,
                    "width": int(video_stream.get("width", 0)),
                    "height": int(video_stream.get("height", 0)),
                    "codec": video_stream.get("codec_name"),
                    "pix_fmt": video_stream.get("pix_fmt"),
                }
            )

            if fps_corrected:
                metadata["original_fps"] = original_fps
                metadata["fps_corrected"] = True

        # Add audio stream details if present
        if audio_stream:
            metadata.update(
                {
                    "audio_codec": audio_stream.get("codec_name"),
                    "sample_rate": audio_stream.get("sample_rate"),
                    "channels": int(audio_stream.get("channels", 0)),
                }
            )

        return metadata

    except subprocess.CalledProcessError as e:
        return _create_error_metadata(f"ffprobe process failed: {e.stderr.strip()}")
    except json.JSONDecodeError:
        return _create_error_metadata("Failed to parse ffprobe output as JSON")
    except Exception as e:
        return _create_error_metadata(f"Unexpected error: {str(e)}")


def _get_sanitized_fps(
    video_stream: Dict[str, Any], max_fps: float = 60.0
) -> Tuple[float, bool, Optional[float]]:
    """Parse and sanitize frame rate from video stream information.

    Returns:
        Tuple of (sanitized_fps, was_corrected, original_fps_if_corrected)
    """
    original_fps = None
    fps_corrected = False

    # Try r_frame_rate first (usually more accurate)
    fps = _parse_frame_rate_string(video_stream.get("r_frame_rate"))

    # Fall back to avg_frame_rate if needed
    if fps is None:
        fps = _parse_frame_rate_string(video_stream.get("avg_frame_rate"))

    # Save original before correction
    if fps is not None:
        original_fps = fps

    # Sanity check and correct if needed
    if fps is None or not (0 < fps <= max_fps) or not math.isfinite(fps):
        fps_corrected = True
        fps = 30.0  # Default to a standard frame rate

    return fps, fps_corrected, original_fps if fps_corrected else None


def _parse_frame_rate_string(frame_rate_str: Optional[str]) -> Optional[float]:
    """Safely parse a frame rate string in format 'num/den'."""
    if not frame_rate_str:
        return None

    try:
        if "/" in frame_rate_str:
            num, den = frame_rate_str.split("/")
            num, den = float(num), float(den)
            if den <= 0:  # Avoid division by zero
                return None
            return num / den
        else:
            # Handle case where frame rate is just a number
            return float(frame_rate_str)
    except (ValueError, ZeroDivisionError):
        return None


def _create_error_metadata(error_message: str) -> Dict[str, Any]:
    """Create a metadata dictionary for error cases."""
    return {
        "duration": 0,
        "has_video": False,
        "has_audio": False,
        "error": error_message,
    }


def seconds_to_str(seconds):
    """Convert seconds to formatted time string (HH:MM:SS)."""
    seconds = int(float(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"
