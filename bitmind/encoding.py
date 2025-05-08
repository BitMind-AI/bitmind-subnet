import numpy as np
import cv2
import ffmpeg
import os
import tempfile


def image_to_bytes(img):
    """Convert image array to bytes using JPEG encoding.

    Args:
        img (np.ndarray): Image array of shape (C, H, W)
            Can be float32 [0,1] or uint8 [0,255]

    Returns:
        bytes: JPEG encoded image bytes
        str: Content type 'image/jpeg'
    """

    # Convert float32 [0,1] to uint8 [0,255] if needed
    if img.dtype == np.float32:
        img = (img * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise ValueError(f"Image must be float32 or uint8, got {img.dtype}")

    if img.shape[0] < img.shape[2]:
        img = np.transpose(img, (1, 2, 0))  # CHW to HWC

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # BGR for cv2

    success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not success:
        raise RuntimeError("Failed to encode image to JPEG")

    return buffer.tobytes(), "image/jpeg"


def video_to_bytes(video, fps=None):
    """Convert video to in-memory AVI (PNG-compressed) bytes."""
    if video.dtype == np.float32:
        video = (video * 255).clip(0, 255).astype(np.uint8)
    elif video.dtype != np.uint8:
        raise ValueError(f"Unsupported dtype: {video.dtype}")

    fps = fps if fps is not None else 30

    # TCHW → THWC
    if video.shape[1] <= 4 and video.shape[3] > 4:
        video = np.transpose(video, (0, 2, 3, 1))

    if video.ndim != 4 or video.shape[3] not in (1, 3):
        raise ValueError(f"Expected shape (T, H, W, C), got {video.shape}")

    T, H, W, C = video.shape

    for i, frame in enumerate(video):
        if frame.shape != (H, W, C):
            raise ValueError(f"Inconsistent shape at frame {i}: {frame.shape}")

    if C == 3:
        video = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in video])

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "input.raw")
            video_path = os.path.join(tmpdir, "output.mp4")
            with open(raw_path, "wb") as f:
                for i, frame in enumerate(video):
                    if frame.shape != (H, W, C):
                        raise ValueError(f"Frame {i} shape mismatch: {frame.shape}")
                    if frame.dtype != np.uint8:
                        raise ValueError(f"Frame {i} has dtype {frame.dtype}")
                    f.write(frame.tobytes())

            try:
                (
                    ffmpeg.input(
                        raw_path,
                        format="rawvideo",
                        pix_fmt="rgb24",
                        s=f"{W}x{H}",
                        r=fps,
                    )
                    .output(video_path, vcodec="mjpeg", q=7)
                    .global_args("-y", "-hide_banner", "-loglevel", "error")
                    .run()
                )
            except ffmpeg.Error as e:
                raise RuntimeError(
                    f"FFmpeg encoding failed:\n{e.stderr.decode(errors='ignore')}"
                ) from e

            with open(video_path, "rb") as f:
                video_bytes = f.read()

        return video_bytes, "video/mp4"
    except Exception as e:
        raise RuntimeError(f"Video encoding failed: {e}") from e


def media_to_bytes(media, fps=30):
    """Convert image or video array to bytes, using PNG encoding for both.

    Args:
        media (np.ndarray): Either:
            - Image array of shape (C, H, W)
            - Video array of shape (T, C, H, W)
            Can be float32 [0,1] or uint8 [0,255]
        fps (int): Frames per second for video (default: 30)

    Returns:
        bytes: Encoded media bytes
        str: Content type (either 'image/png' or 'video/avi')
    """
    if len(media.shape) == 3:  # Image
        return image_to_bytes(media)
    elif len(media.shape) == 4:  # Video
        return video_to_bytes(media, fps)
    else:
        raise ValueError(
            f"Invalid media shape: {media.shape}. Expected (C,H,W) for image or (T,C,H,W) for video."
        )
