import io
import traceback
import tempfile

import av
import json
import bittensor as bt
import numpy as np
import torch
from fastapi import APIRouter, Depends, Request
from PIL import Image

from neurons.miner.base_miner import BaseMiner, extract_testnet_metadata
from neurons.miner.detector import Detector
from bitmind.types import MinerType


class DetectionMiner(BaseMiner):
    """Miner specialized for image and video detection tasks."""

    def initialize_models(self):
        """Initialize the detection models."""
        self.detector = Detector(self.config)

    def get_miner_type(self):
        return MinerType.DETECTOR.value

    def setup_routes(self, router: APIRouter):
        """Setup detection-specific routes."""
        router.add_api_route(
            "/detect_image",
            self.detect_image,
            dependencies=[Depends(self.determine_epistula_version_and_verify)],
            methods=["POST"],
        )
        router.add_api_route(
            "/detect_video",
            self.detect_video,
            dependencies=[Depends(self.determine_epistula_version_and_verify)],
            methods=["POST"],
        )

    async def detect_image(self, request: Request):
        """Handle image detection requests."""
        content_type = request.headers.get("Content-Type", "application/octet-stream")
        image_data = await request.body()

        signed_by = request.headers.get("Epistula-Signed-By", "")[:8]
        bt.logging.info(
            "\u2713",
            f"Received image ({len(image_data)} bytes) from {signed_by}, type: {content_type}",
        )

        if content_type not in ("image/jpeg", "application/octet-stream"):
            bt.logging.warning(
                f"Unexpected content type: {content_type}, expected image/jpeg"
            )

        testnet_metadata = extract_testnet_metadata(request.headers)
        if len(testnet_metadata) > 0:
            bt.logging.info(json.dumps(testnet_metadata, indent=2))

        try:
            image_array = np.array(Image.open(io.BytesIO(image_data)))
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

            ### PREDICT - update the Detector class with your own model and preprocessing
            pred = self.detector.detect(image_tensor, "image")
            return {"status": "success", "prediction": pred.tolist()}

        except Exception as e:
            bt.logging.error(f"Error processing image: {e}")
            bt.logging.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    async def detect_video(self, request: Request):
        """Handle video detection requests."""
        content_type = request.headers.get("Content-Type", "application/octet-stream")
        video_data = await request.body()
        signed_by = request.headers.get("Epistula-Signed-By", "")[:8]
        bt.logging.info(
            f"Received video ({len(video_data)} bytes) from {signed_by}, type: {content_type}",
        )
        if content_type not in ("video/mp4", "video/mpeg", "application/octet-stream"):
            bt.logging.warning(
                f"Unexpected content type: {content_type}, expected video/mp4 or video/mpeg"
            )

        testnet_metadata = extract_testnet_metadata(request.headers)
        if len(testnet_metadata) > 0:
            bt.logging.info(json.dumps(testnet_metadata, indent=2))

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
                temp_path = temp_file.name
                temp_file.write(video_data)
                temp_file.flush()

                with av.open(temp_path) as container:
                    video_stream = next(
                        (s for s in container.streams if s.type == "video"), None
                    )
                    if not video_stream:
                        raise ValueError("No video stream found")
                    try:
                        codec_info = (
                            f"name: {video_stream.codec.name}"
                            if hasattr(video_stream, "codec")
                            else "unknown"
                        )
                        bt.logging.info(f"Video codec: {codec_info}")
                    except Exception as codec_err:
                        bt.logging.warning(
                            f"Could not get codec info: {str(codec_err)}"
                        )
                    duration = container.duration / 1000000 if container.duration else 0
                    width = video_stream.width
                    height = video_stream.height
                    fps = video_stream.average_rate
                    bt.logging.info(
                        f"Video dimensions: ({width}, {height}), fps: {fps}, duration: {duration:.2f}s"
                    )
                    frames = []
                    for frame in container.decode(video=0):
                        img_array = frame.to_ndarray(format="rgb24")
                        frames.append(img_array)
                    bt.logging.info(f"Extracted {len(frames)} frames")
                    if not frames:
                        raise ValueError("No frames could be extracted from the video")
                    video_array = np.stack(frames)
                    video_tensor = torch.permute(
                        torch.from_numpy(video_array), (3, 0, 1, 2)  # (C, T, H, W)
                    )

            ### PREDICT - update the Detector class with your own model and preprocessing
            pred = self.detector.detect(video_tensor, "video")
            return {"status": "success", "prediction": pred.tolist()}
        except Exception as e:
            bt.logging.error(f"Error processing video: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    try:
        miner = DetectionMiner()
        miner.run()
    except Exception as e:
        bt.logging.error(str(e))
        bt.logging.error(traceback.format_exc())
    exit()
