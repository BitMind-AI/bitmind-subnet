import io
import traceback

import json
import bittensor as bt
import numpy as np
import torch
from fastapi import APIRouter, Depends, Request, Response
from PIL import Image

from neurons.miner.base_miner import BaseMiner, extract_testnet_metadata
from neurons.miner.segmenter import Segmenter
from bitmind.types import MinerType


class SegmentationMiner(BaseMiner):
    """Miner specialized for image segmentation tasks."""

    def initialize_models(self):
        """Initialize the segmentation models."""
        self.segmenter = Segmenter(self.config)

    def get_miner_type(self):
        return MinerType.SEGMENTER.value

    def setup_routes(self, router: APIRouter):
        """Setup segmentation-specific routes."""
        router.add_api_route(
            "/segment_image",
            self.segment_image,
            dependencies=[Depends(self.determine_epistula_version_and_verify)],
            methods=["POST"],
        )

    async def segment_image(self, request: Request):
        """Handle image segmentation requests."""
        content_type = request.headers.get("Content-Type", "application/octet-stream")
        image_data = await request.body()

        signed_by = request.headers.get("Epistula-Signed-By", "")[:8]
        bt.logging.info(
            "\u2713",
            f"Received image for segmentation ({len(image_data)} bytes) from {signed_by}, type: {content_type}",
        )

        if content_type not in ("image/jpeg", "application/octet-stream"):
            bt.logging.warning(
                f"Unexpected content type: {content_type}, expected image/jpeg"
            )

        testnet_metadata, gt_mask = extract_testnet_metadata(request.headers)
        if len(testnet_metadata) > 0:
            bt.logging.info(json.dumps(testnet_metadata, indent=2))

        try:
            image_array = np.array(Image.open(io.BytesIO(image_data)))
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

            ### SEGMENT - update the Segmenter class with your own model and preprocessing
            heatmap = self.segmenter.segment(image_tensor)

            # If testnet mask is provided, compute IOU for validation
            if gt_mask is not None:
                pred_mask = (heatmap > 0.5).astype(np.uint8)

                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                iou = intersection / union if union > 0 else 0.0
                bt.logging.info(f"Testnet mask IOU: {iou:.4f}")
            
            heatmap_bytes = heatmap.astype(np.float16).tobytes()
            
            headers = {
                "X-Mask-Shape": f"{heatmap.shape[0]},{heatmap.shape[1]}",
                "X-Mask-Dtype": str(heatmap.dtype),
                "Content-Type": "application/octet-stream"
            }

            return Response(
                content=heatmap_bytes,
                headers=headers,
                status_code=200
            )

        except Exception as e:
            bt.logging.error(f"Error processing image segmentation: {e}")
            bt.logging.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    try:
        miner = SegmentationMiner()
        miner.run()
    except Exception as e:
        bt.logging.error(str(e))
        bt.logging.error(traceback.format_exc())
    exit()
