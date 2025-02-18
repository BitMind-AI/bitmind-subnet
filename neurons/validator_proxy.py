from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import APIKeyHeader
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from typing import Optional, Dict, List, Union, Any
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import bittensor as bt
import numpy as np
import uvicorn
import base64
import tempfile
import asyncio
import cv2
import os
import httpx
import time
import socket
from functools import lru_cache

from bitmind.validator.config import TARGET_IMAGE_SIZE
from bitmind.utils.image_transforms import get_base_transforms
from bitmind.protocol import prepare_synapse
from bitmind.utils.uids import get_random_uids
from bitmind.validator.proxy import ProxyCounter

# Constants
AUTH_HEADER = APIKeyHeader(name="Authorization")
FRAME_FORMAT = "RGB"
DEFAULT_TIMEOUT = 30
DEFAULT_SAMPLE_SIZE = 256


class MediaProcessor:
    """Handles processing of images and videos"""
    def __init__(self, target_size: tuple):
        self.transforms = get_base_transforms(target_size)

    def process_image(self, b64_image: str) -> Any:
        """Process base64 encoded image"""
        image_bytes = base64.b64decode(b64_image)
        image = Image.open(BytesIO(image_bytes))
        return self.transforms(image)

    def process_video(self, video_data: bytes) -> List[Any]:
        """Process raw video bytes into transformed frames"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as temp_file:
            temp_file.write(video_data)
            temp_file.flush()

            cap = cv2.VideoCapture(temp_file.name)
            if not cap.isOpened():
                raise ValueError("Failed to open video stream")

            try:
                frames = []
                while True:
                    success, frame = cap.read()
                    if not success:
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(rgb_frame)
                    frames.append(pil_frame)

                if not frames:
                    raise ValueError("No frames extracted from video")

                return self.transforms(frames)
            finally:
                cap.release()

class PredictionService:
    """Handles interaction with miners for predictions"""
    def __init__(self, validator, dendrite):
        self.validator = validator
        self.dendrite = dendrite
        self.metagraph = validator.metagraph

    async def get_predictions(
        self, 
        data: Any, 
        modality: str,
        timeout: int = DEFAULT_TIMEOUT
    ) -> tuple[List[float], List[int]]:
        """Get predictions from miners"""
        miner_uids = self._get_miner_uids()

        predictions = await self.dendrite(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=prepare_synapse(data, modality=modality),
            deserialize=True,
            timeout=timeout
        )

        valid_indices = [i for i, v in enumerate(predictions) if v != -1.]
        if not valid_indices:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No valid predictions received"
            )

        valid_preds = np.array(predictions)[valid_indices]
        valid_uids = np.array(miner_uids)[valid_indices]

        return valid_preds.tolist(), valid_uids.tolist()

    def _get_miner_uids(self) -> List[int]:
        """Get list of miner UIDs to query"""
        uids = self.validator.last_responding_miner_uids
        if not uids:
            bt.logging.warning("No recent miner UIDs found, sampling random UIDs")
            uids = get_random_uids(self.validator, k=DEFAULT_SAMPLE_SIZE)
        return uids

    def get_rich_data(self, uids: List[int]) -> Dict[str, List]:
        """Get additional miner metadata"""
        return {
            'uids': [int(uid) for uid in uids],
            'ranks': [float(self.metagraph.R[uid]) for uid in uids],
            'incentives': [float(self.metagraph.I[uid]) for uid in uids],
            'emissions': [float(self.metagraph.E[uid]) for uid in uids],
            'hotkeys': [str(self.metagraph.hotkeys[uid]) for uid in uids],
            'coldkeys': [str(self.metagraph.coldkeys[uid]) for uid in uids]
        }

class ValidatorProxy:
    """FastAPI server that proxies requests to validator miners"""
    def __init__(self, validator):
        self.validator = validator
        self.media_processor = MediaProcessor(TARGET_IMAGE_SIZE)
        self.dendrite = bt.dendrite(wallet=validator.wallet)
        self.prediction_service = PredictionService(validator, self.dendrite)
        self.metrics = ProxyCounter(os.path.join(validator.config.neuron.full_path, "proxy_counter.json"))
        self.app = FastAPI(title="Validator Proxy", version="1.0.0")
        self._configure_routes()

        if self.validator.config.proxy.port:
            self.auth_verifier = self._setup_auth()
            self.start()

    def _configure_routes(self):
        """Configure FastAPI routes"""
        self.app.add_api_route(
            "/forward_image",
            self.handle_image_request,
            methods=["POST"],
            dependencies=[Depends(self.verify_auth)]
        )
        self.app.add_api_route(
            "/forward_video", 
            self.handle_video_request,
            methods=["POST"],
            dependencies=[Depends(self.verify_auth)]
        )
        self.app.add_api_route(
            "/healthcheck",
            self.healthcheck,
            methods=["GET"],
            dependencies=[Depends(self.verify_auth)]
        )

    def _setup_auth(self) -> callable:
        """Set up authentication verifier using synchronous HTTP client"""
        with httpx.Client() as client:
            response = client.post(
                f"{self.validator.config.proxy.proxy_client_url}/get-credentials",
                json={
                    "postfix": f":{self.validator.config.proxy.port}" if self.validator.config.proxy.port else "",
                    "uid": self.validator.uid
                },
                timeout=DEFAULT_TIMEOUT
            )
            creds = response.json()

        signature = base64.b64decode(creds["signature"])
        message = creds["message"]

        def verify(key_bytes: bytes) -> bool:
            try:
                key = Ed25519PublicKey.from_public_bytes(key_bytes)
                key.verify(signature, message.encode())
                return True
            except InvalidSignature:
                return False

        return verify

    async def verify_auth(self, auth: str = Depends(AUTH_HEADER)) -> None:
        """Verify authentication token"""
        try:
            key_bytes = base64.b64decode(auth)
            if not self.auth_verifier(key_bytes):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication token"
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e)
            )

    async def handle_image_request(self, request: Request) -> Dict[str, Any]:
        """Handle image processing requests"""
        payload = await request.json()

        try:
            image = self.media_processor.process_image(payload['image'])
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process image: {str(e)}"
            )

        predictions, uids = await self.prediction_service.get_predictions(
            image, 
            modality='image'
        )

        response = {
            'preds': predictions,
            'fqdn': socket.getfqdn()
        }

        # add rich data if requested
        if payload.get('rich', '').lower() == 'true':
            response.update(self.prediction_service.get_rich_data(uids))

        self.metrics.update(is_success=True)
        return response

    async def handle_video_request(self, request: Request) -> Dict[str, Any]:
        """Handle video processing requests"""
        video_data = await request.body()
        if not video_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing video data"
            )

        s = time.time()
        try:
            video = self.media_processor.process_video(video_data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process video: {str(e)}"
            )
        bt.logging.info(f"finished processing video in {time.time() - s:.6f}s")
        predictions, uids = await self.prediction_service.get_predictions(
            video, 
            modality='video',
        )

        response = {
            'preds': predictions,
            'fqdn': socket.getfqdn()
        }

        # add rich data if requested
        request_json = await request.json() if request.headers.get('content-type') == 'application/json' else {}
        if request_json.get('rich', '').lower() == 'true':
            response.update(self.prediction_service.get_rich_data(uids))

        self.metrics.update(is_success=True)
        return response

    async def healthcheck(self, request: Request) -> Dict[str, str]:
        """Health check endpoint"""
        return {'status': 'healthy'}

    def start(self):
        """Start the FastAPI server"""
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(
            uvicorn.run, self.app, host="0.0.0.0", port=self.validator.config.proxy.port
        )
