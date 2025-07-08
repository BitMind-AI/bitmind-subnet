import asyncio
import base64
import io
import uuid
import os
import socket
import tempfile
import time
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple

import aiohttp
import bittensor as bt
import cv2
import httpx
import numpy as np
import uvicorn
from bittensor.core.settings import SS58_FORMAT, TYPE_REGISTRY
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import (
    FastAPI,
    Request,
    HTTPException,
    Depends,
    status,
    APIRouter,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from PIL import Image
from bittensor.core.axon import FastAPIThreadedServer
from substrateinterface import SubstrateInterface

from bitmind.config import MAINNET_UID
from bitmind.encoding import media_to_bytes
from bitmind.epistula import query_miner
from bitmind.metagraph import get_miner_uids, run_block_callback_thread
from bitmind.scoring.miner_history import MinerHistory
from bitmind.transforms import get_base_transforms
from bitmind.types import Modality, NeuronType, MinerType
from bitmind.utils import on_block_interval
from neurons.base import BaseNeuron

AUTH_HEADER = APIKeyHeader(name="Authorization")
DEFAULT_TIMEOUT = 6


class MediaProcessor:
    def __init__(self, target_size: tuple = (256, 256)):
        self.target_size = target_size

    def process_image(self, b64_image: str) -> np.ndarray:
        """
        Decode base64 image and preprocess

        Args:
            b64_image: Base64 encoded image string

        Returns:
            Processed image as numpy array
        """
        try:
            image_bytes = base64.b64decode(b64_image)
            image = Image.open(io.BytesIO(image_bytes))
            transformed_image, _ = get_base_transforms(self.target_size)(np.array(image))
            image_bytes, content_type = media_to_bytes(transformed_image)
            return image_bytes, content_type

        except Exception as e:
            bt.logging.error(f"Error processing image: {e}")
            raise ValueError(f"Failed to process image: {str(e)}")

    def process_video(self, video_data: bytes, transform_frames=True) -> np.ndarray:
        """
        Process raw video bytes into frames and preprocess

        Args:
            video_data: Raw video bytes

        Returns:
            Processed video frames as numpy array
        """
        bt.logging.trace(f"Starting video processing with {len(video_data)} bytes")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_file:
            temp_file.write(video_data)
            temp_file.flush()

            cap = cv2.VideoCapture(temp_file.name)
            if not cap.isOpened():
                bt.logging.error("Failed to open video stream")
                raise ValueError("Failed to open video stream")

            try:
                frames = []
                while True:
                    success, frame = cap.read()
                    if not success:
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_frame)

                bt.logging.trace(f"Extracted {len(frames)} frames")

                if not frames:
                    bt.logging.error("No frames extracted from video")
                    raise ValueError("No frames extracted from video")

                frames = np.stack(frames)
                if transform_frames:
                    frames, _ = get_base_transforms(self.target_size)(frames)
 
                video_bytes, content_type = media_to_bytes(frames)
                return video_bytes, content_type

            except Exception as e:
                bt.logging.error(f"Error in video processing: {str(e)}")
                raise
            finally:
                cap.release()


class ValidatorProxy(BaseNeuron):
    """
    Proxy server that handles requests from external applications and forwards them to miners.
    Uses FastAPIThreadedServer for improved concurrency.
    """

    neuron_type = NeuronType.VALIDATOR_PROXY

    def __init__(self, config=None):
        """
        Initialize the proxy server.

        Args:
            validator: The validator instance that manages miners and scoring
        """
        super().__init__(config=config)

        if not (
            hasattr(self.config, "proxy")
            and hasattr(self.config.proxy, "client_url")
            and hasattr(self.config.proxy, "port")
        ):
            raise ValueError(
                "Missing proxy configuration - cannot initialize ValidatorProxy"
            )

        self.block_callbacks.extend([
            self.log_on_block,
            self.sync_miner_health_on_interval
        ])

        self.tracker_state_path = os.path.join(self.config.neuron.full_path, "state_current")
        self.tracker = MinerHistory()
        self.tracker.load_state(self.tracker_state_path)

        self.port = self.config.proxy.port
        self.external_port = self.config.proxy.external_port
        self.host = self.config.proxy.host
        self.media_processor = MediaProcessor()
        self.auth_verifier = self._setup_auth()

        self.max_connections = 50
        self.fast_api = None

        self.request_times = {
            "image": [],
            "video": [],
        }
        self.max_request_history = 100

        self.setup_app()

        bt.logging.info(f"Initialized proxy server on {self.host}:{self.port}")

    def _setup_auth(self) -> callable:
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.config.proxy.client_url}/get-credentials",
                    json={"postfix": f":{self.external_port}", "uid": self.uid},
                    timeout=DEFAULT_TIMEOUT,
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

            bt.logging.info("Authentication setup successful")
            return verify

        except Exception as e:
            bt.logging.error(f"Error setting up authentication: {e}")
            bt.logging.error("Authentication will be disabled")
            return None

    async def verify_auth(self, auth: str = Depends(AUTH_HEADER)) -> None:
        if not self.auth_verifier:
            return

        try:
            key_bytes = base64.b64decode(auth)
            if not self.auth_verifier(key_bytes):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication token",
                )
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    async def healthcheck(self, request: Request) -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    async def log_on_block(self, block):
        """
        Log avg request times

        Args:
            block: Current block number
        """
        log_items = [f"Forward Block: {self.subtensor.block}"]

        log_items.extend([
            f"Healthy: {len(self.tracker.get_healthy_miner_uids())}",
            f"Unhealthy: {len(self.tracker.get_unhealthy_miner_uids())}",
        ])

        if self.request_times.get("image"):
            avg_image_time = sum(self.request_times["image"]) / len(
                self.request_times["image"]
            )
            log_items.append(f"Avg image request: {avg_image_time:.2f}s")

        if self.request_times.get("video"):
            avg_video_time = sum(self.request_times["video"]) / len(
                self.request_times["video"]
            )
            log_items.append(f"Avg video request: {avg_video_time:.2f}s")

        bt.logging.info(" | ".join(log_items))


    def setup_app(self):
        app = FastAPI(title="BitMind Proxy Server")
        router = APIRouter()

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        router.add_api_route(
            "/forward_image",
            self.handle_image_request,
            methods=["POST"],
            dependencies=[Depends(self.verify_auth)],
        )
        router.add_api_route(
            "/forward_image_binary",
            self.handle_binary_image_request,
            methods=["POST"],
            dependencies=[Depends(self.verify_auth)],
        )
        router.add_api_route(
            "/forward_video",
            self.handle_video_request,
            methods=["POST"],
            dependencies=[Depends(self.verify_auth)],
        )
        router.add_api_route(
            "/forward_video_binary",
            self.handle_binary_video_request,
            methods=["POST"],
            dependencies=[Depends(self.verify_auth)],
        )
        router.add_api_route(
            "/healthcheck",
            self.healthcheck,
            methods=["GET"],
            dependencies=[Depends(self.verify_auth)],
        )

        app.include_router(router)

        fast_config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
            loop="asyncio",
            workers=9,
        )
        self.fast_api = FastAPIThreadedServer(config=fast_config)

    async def handle_binary_image_request(self, request: Request) -> Dict[str, Any]:
        """
        Handle raw JPEG image processing requests.

        Args:
            request: FastAPI request object with binary JPEG image data

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]
        bt.logging.trace(f"[{request_id}] Starting binary image request processing")

        try:
            image_data = await request.body()
            if not image_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Empty image data",
                )

            query_start = time.time()
            results = await self.query_miners(
                media_bytes=image_data,
                content_type="image/jpeg",
                modality=Modality.IMAGE,
                request_id=request_id,
            )
            bt.logging.debug(
                f"[{request_id}] Miners queried in {time.time() - query_start:.2f}s"
            )

            predictions, uids = self.process_query_results(results)
            response = {
                "preds": [float(p) for p in predictions],
                "fqdn": socket.getfqdn(),
            }

            # Add rich data if requested
            rich_param = request.query_params.get("rich", "").lower()
            if rich_param == "true":
                response.update(self.get_rich_data(uids))

            total_time = time.time() - start_time
            bt.logging.debug(
                f"[{request_id}] Binary image request processed in {total_time:.2f}s"
            )

            if len(self.request_times["image"]) >= self.max_request_history:
                self.request_times["image"].pop(0)
            self.request_times["image"].append(total_time)

            return response

        except Exception as e:
            bt.logging.error(f"[{request_id}] Error processing binary image request: {e}")
            bt.logging.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing request: {str(e)}",
            )

    async def handle_image_request(self, request: Request) -> Dict[str, Any]:
        """
        Handle image processing requests.

        Args:
            request: FastAPI request object with JSON body containing base64 image

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]
        bt.logging.trace(f"[{request_id}] Starting image request processing")

        try:
            payload = await request.json()
            if "image" not in payload:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing 'image' field in request body",
                )
            b64_image = payload["image"]

            proc_start = time.time()
            media_bytes, content_type = await asyncio.to_thread(
                self.media_processor.process_image, b64_image
            )
            bt.logging.trace(
                f"[{request_id}] Image processed in {time.time() - proc_start:.2f}s"
            )

            query_start = time.time()
            results = await self.query_miners(
                media_bytes=media_bytes,
                content_type=content_type,
                modality=Modality.IMAGE,
                request_id=request_id,
            )
            bt.logging.debug(
                f"[{request_id}] Miners queried in {time.time() - query_start:.2f}s"
            )

            predictions, uids = self.process_query_results(results)
            response = {
                "preds": [float(p) for p in predictions],
                "fqdn": socket.getfqdn(),
            }

            # Add rich data if requested
            if payload.get("rich", "").lower() == "true":
                response.update(self.get_rich_data(uids))

            total_time = time.time() - start_time
            bt.logging.debug(
                f"[{request_id}] Image request processed in {total_time:.2f}s"
            )

            if len(self.request_times["image"]) >= self.max_request_history:
                self.request_times["image"].pop(0)
            self.request_times["image"].append(total_time)

            return response

        except Exception as e:
            bt.logging.error(f"[{request_id}] Error processing image request: {e}")
            bt.logging.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing request: {str(e)}",
            )

    async def handle_video_request(self, request: Request) -> Dict[str, Any]:
        """
        Handle video processing requests.

        Args:
            request: FastAPI request object with form data containing video file

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]
        bt.logging.trace(f"[{request_id}] Starting video request processing")

        try:
            form = await request.form()
            if "video" not in form:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing 'video' field in form data",
                )

            video_file = form["video"]
            video_data = await video_file.read()

            if not video_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Empty video file"
                )

            rich_param = form.get("rich", "").lower()

            proc_start = time.time()
            media_bytes, content_type = await asyncio.to_thread(
                self.media_processor.process_video, video_data
            )
            bt.logging.trace(
                f"[{request_id}] Video processed in {time.time() - proc_start:.2f}s"
            )

            query_start = time.time()
            results = await self.query_miners(
                media_bytes=media_bytes,
                content_type=content_type,
                modality=Modality.VIDEO,
                request_id=request_id,
            )
            bt.logging.debug(
                f"[{request_id}] Miners queried in {time.time() - query_start:.2f}s"
            )

            predictions, uids = self.process_query_results(results)
            response = {
                "preds": [float(p) for p in predictions],
                "fqdn": socket.getfqdn(),
            }

            # Add rich data if requested
            if rich_param == "true":
                response.update(self.get_rich_data(uids))

            total_time = time.time() - start_time
            bt.logging.debug(
                f"[{request_id}] Video request processed in {total_time:.2f}s"
            )

            if len(self.request_times["video"]) >= self.max_request_history:
                self.request_times["video"].pop(0)
            self.request_times["video"].append(total_time)
            return response

        except Exception as e:
            bt.logging.error(f"Error processing video request: {e}")
            bt.logging.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing request: {str(e)}",
            )

    async def handle_binary_video_request(self, request: Request) -> Dict[str, Any]:
        """
        Handle video processing requests.

        Args:
            request: FastAPI request object with form data containing video file

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]
        bt.logging.trace(f"[{request_id}] Starting video request processing")

        try:
            form = await request.form()
            if "video" not in form:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing 'video' field in form data",
                )

            video_file = form["video"]
            video_data = await video_file.read()

            if not video_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Empty video file"
                )

            rich_param = form.get("rich", "").lower()

            proc_start = time.time()
            media_bytes, content_type = await asyncio.to_thread(
                self.media_processor.process_video, video_data, transform_frames=False
            )
            bt.logging.trace(
                f"[{request_id}] Video processed in {time.time() - proc_start:.2f}s"
            )

            query_start = time.time()
            results = await self.query_miners(
                media_bytes=media_bytes,
                content_type=content_type,
                modality=Modality.VIDEO,
                request_id=request_id,
            )
            bt.logging.debug(
                f"[{request_id}] Miners queried in {time.time() - query_start:.2f}s"
            )

            predictions, uids = self.process_query_results(results)
            response = {
                "preds": [float(p) for p in predictions],
                "fqdn": socket.getfqdn(),
            }

            # Add rich data if requested
            if rich_param == "true":
                response.update(self.get_rich_data(uids))

            total_time = time.time() - start_time
            bt.logging.debug(
                f"[{request_id}] Video request processed in {total_time:.2f}s"
            )

            if len(self.request_times["video"]) >= self.max_request_history:
                self.request_times["video"].pop(0)
            self.request_times["video"].append(total_time)
            return response

        except Exception as e:
            bt.logging.error(f"Error processing video request: {e}")
            bt.logging.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing request: {str(e)}",
            )


    async def query_miners(
        self,
        media_bytes: bytes,
        content_type: str,
        modality: Modality,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query a set of miners with the given media.

        Args:
            media_bytes: Encoded media bytes
            content_type: Media content type
            modality: Media modality (image or video)
            num_miners: Number of miners to sample

        Returns:
            List of miner responses
        """
        query_start = time.time()

        miner_uids = self.tracker.get_healthy_miner_uids()
        if len(miner_uids) == 0:
            bt.logging.warning(f"Miner health not available, defaulting to random selection")
            miner_uids = get_miner_uids(
                self.metagraph,
                self.uid,
                self.config.vpermit_tao_limit,
            )

        sample_size = min(self.config.proxy.sample_size, len(miner_uids))
        miner_uids = np.random.choice(
            miner_uids, size=sample_size, replace=False
        )
        bt.logging.trace(f"Querying {len(miner_uids)} miners")

        challenge_tasks = []
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=5,
                enable_cleanup_closed=True,
                force_close=False,
                ttl_dns_cache=300,
            )
        ) as session:
            for uid in miner_uids:
                axon_info = self.metagraph.axons[uid]
                challenge_tasks.append(
                    query_miner(
                        uid,
                        MinerType.DETECTOR,
                        media_bytes,
                        content_type,
                        modality,
                        axon_info,
                        session,
                        self.wallet.hotkey,
                        self.config.neuron.miner_total_timeout,
                        self.config.neuron.miner_connect_timeout,
                        self.config.neuron.miner_sock_connect_timeout,
                    )
                )

            try:
                responses = []
                for future in asyncio.as_completed(
                    challenge_tasks, timeout=self.config.neuron.miner_total_timeout,
                ):
                    try:
                        response = await future
                        responses.append(response)
                    except Exception as e:
                        bt.logging.warning(f"Miner query error: {str(e)}")
                        bt.logging.error(traceback.format_exc())

                query_time = time.time() - query_start
                bt.logging.info(
                    f"Received {len(responses)} total responses for {modality} request in {query_time}s."
                )

            except asyncio.TimeoutError:
                bt.logging.warning(
                    f"Timed out waiting for miner responses after {self.config.neuron.miner_total_timeout}s"
                )
                return []

        return responses

    @on_block_interval('miner_health_sync_interval')
    async def sync_miner_health_on_interval(self, block):
        self.tracker.load_state(self.tracker_state_path)

    def process_query_results(
        self, results: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Aggregate miner responses into a final result.

        Args:
            results: List of miner responses

        Returns:
            Tuple of (aggregated predictions, responding miner UIDs)
        """
        valid_responses = [r for r in results if not r["error"] and r["prediction"] is not None]

        if not valid_responses:
            bt.logging.warning("No valid responses received from miners")
            return [], []

        predictions = np.array([r["prediction"] for r in valid_responses])
        uids = [r["uid"] for r in valid_responses]

        predictions = [p[1] + p[2] for p in predictions]
        return predictions, uids

    def get_rich_data(self, uids: List[int]) -> Dict[str, List]:
        """Get additional miner metadata."""
        return {
            "uids": [int(uid) for uid in uids],
            "ranks": [float(self.metagraph.R[uid]) for uid in uids],
            "incentives": [float(self.metagraph.I[uid]) for uid in uids],
            "emissions": [float(self.metagraph.E[uid]) for uid in uids],
            "hotkeys": [str(self.metagraph.hotkeys[uid]) for uid in uids],
            "coldkeys": [str(self.metagraph.coldkeys[uid]) for uid in uids],
        }

    async def run(self):
        await self.start()

        while not self.exit_context.isExiting:
            # Make sure our substrate thread is alive
            if not self.substrate_thread.is_alive():
                bt.logging.info("Restarting substrate interface due to killed node")
                self.substrate = SubstrateInterface(
                    ss58_format=SS58_FORMAT,
                    use_remote_preset=True,
                    url=self.config.subtensor.chain_endpoint,
                    type_registry=TYPE_REGISTRY,
                )
                self.substrate_thread = run_block_callback_thread(
                    self.substrate, self.run_callbacks
                )

            await asyncio.sleep(1)

        await self.shutdown()

    async def start(self):
        """Start the FastAPI threaded server and initialize connection pooling."""
        bt.logging.info(f"Starting proxy server on {self.host}:{self.port}")

        if self.fast_api:
            self.fast_api.start()
        else:
            bt.logging.error("FastAPI server not initialized")

    async def shutdown(self):
        """Shutdown the server and clean up resources."""
        bt.logging.info("Shutting down proxy server")

        if self.fast_api:
            self.fast_api.stop()
            self.fast_api = None


if __name__ == "__main__":
    try:
        proxy = ValidatorProxy()
        asyncio.run(proxy.run())
    except KeyboardInterrupt:
        bt.logging.info("Proxy interrupted by KeyboardInterrupt, shutting down")
    except Exception as e:
        bt.logging.error(f"Unhandled exception: {e}")
        bt.logging.error(traceback.format_exc())
