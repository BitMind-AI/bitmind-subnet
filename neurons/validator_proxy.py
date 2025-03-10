from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import APIKeyHeader
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from typing import Optional, Dict, List, Union, Any
from io import BytesIO
import numpy as np
import uvicorn
import base64
import tempfile
import asyncio
import aiohttp
import cv2
import os
import httpx
import time
import socket
import logging
import json
from dotenv import load_dotenv
from substrateinterface.keypair import Keypair

# Import directly from bitmind utils
from bitmind.validator.config import TARGET_IMAGE_SIZE
from bitmind.utils.image_transforms import get_base_transforms
from bitmind.protocol import prepare_synapse
from bitmind.utils.uids import get_random_uids
from bitmind.validator.proxy import ProxyCounter

# Import our minimal dendrite
from minimal_dendrite import MinimalDendrite

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("validator_proxy")

# Constants
AUTH_HEADER = APIKeyHeader(name="Authorization")
DEFAULT_TIMEOUT = 12  # seconds, shorter timeout to avoid long waits
DEFAULT_SAMPLE_SIZE = 50

class MediaProcessor:
    """Handles processing of images and videos"""
    def __init__(self, target_size: tuple):
        logger.info(f"Initializing MediaProcessor with target_size: {target_size}")
        if target_size is None:
            logger.warning("Target size is None, using fallback (224, 224)")
            target_size = (224, 224)
        self.transforms = get_base_transforms(target_size)

    def process_image(self, b64_image: str) -> Any:
        """Process base64 encoded image"""
        image_bytes = base64.b64decode(b64_image)
        image = Image.open(BytesIO(image_bytes))
        return self.transforms(image)

    def process_video(self, video_data: bytes) -> List[Any]:
        """Process raw video bytes into transformed frames"""
        logger.debug(f"Starting video processing with {len(video_data)} bytes")

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as temp_file:
            logger.debug(f"Created temp file: {temp_file.name}")
            temp_file.write(video_data)
            temp_file.flush()

            cap = cv2.VideoCapture(temp_file.name)
            if not cap.isOpened():
                logger.error("Failed to open video stream")
                raise ValueError("Failed to open video stream")
            try:
                frames = []
                frame_count = 0
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    frame_count += 1
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(rgb_frame)
                    frames.append(pil_frame)

                logger.debug(f"Extracted {frame_count} frames")

                if not frames:
                    logger.error("No frames extracted from video")
                    raise ValueError("No frames extracted from video")
    
                transformed = self.transforms(frames)
                logger.debug(f"Transformed frames shape: {type(transformed)}")
                return transformed

            except Exception as e:
                logger.error(f"Error in video processing: {str(e)}")
                raise
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
        
        # Track responding miners for future optimization
        responding_uids = []
        
        try:
            # Query multiple miners in parallel
            predictions = await self.dendrite(
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse=prepare_synapse(data, modality=modality),
                deserialize=True,
                timeout=timeout
            )
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Error getting predictions: {str(e)}"
            )

        # Filter valid predictions
        valid_indices = []
        valid_predictions = []
        
        for i, pred in enumerate(predictions):
            try:
                # Skip None responses
                if pred is None:
                    continue
                    
                # Convert tensor to list if needed
                if hasattr(pred, 'tolist'):
                    pred_list = pred.tolist()
                else:
                    pred_list = pred
                    
                # Check if the prediction contains an error indicator
                if not isinstance(pred_list, list) or -1 in pred_list:
                    continue
                    
                # Store valid prediction
                valid_indices.append(i)
                valid_predictions.append(pred_list)
                responding_uids.append(miner_uids[i])
                
            except Exception as e:
                logger.debug(f"Error processing prediction {i}: {e}")
                continue
        
        print(responding_uids)
        
        # Update the validator's responding miners list if we have access
        if hasattr(self.validator, 'last_responding_miner_uids'):
            self.validator.last_responding_miner_uids = responding_uids
            
        # Check if we have any valid predictions
        if not valid_indices:
            logger.warning("No valid predictions received from any miners")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No valid predictions received"
            )

        # Get UIDs for valid predictions
        valid_uids = [miner_uids[i] for i in valid_indices]

        # Process predictions for the expected format
        try:
            # For bitmind predictions, we sum indices 1 and 2
            results = [p[1] + p[2] for p in valid_predictions]
            return results, valid_uids
            
        except (IndexError, TypeError) as e:
            logger.error(f"Error processing predictions: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Error processing predictions: {str(e)}"
            )

    def _get_miner_uids(self):
        """Get list of miner UIDs to query"""
        # First try to use recently responding miners
        uids = getattr(self.validator, 'last_responding_miner_uids', None)
        if not uids:
            logger.warning("No recent miner UIDs found, sampling random UIDs")
            uids = get_random_uids(self.validator, k=DEFAULT_SAMPLE_SIZE)
        return uids

    def get_rich_data(self, uids):
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
        
        # Safely handle target image size
        try:
            # Try getting directly from imported constant
            target_size = TARGET_IMAGE_SIZE
            logger.info(f"Using TARGET_IMAGE_SIZE: {target_size}")
        except (NameError, AttributeError):
            # Fallback to config if available
            target_size = getattr(validator.config.neuron, 'target_image_size', (224, 224))
            logger.info(f"Using config target_image_size: {target_size}")
            
        # Initialize media processor with safe target size
        self.media_processor = MediaProcessor(target_size)
        
        # Get keypair from environment or create from validator's wallet
        keypair = self._get_keypair()
        
        # Initialize our minimal Dendrite implementation
        self.dendrite = MinimalDendrite(keypair=keypair)
        
        self.prediction_service = PredictionService(validator, self.dendrite)
        
        # Initialize the metrics counter
        metrics_path = os.path.join(validator.config.neuron.full_path, "proxy_counter.json")
        self.metrics = ProxyCounter(metrics_path)
        
        # Setup FastAPI app
        self.app = FastAPI(title="Validator Proxy", version="1.0.0")
        self._configure_routes()

        # Start the server if port is configured
        if hasattr(validator.config.proxy, 'port') and validator.config.proxy.port:
            self.auth_verifier = self._setup_auth()
            self.start()

    def _get_keypair(self):
        """Get a keypair from environment or validator wallet"""
        # Try to get hotkey from environment
        hotkey = os.getenv('HOTKEY')
        if hotkey:
            try:
                # Try to initialize keypair from mnemonic/seed phrase
                keypair = Keypair.create_from_uri(hotkey)
                logger.info(f"Using keypair from environment with address: {keypair.ss58_address}")
                return keypair
            except Exception as e1:
                try:
                    # Try to interpret it as a SS58 address
                    keypair = Keypair(ss58_address=hotkey)
                    logger.info(f"Using keypair from address: {keypair.ss58_address}")
                    return keypair
                except Exception as e2:
                    logger.error(f"Failed to create keypair from environment: {e1}, {e2}")
        
        # Fall back to validator's wallet if available
        try:
            if hasattr(self.validator, 'wallet') and hasattr(self.validator.wallet, 'hotkey'):
                logger.info("Using validator wallet keypair")
                return self.validator.wallet.hotkey
        except Exception as e:
            logger.error(f"Failed to get validator wallet keypair: {e}")
        
        logger.warning("No keypair available, requests may fail authentication")
        return None

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

    def _setup_auth(self):
        """Set up authentication verifier using synchronous HTTP client"""
        try:
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

            def verify(key_bytes: bytes):
                try:
                    key = Ed25519PublicKey.from_public_bytes(key_bytes)
                    key.verify(signature, message.encode())
                    return True
                except InvalidSignature:
                    return False

            return verify
        except Exception as e:
            logger.error(f"Failed to set up auth verifier: {e}")
            # Return a function that always passes for development
            return lambda key_bytes: True

    async def verify_auth(self, auth: str = Depends(AUTH_HEADER)):
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

    async def handle_image_request(self, request: Request):
        """Handle image processing requests"""
        start_time = time.time()
        
        try:
            # Parse request body
            payload = await request.json()
            
            if 'image' not in payload:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing 'image' field in request body"
                )

            # Process the image
            try:
                image = self.media_processor.process_image(payload['image'])
                logger.info("Image processed successfully")
            except Exception as e:
                logger.error(f"Failed to process image: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to process image: {str(e)}"
                )

            # Get predictions from miners
            predictions, uids = await self.prediction_service.get_predictions(
                image, 
                modality='image',
                timeout=DEFAULT_TIMEOUT
            )
            process_time = time.time() - start_time
            logger.info(f"Got {len(predictions)} predictions in {process_time:.2f}s")

            # Prepare response
            response = {
                'preds': predictions,
                'fqdn': socket.getfqdn()
            }

            # Add rich data if requested
            if payload.get('rich', '').lower() == 'true':
                response.update(self.prediction_service.get_rich_data(uids))

            # Update metrics
            self.metrics.update(is_success=True)
            
            return response
            
        except HTTPException:
            # Let FastAPI handle HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error in handle_image_request: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error: {str(e)}"
            )

    async def handle_video_request(self, request: Request):
        """Handle video processing requests"""
        start_time = time.time()
        
        try:
            # Parse multipart form
            form = await request.form()
            if "video" not in form:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing video file in form data"
                )

            # Get video file
            video_file = form["video"]
            logger.debug(f"Received video file of type: {type(video_file)}")

            # Read video data
            video_data = await video_file.read()
            logger.debug(f"Read video data of size: {len(video_data)} bytes")

            if not video_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Empty video file"
                )

            # Process video
            try:
                video = self.media_processor.process_video(video_data)
                logger.debug(f"Processed video into frames")
            except Exception as e:
                logger.error(f"Video processing error: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to process video: {str(e)}"
                )
  
            logger.info(f"Finished processing video in {time.time() - start_time:.3f}s")
            
            # Get predictions
            predictions, uids = await self.prediction_service.get_predictions(
                video, 
                modality='video',
                timeout=DEFAULT_TIMEOUT
            )
            total_time = time.time() - start_time
            logger.info(f"Got {len(predictions)} predictions in {total_time:.3f}s")

            # Prepare response
            response = {
                'preds': predictions,
                'fqdn': socket.getfqdn()
            }

            # Add rich data if requested
            rich_param = form.get('rich', '').lower()
            if rich_param == 'true':
                response.update(self.prediction_service.get_rich_data(uids))

            # Update metrics
            self.metrics.update(is_success=True)
            
            return response

        except HTTPException:
            # Let FastAPI handle HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error in handle_video_request: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error: {str(e)}"
            )

    async def healthcheck(self, request: Request):
        """Health check endpoint"""
        return {'status': 'healthy'}

    def start(self):
        """Start the FastAPI server"""
        try:
            logger.info(f"Starting validator proxy on port {self.validator.config.proxy.port}")
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.executor.submit(
                uvicorn.run, 
                self.app, 
                host="0.0.0.0", 
                port=self.validator.config.proxy.port,
                log_level="info"
            )
        except Exception as e:
            logger.error(f"Failed to start validator proxy: {e}")