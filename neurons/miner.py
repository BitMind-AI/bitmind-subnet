import io
import os
import time
import traceback

import av
import json
import bittensor as bt
import numpy as np
import requests
import tempfile
import torch
import uvicorn
from bittensor.core.axon import FastAPIThreadedServer
from bittensor.core.extrinsics.serving import serve_extrinsic
from bittensor.core.settings import SS58_FORMAT, TYPE_REGISTRY
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from PIL import Image
from torchvision import models
from substrateinterface import SubstrateInterface

from bitmind.epistula import verify_signature, EPISTULA_VERSION
from bitmind.metagraph import (
    run_block_callback_thread,
)
from bitmind.types import NeuronType
from bitmind.utils import print_info
from neurons.base import BaseNeuron


class Detector:
    def __init__(self, config):
        self.config = config
        self.image_detector = None
        self.video_detector = None
        self.device = (
            self.config.device
            if hasattr(self.config, "device")
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.load_model()

    def load_model(self, modality=None):
        """Load the appropriate detection model based on modality.

        MINER TODO:
            This class has placeholder models to demonstrate the required outputs
            for validator requests. They have not been trained and will perform
            poorly. Your task is to train performant video and image detection
            models and load them here. Happy mining!

        Args:
            modality (str): Type of detection model to load ('image' or 'video')
        """
        bt.logging.info(f"Loading {modality} detection model...")
        if modality in ("image", None):
            ### REPLACE WITH YOUR OWN MODEL
            self.image_detector = models.resnet50(pretrained=True)
            num_ftrs = self.image_detector.fc.in_features
            self.image_detector.fc = torch.nn.Linear(num_ftrs, 3)
            self.image_detector = self.image_detector.to(self.device)
            self.image_detector.eval()

        if modality in ("video", None):
            ### REPLACE WITH YOUR OWN MODEL
            self.video_detector = models.video.r3d_18(pretrained=True)
            num_ftrs = self.video_detector.fc.in_features
            self.video_detector.fc = torch.nn.Linear(num_ftrs, 3)
            self.video_detector = self.video_detector.to(self.device)
            self.video_detector.eval()

        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def preprocess(self, media_tensor, modality):
        bt.logging.debug(
            json.dumps(
                {
                    "modality": "video",
                    "shape": tuple(media_tensor.shape),
                    "dtype": str(media_tensor.dtype),
                    "min": torch.min(media_tensor).item(),
                    "max": torch.max(media_tensor).item(),
                },
                indent=2,
            )
        )

        if modality == "image":
            media_tensor = media_tensor.unsqueeze(0).float().to(self.device)
        elif modality == "video":
            media_tensor = media_tensor.unsqueeze(0).float().to(self.device)
        return media_tensor

    def detect(self, media_tensor, modality):
        """Perform inference with either self.video_detector or self.image_detector

        MINER TODO: Update detection logic as necessary for your own model

        Args:
            tensor (torch.tensor): Input media tensor
            modality (str): Type of detection to perform ('image' or 'video')

        Returns:
            torch.Tensor: Probability vector containing 3 class probabilities
                [p_real, p_synthetic, p_semisynthetic]
        """
        media_tensor = self.preprocess(media_tensor, modality)

        if modality == "image":
            if self.image_detector is None:
                self.load_model("image")

            bt.logging.debug(
                f"Running image detection on array shape {media_tensor.shape}"
            )

            # MINER TODO update detection logic as necessary
            with torch.no_grad():
                outputs = self.image_detector(media_tensor)
                probs = torch.softmax(outputs, dim=1)[0]

        elif modality == "video":
            if self.video_detector is None:
                self.load_model("video")

            bt.logging.debug(
                f"Running video detection on array shape {media_tensor.shape}"
            )

            # MINER TODO update detection logic as necessary
            with torch.no_grad():
                outputs = self.video_detector(media_tensor)
                probs = torch.softmax(outputs, dim=1)[0]

        else:
            raise ValueError(f"Unsupported modality: {modality}")

        bt.logging.success(f"Prediction: {probs}")
        return probs


class Miner(BaseNeuron):
    neuron_type = NeuronType.MINER
    fast_api: FastAPIThreadedServer
    initialization_complete: bool = False

    def __init__(self, config=None):
        super().__init__(config)
        bt.logging.set_info()
        ## Typesafety
        assert self.config.netuid
        assert self.config.logging

        self.detector = Detector(self.config)

        # Register log callback
        self.block_callbacks.append(self.log_on_block)

        ## BITTENSOR INITIALIZATION
        bt.logging.info(
            "\N{GRINNING FACE WITH SMILING EYES}", "Successfully Initialized!"
        )
        self.initialization_complete = True

    def shutdown(self):
        if self.fast_api:
            self.fast_api.stop()

    async def log_on_block(self, block):
        print_info(
            self.metagraph,
            self.wallet.hotkey.ss58_address,
            block,
        )

    async def detect_image(self, request: Request):
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

    async def determine_epistula_version_and_verify(self, request: Request):
        version = request.headers.get("Epistula-Version")
        if version == EPISTULA_VERSION:
            await self.verify_request(request)
            return
        raise HTTPException(status_code=400, detail="Unknown Epistula version")

    async def verify_request(
        self,
        request: Request,
    ):
        bt.logging.debug("Verifying request")
        # We do this as early as possible so that now has a lesser chance
        # of causing a stale request
        now = round(time.time() * 1000)

        # We need to check the signature of the body as bytes
        # But use some specific fields from the body
        signed_by = request.headers.get("Epistula-Signed-By")
        signed_for = request.headers.get("Epistula-Signed-For")
        if signed_for != self.wallet.hotkey.ss58_address:
            raise HTTPException(
                status_code=400, detail="Bad Request, message is not intended for self"
            )
        if signed_by not in self.metagraph.hotkeys:
            raise HTTPException(status_code=401, detail="Signer not in metagraph")

        uid = self.metagraph.hotkeys.index(signed_by)
        stake = self.metagraph.S[uid].item()
        if not self.config.no_force_validator_permit and stake < 10000:
            bt.logging.warning(
                f"Blacklisting request from {signed_by} [uid={uid}], not enough stake -- {stake}"
            )
            raise HTTPException(status_code=401, detail="Stake below minimum: {stake}")

        # If anything is returned here, we can throw
        body = await request.body()
        err = verify_signature(
            request.headers.get("Epistula-Request-Signature"),
            body,
            request.headers.get("Epistula-Timestamp"),
            request.headers.get("Epistula-Uuid"),
            signed_for,
            signed_by,
            now,
        )
        if err:
            bt.logging.error(err)
            raise HTTPException(status_code=400, detail=err)

    def run(self):
        assert self.config.netuid
        assert self.config.subtensor
        assert self.config.axon

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        external_ip = self.config.axon.external_ip or self.config.axon.ip
        if not external_ip or external_ip == "[::]":
            try:
                external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
            except Exception:
                bt.logging.error("Failed to get external IP")

        bt.logging.info(f"Serving miner endpoint {external_ip}:{self.config.axon.port}")
        bt.logging.info(
            f"Netowrk: {self.config.subtensor.chain_endpoint} | Netuid: {self.config.netuid}"
        )

        serve_success = serve_extrinsic(
            subtensor=self.subtensor,
            wallet=self.wallet,
            ip=external_ip,
            port=self.config.axon.port,
            protocol=4,
            netuid=self.config.netuid,
            wait_for_finalization=True,
        )
        if not serve_success:
            bt.logging.error("Failed to serve endpoint")
            return

        # Start  starts the miner's endpoint, making it active on the network.
        # change the config in the axon
        app = FastAPI()
        router = APIRouter()
        router.add_api_route("/", ping, methods=["GET"])

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
        app.include_router(router)
        fast_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.config.axon.port,
            log_level="info",
            loop="asyncio",
        )
        self.fast_api = FastAPIThreadedServer(config=fast_config)
        self.fast_api.start()

        bt.logging.info(f"Miner {self.uid} starting at block: {self.subtensor.block}")

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.exit_context.isExiting:
                time.sleep(1)

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
        except Exception as e:
            bt.logging.error(str(e))
            bt.logging.error(traceback.format_exc())
        finally:
            self.shutdown()


def ping():
    return 200


if __name__ == "__main__":
    try:
        miner = Miner()
        miner.run()
    except Exception as e:
        bt.logging.error(str(e))
        bt.logging.error(traceback.format_exc())
    exit()
