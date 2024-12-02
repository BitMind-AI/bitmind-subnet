from fastapi import FastAPI, HTTPException, Depends, Request
from concurrent.futures import ThreadPoolExecutor
from starlette.concurrency import run_in_threadpool
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
from PIL import Image
from io import BytesIO
import bittensor as bt
import pandas as pd
import numpy as np
import uvicorn
import base64
import json
import os
import random
import asyncio
import traceback
import httpx
import threading
import socket
import base64

from bitmind.image_transforms import base_transforms
from bitmind.protocol import ImageSynapse, prepare_image_synapse
from bitmind.utils.uids import get_random_uids
from bitmind.validator.proxy import ProxyCounter
import bitmind


def preprocess_image(b64_image):
    image_bytes = base64.b64decode(b64_image)
    image_buffer = BytesIO(image_bytes)
    pil_image = Image.open(image_buffer)
    return base_transforms(pil_image)


class ValidatorProxy:
    def __init__(
        self,
        validator,
    ):
        self.validator = validator
        self.get_credentials()
        self.miner_request_counter = {}
        self.dendrite = bt.dendrite(wallet=validator.wallet)
        self.app = FastAPI()
        self.app.add_api_route(
            "/validator_proxy",
            self.forward,
            methods=["POST"],
            dependencies=[Depends(self.get_self)],
        )
        self.app.add_api_route(
            "/healthcheck",
            self.healthcheck,
            methods=["GET"],
            dependencies=[Depends(self.get_self)],
        )

        self.loop = asyncio.get_event_loop()
        self.proxy_counter = ProxyCounter(
            os.path.join(self.validator.config.neuron.full_path, "proxy_counter.json")
        )
        if self.validator.config.proxy.port:
            self.start_server()

    def get_credentials(self):
        with httpx.Client(timeout=httpx.Timeout(30)) as client:
            response = client.post(
                f"{self.validator.config.proxy.proxy_client_url}/get-credentials",
                json={
                    "postfix": (
                        f":{self.validator.config.proxy.port}/validator_proxy"
                        if self.validator.config.proxy.port
                        else ""
                    ),
                    "uid": self.validator.uid,
                },
            )
        response.raise_for_status()
        response = response.json()
        message = response["message"]
        signature = response["signature"]
        signature = base64.b64decode(signature)

        def verify_credentials(public_key_bytes):
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            try:
                public_key.verify(signature, message.encode("utf-8"))
            except InvalidSignature:
                raise Exception("Invalid signature")

        self.verify_credentials = verify_credentials

    def start_server(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(
            uvicorn.run, self.app, host="0.0.0.0", port=self.validator.config.proxy.port
        )

    def authenticate_token(self, public_key_bytes):
        public_key_bytes = base64.b64decode(public_key_bytes)
        try:
            self.verify_credentials(public_key_bytes)
            bt.logging.info("Successfully authenticated token")
            return public_key_bytes
        except Exception as e:
            bt.logging.error(f"Exception occured in authenticating token: {e}")
            bt.logging.error(traceback.print_exc())
            raise HTTPException(
                status_code=401, detail="Error getting authentication token"
            )

    async def healthcheck(self, request: Request):
        authorization: str = request.headers.get("authorization")

        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header missing")

        self.authenticate_token(authorization)
        return {'status': 'healthy'}

    async def forward(self, request: Request):
        authorization: str = request.headers.get("authorization")
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header missing")
        self.authenticate_token(authorization)

        bt.logging.info("Received an organic request!")
        payload = await request.json()

        if "seed" not in payload:
            payload["seed"] = random.randint(0, int(1e9))

        metagraph = self.validator.metagraph
        miner_uids = self.validator.last_responding_miner_uids
        if len(miner_uids) == 0:
            bt.logging.warning("[ORGANIC] No recent miner uids found, sampling random uids")
            miner_uids = get_random_uids(self.validator, k=self.validator.config.neuron.sample_size)

        image = preprocess_image(payload['image'])

        bt.logging.info(f"[ORGANIC] Querying {len(miner_uids)} miners...")
        predictions = await self.dendrite(
            axons=[metagraph.axons[uid] for uid in miner_uids],
    	    synapse=prepare_image_synapse(image=image),
            deserialize=True,
            timeout=9
        )

        bt.logging.info(f"[ORGANIC] {predictions}")
        valid_pred_idx = np.array([i for i, v in enumerate(predictions) if v != -1.])
        if len(valid_pred_idx) > 0:
            valid_preds = np.array(predictions)[valid_pred_idx]
            valid_pred_uids = np.array(miner_uids)[valid_pred_idx]
            if len(valid_preds) > 0:
                self.proxy_counter.update(is_success=True)
                self.proxy_counter.save()

                data = {
                    'preds': [float(p) for p in list(valid_preds)],
                    'fqdn': socket.getfqdn()
                }

                rich_response: bool = payload.get("rich", "false").lower() == "true"
                if rich_response:
                    data['uids'] = [int(uid) for uid in valid_pred_uids],
                    data['ranks'] = [float(metagraph.R[uid]) for uid in valid_pred_uids],
                    data['incentives'] = [float(metagraph.I[uid]) for uid in valid_pred_uids]
                    data['emissions'] = [float(metagraph.E[uid]) for uid in valid_pred_uids]
                    data['hotkeys'] = [str(metagraph.hotkeys[uid]) for uid in valid_pred_uids]
                    data['coldkeys'] = [str(metagraph.coldkeys[uid]) for uid in valid_pred_uids]

                return data

        self.proxy_counter.update(is_success=False)
        self.proxy_counter.save()
        return HTTPException(status_code=500, detail="No valid response received")

    async def get_self(self):
        return self
