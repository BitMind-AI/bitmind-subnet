from fastapi import FastAPI, HTTPException, Depends
from concurrent.futures import ThreadPoolExecutor
from starlette.concurrency import run_in_threadpool
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
import bittensor as bt
import pandas as pd
import numpy as np
import uvicorn
import base64
import os
import random
import asyncio
import traceback
import httpx
import threading

from bitmind.utils.miner_metrics import miner_metrics
from bitmind.protocol import ImageSynapse
from bitmind.utils.uids import get_random_uids
from bitmind.validator.proxy import ProxyCounter
import bitmind


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
            "/miner_performance",
            self.miner_performance,
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
            print("Exception occured in authenticating token", e, flush=True)
            print(traceback.print_exc(), flush=True)
            raise HTTPException(
                status_code=401, detail="Error getting authentication token"
            )

    async def forward(self, payload: dict, request):
        authorization: str = request.headers.get("authorization")

        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header missing")

        #self.authenticate_token(payload["authorization"])
        self.authenticate_token(authorization)

        if "recheck" in payload:
            bt.logging.info("Rechecking validators")
            self.get_credentials()
            return {"message": "done"}
        bt.logging.info("Received an organic request!")
        if "seed" not in payload:
            payload["seed"] = random.randint(0, 1e9)

        #timeout = model_config["timeout"]
        #reward_url = model_config["reward_url"]

        metagraph = self.validator.metagraph

        # TODO: preprocess image
        miner_uids = get_random_uids(self.validator, k=self.validator.config.neuron.sample_size)
        print(f"[ORGANIC] Querying {len(miner_uids)} miners...")
        predictions = await self.dendrite(
            axons=[metagraph.axons[uid] for uid in miner_uids],
    	    synapse=ImageSynapse(image=payload['image'], prediction=-1),
            deserialize=True
        )
        print(f"[ORGANIC] {predictions}")
        valid_pred_idx = np.array([i for i, v in enumerate(predictions) if v != -1.])
        if len(valid_pred_idx) > 0:
            valid_preds = np.array(predictions)[valid_pred_idx]
            valid_pred_uids = np.array(miner_uids)[valid_pred_idx]
            if len(valid_preds) > 0:
                self.proxy_counter.update(is_success=True)
                self.proxy_counter.save()
                return list(valid_preds)

        self.proxy_counter.update(is_success=False)
        self.proxy_counter.save()
        return HTTPException(status_code=500, detail="No valid response received")

    async def miner_performance(self, payload: dict, request):
        authorization: str = request.headers.get("authorization")

        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header missing")

        self.authenticate_token(authorization)
        #self.authenticate_token(payload["authorization"])

        if "recheck" in payload:
            bt.logging.info("Rechecking validators")
            self.get_credentials()
            return {"message": "done"}
        bt.logging.info("Received an organic request!")

        metagraph = self.validator.metagraph

        results_file = '../bitmind-subnet/results.csv'

        results_df = pd.read_csv(results_file)
        results_df.time = pd.to_datetime(results_df.time)
        results_df = results_df[results_df.pred != -1]

        # TODO: rolling mean for hourly performance
        #if cutoff_datetime is not None:
        #    results_df = results_df[results_df.time > pd.to_datetime(cutoff_datetime)]

        metrics_df = miner_metrics(results_df, transpose=False)
        metrics_df = metrics_df.merge(
            pd.DataFrame({
                'miner_uid': list(range(0, len(metagraph.R))),
                'trust': metagraph.T,
                'concensus': metagraph.C,
                'emission': metagraph.E,
                'incentive': metagraph.I,
                'rank': metagraph.R}),
            on='miner_uid')

        return metrics_df.to_json(orient='records')
        #return HTTPException(status_code=500, detail="No valid response received")

    async def get_self(self):
        return self
