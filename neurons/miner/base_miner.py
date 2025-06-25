import time
import traceback
from abc import ABC, abstractmethod

import bittensor as bt
import numpy as np
import requests
import uvicorn
from bittensor.core.axon import FastAPIThreadedServer
from bittensor.core.extrinsics.serving import serve_extrinsic
from bittensor.core.settings import SS58_FORMAT, TYPE_REGISTRY
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from substrateinterface import SubstrateInterface

from bitmind.epistula import verify_signature, EPISTULA_VERSION
from bitmind.metagraph import run_block_callback_thread
from bitmind.types import NeuronType
from bitmind.utils import print_info
from neurons.base import BaseNeuron


def extract_testnet_metadata(headers):
    headers = dict(headers)
    testnet_metadata = {}
    for key, value in headers.items():
        if key.lower().startswith("x-testnet-"):
            metadata_key = key[len("x-testnet-") :].lower()
            testnet_metadata[metadata_key] = value
    return testnet_metadata


class BaseMiner(BaseNeuron, ABC):
    """Base class for all miner types with common functionality."""
    
    neuron_type = NeuronType.MINER
    fast_api: FastAPIThreadedServer
    initialization_complete: bool = False

    def __init__(self, config=None):
        super().__init__(config)
        bt.logging.set_info()
        
        # Typesafety
        assert self.config.netuid
        assert self.config.logging

        # Initialize the model/processor - to be implemented by subclasses
        self.initialize_models()

        # Register log callback
        self.block_callbacks.append(self.log_on_block)

        bt.logging.info(
            "\N{GRINNING FACE WITH SMILING EYES}", "Successfully Initialized!"
        )
        self.initialization_complete = True

    @abstractmethod
    def initialize_models(self):
        """Initialize the models/processors for this miner type."""
        pass

    @abstractmethod
    def setup_routes(self, router: APIRouter):
        """Setup the specific routes for this miner type."""
        pass

    @abstractmethod
    def get_miner_type(self):
        pass

    def get_miner_info(self):
        return {
            "miner_type": self.get_miner_type(),
            "uid": self.uid,
            "hotkey": self.wallet.hotkey.ss58_address
        }

    def shutdown(self):
        if self.fast_api:
            self.fast_api.stop()

    async def log_on_block(self, block):
        print_info(
            self.metagraph,
            self.wallet.hotkey.ss58_address,
            block,
        )

    async def determine_epistula_version_and_verify(self, request: Request):
        version = request.headers.get("Epistula-Version")
        if version == EPISTULA_VERSION:
            await self.verify_request(request)
            return
        raise HTTPException(status_code=400, detail="Unknown Epistula version")

    async def verify_request(self, request: Request):
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
        # This will auto-update if the axon port or external ip have changed.
        external_ip = self.config.axon.external_ip or self.config.axon.ip
        if not external_ip or external_ip == "[::]":
            try:
                external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
            except Exception:
                bt.logging.error("Failed to get external IP")

        bt.logging.info(f"Serving miner endpoint {external_ip}:{self.config.axon.port}")
        bt.logging.info(
            f"Network: {self.config.subtensor.chain_endpoint} | Netuid: {self.config.netuid}"
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

        app = FastAPI()
        router = APIRouter()
        
        router.add_api_route("/", self.ping, methods=["GET"])
        router.add_api_route(
            "/miner_info",
            self.get_miner_info, 
            dependencies=[Depends(self.determine_epistula_version_and_verify)], 
            methods=["GET"]
        )
        self.setup_routes(router)
        
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

        try:
            while not self.exit_context.isExiting:
                time.sleep(1)

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

    def ping(self):
        return 200
