import time
import asyncio
from typing import Callable, List, Tuple
import numpy as np
import bittensor as bt
from bittensor.utils.weight_utils import process_weights_for_netuid
from async_substrate_interface import AsyncSubstrateInterface

from gas.utils import fail_with_none


def get_miner_uids(
    metagraph: "bt.metagraph", self_uid: int, vpermit_tao_limit: int
) -> List[int]:
    available_uids = []
    for uid in range(int(metagraph.n.item())):
        if uid == self_uid:
            continue

        # Filter validator permit > 1024 stake.
        if metagraph.validator_permit[uid]:
            if metagraph.S[uid] > vpermit_tao_limit:
                continue

        available_uids.append(uid)

    return available_uids


def create_set_weights(version: int, netuid: int):
    @fail_with_none("Failed setting weights")
    def set_weights(
        wallet: "bt.wallet",
        metagraph: "bt.metagraph",
        subtensor: "bt.subtensor",
        weights: Tuple[List[int], List[float]],
        mechid: int = 0,
    ):
        uids, raw_weights = weights
        if not len(uids):
            bt.logging.info("No UIDS to score")
            return

        # Set the weights on chain via our subtensor connection.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=np.asarray(uids),
            weights=np.asarray(raw_weights),
            netuid=netuid,
            subtensor=subtensor,
            metagraph=metagraph,
        )

        def _success_and_message(response):
            if isinstance(response, tuple):
                return response
            if hasattr(response, "success"):
                return bool(response.success), getattr(response, "message", "")
            return bool(response), ""

        for _ in range(3):
            kwargs = {
                "wallet": wallet,
                "netuid": netuid,
                "uids": processed_weight_uids,  # type: ignore
                "weights": processed_weights,
                "wait_for_finalization": False,
                "wait_for_inclusion": False,
                "version_key": version,
            }
            try:
                response = subtensor.set_weights(
                    **kwargs,
                    mechid=mechid,
                    max_attempts=1,
                )
            except TypeError as e:
                if "mechid" not in str(e) and "max_attempts" not in str(e):
                    raise
                if mechid != 0 and "mechid" in str(e):
                    raise RuntimeError(
                        "This Bittensor SDK does not support mechanism-specific "
                        "set_weights; upgrade before enabling mechanism 1."
                    ) from e
                response = subtensor.set_weights(
                    **kwargs,
                    max_retries=1,
                )

            result, message = _success_and_message(response)
            if result is True:
                bt.logging.success(
                    f"set_weights on chain successfully for mechanism {mechid}!"
                )
                break
            else:
                bt.logging.error(f"set_weights failed for mechanism {mechid}: {message}")
            time.sleep(15)

    return set_weights


def create_async_subscription_handler(callback: Callable):
    """Create async subscription handler - simple version."""
    async def handler(obj):
        try:
            # Extract block number just like sync version
            block_number = obj["header"]["number"]
            await callback(block_number)
        except Exception as e:
            bt.logging.error(f"Error in async substrate block callback: {e}")
    
    return handler


async def start_async_subscription(substrate, callback: Callable):
    """Start async block subscription - mirroring the sync version."""
    return await substrate.subscribe_block_headers(
        create_async_subscription_handler(callback)
    )


class SubstrateConnectionManager:
    """Async substrate connection manager with auto-reconnection."""

    def __init__(self, url: str, ss58_format: int, type_registry: dict):
        self.url = url
        self.ss58_format = ss58_format
        self.type_registry = type_registry
        self.running = False
        self.task = None

    async def start_subscription(self, callback: Callable):
        """Start subscription with auto-reconnect."""
        self.running = True

        while self.running:
            try:
                bt.logging.info(f"Connecting to async substrate: {self.url}")

                substrate = AsyncSubstrateInterface(
                    url=self.url,
                    ss58_format=self.ss58_format,
                    type_registry=self.type_registry,
                )

                async with substrate:
                    bt.logging.info("Starting async block subscription")
                    await start_async_subscription(substrate, callback)

            except Exception as e:
                bt.logging.error(f"Async substrate failed: {e}")
                if self.running:
                    bt.logging.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

    def start_subscription_task(self, callback: Callable):
        """Start as background task."""
        self.task = asyncio.create_task(self.start_subscription(callback))
        return self.task

    def stop(self):
        """Stop subscription."""
        self.running = False
        if self.task:
            self.task.cancel()
