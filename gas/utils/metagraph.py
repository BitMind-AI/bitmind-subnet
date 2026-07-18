import time
import asyncio
from typing import Callable, List, Tuple
import numpy as np
import bittensor as bt
from bittensor.utils.weight_utils import process_weights_for_netuid
from async_substrate_interface import AsyncSubstrateInterface

from gas.utils import fail_with_none


def get_miner_uids(
    metagraph: "bt.Metagraph", self_uid: int, vpermit_tao_limit: int
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
        wallet: "bt.Wallet",
        metagraph: "bt.Metagraph",
        subtensor: "bt.Subtensor",
        weights: Tuple[List[int], List[float]],
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

        #bt.logging.info("Setting Weights: " + str(processed_weights))
        #bt.logging.info("Weight Uids: " + str(processed_weight_uids))
        for _ in range(3):
            response = subtensor.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=processed_weight_uids,  # type: ignore
                weights=processed_weights,
                wait_for_finalization=False,
                wait_for_inclusion=False,
                version_key=version,
                max_attempts=1,
            )
            success, message = response
            if success is True:
                bt.logging.success("set_weights on chain successfully!")
                break

            # set_weights returns success=False with no message when the weights
            # rate limit hasn't elapsed; retrying can't clear it, so stop.
            reason = message or getattr(response, "error", None)
            if reason is None and _weights_rate_limited(subtensor, wallet, netuid):
                bt.logging.info("Skipping set_weights: rate limit not elapsed (too early)")
                break
            bt.logging.error(f"set_weights failed: {reason or 'unknown error'}")
            time.sleep(15)

    return set_weights


def _weights_rate_limited(
    subtensor: "bt.Subtensor", wallet: "bt.Wallet", netuid: int
) -> bool:
    """Return True if the weights rate limit has not yet elapsed for this hotkey."""
    try:
        uid = subtensor.get_uid_for_hotkey_on_subnet(
            wallet.hotkey.ss58_address, netuid
        )
        if uid is None:
            return False
        blocks_since = subtensor.blocks_since_last_update(netuid, uid)
        rate_limit = subtensor.weights_rate_limit(netuid)
        if blocks_since is None or rate_limit is None:
            return False
        return blocks_since <= rate_limit
    except Exception as e:
        bt.logging.debug(f"Could not check weights rate limit: {e}")
        return False


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

