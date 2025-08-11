from typing import Dict, List, Optional, Any
import time

import bittensor as bt
import aiohttp
import asyncio

from gas.protocol.validator_requests import get_miner_type
from gas.types import MinerType


class MinerTypeTracker:
    """Tracks miner types and manages periodic updates"""

    def __init__(self, config, wallet, metagraph, subtensor):
        self.config = config
        self.wallet = wallet
        self.metagraph = metagraph
        self.subtensor = subtensor

        # Storage for miner types
        self.miner_types: Dict[int, MinerType] = {}
        self.last_update: Dict[int, float] = {}
        self.recheck_miner_type = set([])

    async def initialize_miner_types(self):
        """Initialize miner types for all registered miners"""
        bt.logging.debug("Initializing miner types for all registered miners...")
        all_miners = list(range(self.metagraph.n.item()))
        await self.update_miner_types(all_miners)
        bt.logging.debug(f"Initialized miner types for {len(self.miner_types)} miners")

    async def update_miner_types(self, miner_uids: Optional[List[int]] = None):
        """Update miner types for specified miners"""
        # first filter to only unknown miners or those needing recheck
        if miner_uids is None:
            miner_uids = list(range(self.metagraph.n))

        unk_miners = [
            uid
            for uid in miner_uids
            if uid not in self.miner_types or uid in self.recheck_miner_type
        ]

        if not unk_miners:
            bt.logging.debug("All miner types known.")
            return

        bt.logging.trace(f"Updating miner types for {unk_miners}")
        async with aiohttp.ClientSession() as session:
            responses = await asyncio.gather(
                *[
                    get_miner_type(
                        uid,
                        self.metagraph.axons[uid],
                        session,
                        self.wallet.hotkey,
                        self.config.neuron.miner_total_timeout,
                    )
                    for uid in unk_miners
                ],
                return_exceptions=True,
            )

        current_time = time.time()
        for uid, response in zip(unk_miners, responses):
            # Remove from recheck queue
            if uid in self.recheck_miner_type:
                self.recheck_miner_type.remove(uid)

            # We assume a miner is a discriminator (which does not run any endpoint/hardware)
            # until we successfully get a response from a generator get_miner_info endpoint
            miner_type = MinerType.DISCRIMINATOR

            if isinstance(response, Exception):
                continue

            miner_type_str = response.get("miner_type", "") or ""
            if miner_type_str.lower() == "generator":
                miner_type = MinerType.GENERATOR

            if miner_type:
                old_type = self.miner_types.get(uid)
                self.miner_types[uid] = miner_type
                self.last_update[uid] = current_time

                if old_type != miner_type:
                    bt.logging.success(f"UID {uid}: {old_type} -> {miner_type}")
                else:
                    bt.logging.debug(f"UID {uid}: {miner_type}")

    def get_miner_type(self, uid: int) -> Optional[MinerType]:
        return self.miner_types.get(uid)

    def get_miners_by_type(self, miner_type: MinerType) -> List[int]:
        return [uid for uid, mt in self.miner_types.items() if mt == miner_type]
