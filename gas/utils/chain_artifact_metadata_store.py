import functools
from typing import List, Optional

import bittensor as bt

from gas.types import ArtifactChainMetadata, ChainMetadataRegistry, MinerType
from gas.utils import run_in_thread


class ChainArtifactMetadataStore:
    """Stores and retrieves DPS artifact R2 metadata from chain commitments."""

    def __init__(self, subtensor: bt.subtensor, netuid: int):
        self.subtensor = subtensor
        self.netuid = netuid

    async def store_artifact_metadata(
        self,
        wallet: bt.wallet,
        metadata: ArtifactChainMetadata,
        ttl: int = 60,
    ):
        registry = self.retrieve_registry_for_uid(self._wallet_uid(wallet), ttl=ttl)
        registry.upsert_artifact(metadata)
        data = registry.to_compressed_str()
        commit_partial = functools.partial(
            self.subtensor.commit,
            wallet,
            self.netuid,
            data,
        )
        return run_in_thread(commit_partial, ttl)

    def retrieve_registry_for_uid(
        self,
        uid: int,
        ttl: int = 60,
    ) -> ChainMetadataRegistry:
        chain_str = run_in_thread(
            functools.partial(self.subtensor.get_commitment, self.netuid, uid),
            ttl,
        )
        if not chain_str:
            return ChainMetadataRegistry()
        try:
            return ChainMetadataRegistry.from_compressed_str(chain_str)
        except Exception:
            return ChainMetadataRegistry()

    async def retrieve_artifact_metadata(
        self,
        uid: int,
        expected_kind: Optional[str] = None,
        role: Optional[MinerType] = None,
        task_id: Optional[str] = None,
        ttl: int = 60,
    ) -> Optional[ArtifactChainMetadata]:
        artifacts = await self.retrieve_artifact_metadata_list(
            uid=uid,
            expected_kind=expected_kind,
            role=role,
            task_id=task_id,
            ttl=ttl,
        )
        return artifacts[-1] if artifacts else None

    async def retrieve_artifact_metadata_list(
        self,
        uid: int,
        expected_kind: Optional[str] = None,
        role: Optional[MinerType] = None,
        task_id: Optional[str] = None,
        ttl: int = 60,
    ) -> List[ArtifactChainMetadata]:
        registry = self.retrieve_registry_for_uid(uid, ttl=ttl)
        return registry.get_artifacts(
            expected_kind=expected_kind,
            role=role,
            task_id=task_id,
        )

    def _wallet_uid(self, wallet: bt.wallet) -> int:
        return self.subtensor.get_uid_for_hotkey_on_subnet(
            wallet.hotkey.ss58_address,
            self.netuid,
        )
