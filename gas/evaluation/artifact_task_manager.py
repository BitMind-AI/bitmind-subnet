import asyncio
import hashlib
import json
import time

import aiohttp
import bittensor as bt

from gas.protocol.validator_requests import query_artifact_miner
from gas.types import (
    ArtifactChainMetadata,
    ArtifactR2Location,
    ArtifactTaskSpec,
    MinerType,
)
from gas.utils.chain_artifact_metadata_store import ChainArtifactMetadataStore
from gas.evaluation.artifact_verifier import ArtifactVerifier


class ArtifactTaskManager:
    """Assigns DPS artifact miners dataset shards to pull from R2."""

    def __init__(self, config, wallet, metagraph, subtensor, miner_type_tracker):
        self.config = config
        self.wallet = wallet
        self.metagraph = metagraph
        self.subtensor = subtensor
        self.miner_type_tracker = miner_type_tracker
        self.last_assignment_at = {}
        self.last_task_id_by_uid = {}
        self.latest_reward_stats = {"encoder_stats": [], "captioner_stats": []}
        self.metadata_store = ChainArtifactMetadataStore(subtensor, config.netuid)
        self.artifact_verifier = ArtifactVerifier()

    async def publish_input_metadata(self):
        if not getattr(self.config, "enable_dps_artifact_mechanism", False):
            return
        if not getattr(self.config.dps_artifact, "publish_input_to_chain", False):
            return

        r2_location = self._r2_location()
        if r2_location is None:
            bt.logging.warning(
                "DPS artifact input metadata not published: R2 bucket and path are required"
            )
            return

        metadata = ArtifactChainMetadata(
            kind="dps_input",
            role=MinerType.ENCODER,
            r2=r2_location,
            artifact_spec=self._artifact_spec(),
        )
        await self.metadata_store.store_artifact_metadata(self.wallet, metadata)
        bt.logging.info("Published DPS artifact input R2 metadata to chain")

    async def issue_artifact_tasks(self, block: int = 0):
        if not getattr(self.config, "enable_dps_artifact_mechanism", False):
            return

        r2_source = await self._r2_source()
        if not r2_source:
            bt.logging.trace("DPS artifact task assignment skipped: no R2 source configured")
            return

        await self.miner_type_tracker.update_miner_types()
        tasks = []
        for role in (MinerType.ENCODER, MinerType.CAPTIONER):
            miner_uids = self._sample_role_uids(role)
            for uid in miner_uids:
                tasks.append(self.send_artifact_task(uid, role, r2_source, block))

        if not tasks:
            bt.logging.trace("No DPS artifact miners found to assign R2 tasks.")
            return

        await asyncio.gather(*tasks)

    async def get_miner_output_metadata(self, miner_uids=None):
        if miner_uids is None:
            miner_uids = (
                self.miner_type_tracker.get_miners_by_type(MinerType.ENCODER)
                + self.miner_type_tracker.get_miners_by_type(MinerType.CAPTIONER)
            )

        metadata_by_uid = {}
        for uid in miner_uids:
            metadata = await self.metadata_store.retrieve_artifact_metadata(
                uid=uid,
                expected_kind="dps_output",
            )
            if metadata is not None:
                metadata_by_uid[uid] = metadata
        return metadata_by_uid

    async def validate_miner_outputs(self):
        await self.miner_type_tracker.update_miner_types()
        miner_uids = (
            self.miner_type_tracker.get_miners_by_type(MinerType.ENCODER)
            + self.miner_type_tracker.get_miners_by_type(MinerType.CAPTIONER)
        )
        outputs = await self.get_miner_output_metadata(miner_uids)
        reward_stats = {"encoder_stats": [], "captioner_stats": []}
        for uid, metadata in outputs.items():
            stats = self._verify_output_metadata(uid, metadata)
            if metadata.role == MinerType.ENCODER:
                reward_stats["encoder_stats"].append(stats)
            elif metadata.role == MinerType.CAPTIONER:
                reward_stats["captioner_stats"].append(stats)

        self.latest_reward_stats = reward_stats
        rewards_path = getattr(self.config, "dps_artifact_rewards_path", None)
        if rewards_path:
            try:
                with open(rewards_path, "w", encoding="utf-8") as f:
                    json.dump(reward_stats, f, separators=(",", ":"))
            except Exception as e:
                bt.logging.warning(f"Failed writing DPS artifact reward stats: {e}")

        bt.logging.info(
            "Validated DPS artifact output metadata: "
            f"{len(reward_stats['encoder_stats'])} encoder, "
            f"{len(reward_stats['captioner_stats'])} captioner"
        )
        return reward_stats

    async def send_artifact_task(self, uid: int, role: MinerType, r2_source, block: int = 0):
        artifact_spec = self._artifact_spec()
        task_id = self._task_id(role, uid, r2_source, block, artifact_spec)
        parameters = {
            "assignment_time": time.time(),
            "expected_output": role.value.lower(),
            "assignment_block": block,
        }
        async with aiohttp.ClientSession() as session:
            response_data = await query_artifact_miner(
                uid=uid,
                axon_info=self.metagraph.axons[uid],
                session=session,
                hotkey=self.wallet.hotkey,
                role=role,
                task_id=task_id,
                r2_source=r2_source,
                parameters=parameters,
                artifact_spec=artifact_spec.to_dict() if artifact_spec else None,
                total_timeout=self.config.neuron.miner_total_timeout,
                connect_timeout=getattr(self.config.neuron, "miner_connect_timeout", None),
                sock_connect_timeout=getattr(
                    self.config.neuron, "miner_sock_connect_timeout", None
                ),
            )

        if response_data and response_data.get("accepted"):
            self.last_assignment_at[(uid, role)] = time.time()
            self.last_task_id_by_uid[(uid, role)] = task_id
            bt.logging.info(
                f"Assigned DPS {role.value.lower()} artifact task {task_id} to UID {uid}"
            )
        else:
            error = response_data.get("error") if response_data else "Unknown error"
            bt.logging.warning(
                f"Failed to assign DPS {role.value.lower()} artifact task to UID {uid}: {error}"
            )

    def _sample_role_uids(self, role: MinerType):
        miner_uids = self.miner_type_tracker.get_miners_by_type(role)
        sample_size = int(getattr(self.config.dps_artifact, "sample_size", 0))
        if sample_size > 0 and len(miner_uids) > sample_size:
            import numpy as np

            miner_uids = np.random.choice(
                miner_uids,
                size=sample_size,
                replace=False,
            ).tolist()
        return miner_uids

    async def _r2_source(self):
        metadata = await self.metadata_store.retrieve_artifact_metadata(
            uid=self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address),
            expected_kind="dps_input",
        )
        if metadata is not None:
            return metadata.r2.to_dict()

        r2_location = self._r2_location()
        return r2_location.to_dict() if r2_location is not None else {}

    def _r2_location(self):
        artifact_config = getattr(self.config, "dps_artifact", None)
        if artifact_config is None:
            return None

        bucket = getattr(artifact_config, "r2_bucket", None)
        path = getattr(artifact_config, "r2_prefix", None)
        if not bucket or not path:
            return None

        return ArtifactR2Location(
            bucket=bucket,
            path=path,
            endpoint_url=getattr(artifact_config, "r2_endpoint_url", None),
            region=getattr(artifact_config, "r2_region", None),
            access_key_id=getattr(
                artifact_config, "r2_read_access_key_id", None
            ),
            secret_access_key=getattr(
                artifact_config, "r2_read_secret_access_key", None
            ),
            session_token=getattr(artifact_config, "r2_read_session_token", None),
            manifest_url=getattr(artifact_config, "r2_manifest_url", None),
            manifest_key=getattr(artifact_config, "r2_manifest_key", None),
        )

    def _artifact_spec(self):
        artifact_config = getattr(self.config, "dps_artifact", None)
        if artifact_config is None:
            return None

        spec = ArtifactTaskSpec(
            resolution=getattr(artifact_config, "resolution", None),
            max_frames=getattr(artifact_config, "max_frames", None),
            encoding_model=getattr(artifact_config, "encoding_model", None),
        )
        return spec if spec.to_dict() else None

    def _verify_output_metadata(self, uid: int, metadata: ArtifactChainMetadata):
        try:
            stats = self.artifact_verifier.verify(
                uid=uid,
                hotkey=self.metagraph.hotkeys[uid],
                metadata=metadata,
            )
        except Exception as e:
            bt.logging.warning(f"DPS artifact verification failed for UID {uid}: {e}")
            stats = self._stats_from_output_metadata(uid, metadata)
            stats["accepted_work_units"] = 0
            if metadata.role == MinerType.ENCODER:
                stats["deterministic_correctness_rate"] = 0.0
            else:
                stats["caption_quality_rate"] = 0.0
            stats["penalties"] = max(1.0, stats.get("penalties", 0.0))
        return stats

    def _stats_from_output_metadata(self, uid: int, metadata: ArtifactChainMetadata):
        accepted = 1 if metadata.r2.bucket and metadata.r2.path else 0
        expected_task_ids = {
            task_id
            for (task_uid, _role), task_id in self.last_task_id_by_uid.items()
            if task_uid == uid
        }
        task_matches = (
            not expected_task_ids
            or metadata.task_id is None
            or metadata.task_id in expected_task_ids
        )
        has_integrity = bool(metadata.artifact_hash or metadata.r2.manifest_url or metadata.r2.manifest_key)
        quality = 1.0 if accepted and task_matches and has_integrity else 0.5 if accepted else 0.0
        stats = {
            "uid": uid,
            "hotkey": self.metagraph.hotkeys[uid],
            "task_id": metadata.task_id,
            "accepted_work_units": accepted,
            "availability_rate": 1.0 if accepted else 0.0,
            "timeliness_multiplier": 1.0,
            "novelty_multiplier": 1.0,
            "penalties": 0.0 if task_matches else 1.0,
            "artifact_bucket": metadata.r2.bucket,
            "artifact_path": metadata.r2.path,
            "artifact_hash": metadata.artifact_hash,
        }
        if metadata.role == MinerType.ENCODER:
            stats["deterministic_correctness_rate"] = quality
        else:
            stats["caption_quality_rate"] = quality
        return stats

    def _task_id(
        self,
        role: MinerType,
        uid: int,
        r2_source,
        block: int = 0,
        artifact_spec: ArtifactTaskSpec | None = None,
    ):
        interval = int(getattr(self.config, "dps_artifact_task_interval", 1) or 1)
        epoch = int(block // interval) if block else 0
        source_identity = {
            "source": r2_source,
            "artifact_spec": artifact_spec.to_dict() if artifact_spec else {},
        }
        source_hash = hashlib.sha256(
            json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        return f"dps-{role.value.lower()}-{uid}-e{epoch}-{source_hash}"
