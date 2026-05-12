import unittest
import json
import os
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from gas.artifacts.processor import main as artifact_processor_main
from gas.artifacts.r2_transport import ArtifactR2Transport
from gas.evaluation.artifact_verifier import ArtifactVerifier
from gas.evaluation.rewards import (
    artifact_stats_with_uids,
    get_captioner_rewards,
    get_encoder_rewards,
    normalize_rewards_to_weight_budget,
)
from gas.protocol.validator_requests import build_artifact_task_payload
from gas.types import (
    ArtifactChainMetadata,
    ArtifactR2Location,
    ArtifactTaskSpec,
    ChainMetadataRegistry,
    DiscriminatorModelId,
    MinerType,
)


class EncoderIncentiveTest(unittest.TestCase):
    def test_encoder_miner_type_exists(self):
        self.assertEqual(MinerType.ENCODER.value, "ENCODER")
        self.assertEqual(MinerType.CAPTIONER.value, "CAPTIONER")

    def test_encoder_rewards_use_dps_formula(self):
        rewards = get_encoder_rewards(
            [
                {
                    "uid": 2,
                    "accepted_work_units": 10,
                    "deterministic_correctness_rate": 0.9,
                    "availability_rate": 0.8,
                    "timeliness_multiplier": 1.1,
                    "novelty_multiplier": 1.2,
                    "penalties": 1.0,
                },
                {
                    "uid": 3,
                    "accepted_work_units": 5,
                    "deterministic_correctness_rate": 0.2,
                    "availability_rate": 0.5,
                    "penalties": 10.0,
                },
            ]
        )

        self.assertAlmostEqual(rewards[2], 8.504)
        self.assertEqual(rewards[3], 0.0)

    def test_encoder_rewards_accept_existing_verification_aliases(self):
        rewards = get_encoder_rewards(
            {
                "hotkey": {
                    "uid": 4,
                    "total_verified": 8,
                    "pass_rate": 0.75,
                }
            }
        )

        self.assertEqual(rewards[4], 6.0)

    def test_captioner_rewards_use_quality_aliases(self):
        rewards = get_captioner_rewards(
            [
                {
                    "uid": 5,
                    "accepted_work_units": 4,
                    "caption_quality_rate": 0.7,
                    "availability_rate": 0.5,
                    "timeliness_multiplier": 1.0,
                    "novelty_multiplier": 1.5,
                    "penalties": 0.1,
                },
                {
                    "uid": 6,
                    "accepted_work_units": 3,
                    "quality_score": 0.8,
                },
            ]
        )

        self.assertAlmostEqual(rewards[5], 2.0)
        self.assertAlmostEqual(rewards[6], 2.4)

    def test_normalize_rewards_to_budget_excludes_specials(self):
        weights, unallocated = normalize_rewards_to_weight_budget(
            scores=np.array([0.0, 2.0, 3.0, 10.0], dtype=np.float32),
            active_uids=[1, 2, 3],
            special_uids={3},
            budget=0.10,
        )

        self.assertEqual(unallocated, 0.0)
        self.assertEqual(weights[0], 0.0)
        self.assertAlmostEqual(weights[1], 0.04, places=6)
        self.assertAlmostEqual(weights[2], 0.06, places=6)
        self.assertEqual(weights[3], 0.0)

    def test_normalize_rewards_returns_unallocated_budget_without_scores(self):
        weights, unallocated = normalize_rewards_to_weight_budget(
            scores=np.zeros(3, dtype=np.float32),
            active_uids=[1, 2],
            special_uids=set(),
            budget=0.10,
        )

        self.assertTrue(np.all(weights == 0.0))
        self.assertEqual(unallocated, 0.10)

    def test_artifact_stats_accept_hotkey_keyed_records(self):
        payload = artifact_stats_with_uids(
            {
                "hk1": {
                    "accepted_work_units": 4,
                    "deterministic_correctness_rate": 0.5,
                },
                "hk2": {
                    "accepted_work_units": 5,
                    "deterministic_correctness_rate": 0.8,
                },
            },
            hotkeys=["hk0", "hk1", "hk2"],
        )

        rewards = get_encoder_rewards(payload)

        self.assertEqual(payload[0]["uid"], 1)
        self.assertEqual(payload[1]["uid"], 2)
        self.assertEqual(rewards[1], 2.0)
        self.assertEqual(rewards[2], 4.0)

    def test_artifact_task_payload_points_miners_at_r2_source(self):
        payload = build_artifact_task_payload(
            role=MinerType.ENCODER,
            task_id="task-1",
            r2_source={
                "endpoint_url": "https://abc.r2.cloudflarestorage.com",
                "bucket": "dps",
                "path": "encoder/shard-001/",
                "manifest_url": "https://example.com/manifest.json",
                "access_key_id": "read-key",
                "secret_access_key": "read-secret",
                "ignored": None,
            },
            parameters={"expected_output": "encoder"},
            artifact_spec={
                "resolution": "512x512",
                "max_frames": 16,
                "encoding_model": "stabilityai/sd-vae-ft-mse",
            },
        )

        self.assertEqual(payload["task_id"], "task-1")
        self.assertEqual(payload["role"], "ENCODER")
        self.assertEqual(payload["source"]["type"], "r2")
        self.assertEqual(payload["source"]["bucket"], "dps")
        self.assertEqual(payload["source"]["path"], "encoder/shard-001/")
        self.assertEqual(payload["source"]["prefix"], "encoder/shard-001/")
        self.assertEqual(payload["source"]["access_key_id"], "read-key")
        self.assertEqual(payload["source"]["secret_access_key"], "read-secret")
        self.assertNotIn("ignored", payload["source"])
        self.assertEqual(payload["artifact_spec"]["resolution"], "512x512")
        self.assertEqual(payload["artifact_spec"]["max_frames"], 16)
        self.assertEqual(
            payload["artifact_spec"]["encoding_model"], "stabilityai/sd-vae-ft-mse"
        )
        self.assertEqual(payload["parameters"]["expected_output"], "encoder")

    def test_artifact_chain_metadata_round_trip(self):
        metadata = ArtifactChainMetadata(
            kind="dps_output",
            role=MinerType.ENCODER,
            task_id="task-1",
            artifact_format="npz",
            artifact_hash="sha256:abc",
            artifact_spec=ArtifactTaskSpec(
                resolution="512x512",
                max_frames=16,
                encoding_model="vae-a",
            ),
            r2=ArtifactR2Location(
                bucket="miner-output",
                path="encodings/task-1/",
                endpoint_url="https://miner.r2.cloudflarestorage.com",
                access_key_id="validator-read-key",
                secret_access_key="validator-read-secret",
            ),
        )

        decoded = ArtifactChainMetadata.from_compressed_str(
            metadata.to_compressed_str()
        )

        self.assertEqual(decoded.kind, "dps_output")
        self.assertEqual(decoded.role, MinerType.ENCODER)
        self.assertEqual(decoded.task_id, "task-1")
        self.assertEqual(decoded.artifact_format, "npz")
        self.assertEqual(decoded.artifact_spec.resolution, "512x512")
        self.assertEqual(decoded.artifact_spec.max_frames, 16)
        self.assertEqual(decoded.artifact_spec.encoding_model, "vae-a")
        self.assertEqual(decoded.r2.bucket, "miner-output")
        self.assertEqual(decoded.r2.path, "encodings/task-1/")
        self.assertEqual(decoded.r2.access_key_id, "validator-read-key")

    def test_chain_registry_preserves_discriminator_and_multiple_artifacts(self):
        registry = ChainMetadataRegistry(
            discriminator_model=DiscriminatorModelId(key="models/detector.zip", hash="abc")
        )
        registry.upsert_artifact(
            ArtifactChainMetadata(
                kind="dps_input",
                role=MinerType.ENCODER,
                r2=ArtifactR2Location(bucket="validator-input", path="shard-1/"),
            )
        )
        registry.upsert_artifact(
            ArtifactChainMetadata(
                kind="dps_output",
                role=MinerType.ENCODER,
                task_id="task-1",
                r2=ArtifactR2Location(bucket="miner-output", path="task-1/"),
            )
        )

        decoded = ChainMetadataRegistry.from_compressed_str(
            registry.to_compressed_str()
        )

        self.assertEqual(decoded.discriminator_model.key, "models/detector.zip")
        self.assertEqual(
            decoded.get_artifacts(expected_kind="dps_input")[0].r2.bucket,
            "validator-input",
        )
        self.assertEqual(
            decoded.get_artifacts(expected_kind="dps_output", task_id="task-1")[0].r2.path,
            "task-1/",
        )

    def test_fake_r2_processor_and_validator_verifier_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "validator-input" / "shard-1"
            source_dir.mkdir(parents=True)
            source_manifest = {
                "items": [
                    {"path": "sample-1.bin"},
                    {"path": "sample-2.bin"},
                ]
            }
            (source_dir / "manifest.json").write_text(json.dumps(source_manifest))

            old_env = os.environ.copy()
            try:
                os.environ.update(
                    {
                        "DPS_TASK_ID": "task-1",
                        "DPS_ROLE": "ENCODER",
                        "DPS_SOURCE_ENDPOINT_URL": f"file://{root}",
                        "DPS_SOURCE_BUCKET": "validator-input",
                        "DPS_SOURCE_PATH": "shard-1",
                        "DPS_OUTPUT_ENDPOINT_URL": f"file://{root}",
                        "DPS_OUTPUT_BUCKET": "miner-output",
                        "DPS_OUTPUT_PREFIX": "encodings/task-1",
                        "DPS_ARTIFACT_RESOLUTION": "512x512",
                        "DPS_ARTIFACT_MAX_FRAMES": "16",
                        "DPS_ARTIFACT_ENCODING_MODEL": "vae-a",
                    }
                )
                with redirect_stdout(StringIO()):
                    artifact_processor_main()
            finally:
                os.environ.clear()
                os.environ.update(old_env)

            metadata = ArtifactChainMetadata(
                kind="dps_output",
                role=MinerType.ENCODER,
                task_id="task-1",
                artifact_format="jsonl",
                artifact_spec=ArtifactTaskSpec(
                    resolution="512x512",
                    max_frames=16,
                    encoding_model="vae-a",
                ),
                r2=ArtifactR2Location(
                    bucket="miner-output",
                    path="encodings/task-1",
                    endpoint_url=f"file://{root}",
                    manifest_key="manifest.json",
                ),
            )

            stats = ArtifactVerifier().verify(
                uid=7,
                hotkey="hk7",
                metadata=metadata,
            )

            self.assertEqual(stats["accepted_work_units"], 2)
            self.assertEqual(stats["deterministic_correctness_rate"], 1.0)
            self.assertEqual(stats["penalties"], 0.0)

    def test_verifier_rejects_bad_single_artifact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "miner-output" / "encodings" / "task-1"
            output_dir.mkdir(parents=True)
            (output_dir / "encodings.jsonl").write_text("{}\n")
            manifest = {
                "artifacts": [
                    {
                        "path": "encodings.jsonl",
                        "sha256": "sha256:"
                        + "0" * 64,
                        "work_units": 3,
                    }
                ]
            }
            (output_dir / "manifest.json").write_text(json.dumps(manifest))

            metadata = ArtifactChainMetadata(
                kind="dps_output",
                role=MinerType.ENCODER,
                task_id="task-1",
                artifact_hash="sha256:" + "1" * 64,
                r2=ArtifactR2Location(
                    bucket="miner-output",
                    path="encodings/task-1",
                    endpoint_url=f"file://{root}",
                    manifest_key="manifest.json",
                ),
            )

            stats = ArtifactVerifier().verify(uid=7, hotkey="hk7", metadata=metadata)

            self.assertEqual(stats["accepted_work_units"], 0)
            self.assertEqual(stats["deterministic_correctness_rate"], 0.0)
            self.assertEqual(stats["penalties"], 3.0)

    def test_transport_uses_s3_client_for_authenticated_r2(self):
        calls = {}

        class Body:
            def read(self):
                return b"payload"

        class Client:
            def get_object(self, Bucket, Key):
                calls["get"] = (Bucket, Key)
                return {"Body": Body()}

            def put_object(self, Bucket, Key, Body, ContentType):
                calls["put"] = (Bucket, Key, Body, ContentType)

        transport = ArtifactR2Transport()
        transport._s3_client = lambda **_: Client()

        data = transport.read_object(
            endpoint_url="https://abc.r2.cloudflarestorage.com",
            bucket="dps",
            key="source/manifest.json",
            access_key_id="key",
            secret_access_key="secret",
        )
        transport.write_object(
            endpoint_url="https://abc.r2.cloudflarestorage.com",
            bucket="out",
            key="task/manifest.json",
            data=b"{}",
            content_type="application/json",
            access_key_id="key",
            secret_access_key="secret",
        )

        self.assertEqual(data, b"payload")
        self.assertEqual(calls["get"], ("dps", "source/manifest.json"))
        self.assertEqual(
            calls["put"],
            ("out", "task/manifest.json", b"{}", "application/json"),
        )


if __name__ == "__main__":
    unittest.main()
