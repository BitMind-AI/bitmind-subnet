import hashlib
import json

from gas.artifacts.r2_transport import ArtifactR2Transport, join_key
from gas.types import ArtifactChainMetadata, MinerType


class ArtifactVerificationError(Exception):
    pass


class ArtifactVerifier:
    """Verifies DPS artifact manifests and files before rewards are updated."""

    def __init__(self, transport: ArtifactR2Transport | None = None):
        self.transport = transport or ArtifactR2Transport()

    def verify(self, uid: int, hotkey: str, metadata: ArtifactChainMetadata) -> dict:
        manifest = self._load_manifest(metadata)
        self._verify_manifest_spec(metadata, manifest)
        files = self._manifest_files(manifest)
        if not files:
            raise ArtifactVerificationError("Artifact manifest contained no files")

        valid_units = 0
        invalid_units = 0
        for file_info in files:
            try:
                content = self._read_artifact_file(metadata, file_info)
                expected_sha = file_info.get("sha256") or file_info.get("hash")
                if expected_sha and self._sha256(content) != self._normalize_hash(expected_sha):
                    invalid_units += int(file_info.get("work_units", 1))
                    continue
                if (
                    metadata.artifact_hash
                    and len(files) == 1
                    and self._sha256(content)
                    != self._normalize_hash(metadata.artifact_hash)
                ):
                    invalid_units += int(file_info.get("work_units", 1))
                    continue
                valid_units += int(file_info.get("work_units", 1))
            except Exception:
                invalid_units += int(file_info.get("work_units", 1))

        total_units = valid_units + invalid_units
        correctness = valid_units / total_units if total_units else 0.0
        stats = {
            "uid": uid,
            "hotkey": hotkey,
            "task_id": metadata.task_id,
            "accepted_work_units": valid_units,
            "availability_rate": 1.0 if valid_units else 0.0,
            "timeliness_multiplier": 1.0,
            "novelty_multiplier": 1.0,
            "penalties": float(invalid_units),
            "artifact_bucket": metadata.r2.bucket,
            "artifact_path": metadata.r2.path,
            "artifact_hash": metadata.artifact_hash,
            "manifest_sha256": self._sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ),
        }
        if metadata.role == MinerType.ENCODER:
            stats["deterministic_correctness_rate"] = correctness
        else:
            stats["caption_quality_rate"] = correctness
        return stats

    def _load_manifest(self, metadata: ArtifactChainMetadata) -> dict:
        if metadata.r2.manifest_url:
            data = self.transport.read_url(metadata.r2.manifest_url)
        else:
            manifest_key = metadata.r2.manifest_key or "manifest.json"
            data = self._read_location(metadata, manifest_key)
        try:
            manifest = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ArtifactVerificationError("Artifact manifest was not valid JSON") from e
        if not isinstance(manifest, dict):
            raise ArtifactVerificationError("Artifact manifest must be a JSON object")
        return manifest

    def _verify_manifest_spec(self, metadata: ArtifactChainMetadata, manifest: dict) -> None:
        expected = metadata.artifact_spec.to_dict() if metadata.artifact_spec else {}
        actual = manifest.get("artifact_spec") or {}
        for key, value in expected.items():
            if value is not None and str(actual.get(key, "")) != str(value):
                raise ArtifactVerificationError(
                    f"Artifact manifest does not match required {key}"
                )

    def _manifest_files(self, manifest: dict) -> list[dict]:
        files = manifest.get("artifacts") or manifest.get("files") or []
        return [file_info for file_info in files if isinstance(file_info, dict)]

    def _read_artifact_file(self, metadata: ArtifactChainMetadata, file_info: dict) -> bytes:
        url = file_info.get("url")
        if url:
            return self._read_url(url)
        path = file_info.get("path") or file_info.get("key")
        if not path:
            raise ArtifactVerificationError("Artifact file is missing path")
        return self._read_location(metadata, path)

    def _read_location(self, metadata: ArtifactChainMetadata, relative_path: str) -> bytes:
        return self.transport.read_object(
            endpoint_url=metadata.r2.endpoint_url or "",
            bucket=metadata.r2.bucket,
            key=join_key(metadata.r2.path, relative_path),
            region=metadata.r2.region or "auto",
            access_key_id=metadata.r2.access_key_id or "",
            secret_access_key=metadata.r2.secret_access_key or "",
            session_token=metadata.r2.session_token or "",
        )

    def _sha256(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _normalize_hash(self, value: str) -> str:
        return value.removeprefix("sha256:")
