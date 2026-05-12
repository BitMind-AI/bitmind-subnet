import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from gas.types import ArtifactChainMetadata, MinerType


class ArtifactVerificationError(Exception):
    pass


class ArtifactVerifier:
    """Verifies DPS artifact manifests and files before rewards are updated."""

    def verify(self, uid: int, hotkey: str, metadata: ArtifactChainMetadata) -> dict:
        manifest = self._load_manifest(metadata)
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
            data = self._read_url(metadata.r2.manifest_url)
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
        endpoint_url = metadata.r2.endpoint_url or ""
        if endpoint_url.startswith("file://"):
            root = Path(urlparse(endpoint_url).path)
            path = root / metadata.r2.bucket / metadata.r2.path / relative_path
            return path.read_bytes()
        if endpoint_url.startswith(("http://", "https://")):
            base = endpoint_url.rstrip("/")
            url = f"{base}/{metadata.r2.bucket}/{metadata.r2.path.strip('/')}/{relative_path}"
            return self._read_url(url)
        path = Path(metadata.r2.bucket) / metadata.r2.path / relative_path
        return path.read_bytes()

    def _read_url(self, url: str) -> bytes:
        parsed = urlparse(url)
        if parsed.scheme == "file":
            return Path(parsed.path).read_bytes()
        if parsed.scheme in ("http", "https"):
            with urlopen(url, timeout=30) as response:
                return response.read()
        return Path(url).read_bytes()

    def _sha256(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _normalize_hash(self, value: str) -> str:
        return value.removeprefix("sha256:")
