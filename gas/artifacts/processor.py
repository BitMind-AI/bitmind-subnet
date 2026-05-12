import hashlib
import json
import os

from gas.artifacts.r2_transport import ArtifactR2Transport, join_key


def main():
    task_id = os.environ["DPS_TASK_ID"]
    role = os.environ.get("DPS_ROLE", "ENCODER").upper()
    output_bucket = os.environ["DPS_OUTPUT_BUCKET"]
    output_prefix = os.environ["DPS_OUTPUT_PREFIX"].strip("/")
    output_endpoint = os.environ.get("DPS_OUTPUT_ENDPOINT_URL", "")
    output_region = os.environ.get("DPS_OUTPUT_REGION", "auto")
    output_write_key = os.environ.get("DPS_OUTPUT_WRITE_ACCESS_KEY_ID", "")
    output_write_secret = os.environ.get("DPS_OUTPUT_WRITE_SECRET_ACCESS_KEY", "")
    output_write_token = os.environ.get("DPS_OUTPUT_WRITE_SESSION_TOKEN", "")

    source_items = _load_source_items()
    artifact_name = "encodings.jsonl" if role == "ENCODER" else "captions.jsonl"
    artifact_bytes = _build_artifact_bytes(task_id, role, source_items)
    artifact_sha = _sha256(artifact_bytes)

    transport = ArtifactR2Transport()
    artifact_key = join_key(output_prefix, artifact_name)
    transport.write_object(
        endpoint_url=output_endpoint,
        bucket=output_bucket,
        key=artifact_key,
        data=artifact_bytes,
        content_type="application/jsonl",
        region=output_region,
        access_key_id=output_write_key,
        secret_access_key=output_write_secret,
        session_token=output_write_token,
    )

    manifest = {
        "task_id": task_id,
        "role": role,
        "artifact_spec": _artifact_spec(),
        "artifacts": [
            {
                "path": artifact_name,
                "sha256": artifact_sha,
                "work_units": max(1, len(source_items)),
                "format": "jsonl",
            }
        ],
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    transport.write_object(
        endpoint_url=output_endpoint,
        bucket=output_bucket,
        key=join_key(output_prefix, "manifest.json"),
        data=manifest_bytes,
        content_type="application/json",
        region=output_region,
        access_key_id=output_write_key,
        secret_access_key=output_write_secret,
        session_token=output_write_token,
    )

    print(
        json.dumps(
            {
                "artifact_hash": f"sha256:{artifact_sha}",
                "manifest_key": "manifest.json",
                "path": f"{output_prefix}/",
            },
            separators=(",", ":"),
        )
    )


def _load_source_items():
    manifest_url = os.environ.get("DPS_SOURCE_MANIFEST_URL")
    manifest_key = os.environ.get("DPS_SOURCE_MANIFEST_KEY") or "manifest.json"
    if manifest_url:
        raw = ArtifactR2Transport().read_url(manifest_url)
    else:
        raw = _read_source_object(manifest_key)

    try:
        manifest = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        manifest = {}
    items = manifest.get("items") or manifest.get("files") or manifest.get("artifacts") or []
    if not items:
        source_path = os.environ.get("DPS_SOURCE_PATH", "")
        items = [{"path": source_path or "source"}]
    return items


def _read_source_object(relative_key: str) -> bytes:
    transport = ArtifactR2Transport()
    try:
        return transport.read_object(
            endpoint_url=os.environ.get("DPS_SOURCE_ENDPOINT_URL", ""),
            bucket=os.environ.get("DPS_SOURCE_BUCKET", ""),
            key=join_key(os.environ.get("DPS_SOURCE_PATH", ""), relative_key),
            region=os.environ.get("DPS_SOURCE_REGION", "auto"),
            access_key_id=os.environ.get("DPS_SOURCE_ACCESS_KEY_ID", ""),
            secret_access_key=os.environ.get("DPS_SOURCE_SECRET_ACCESS_KEY", ""),
            session_token=os.environ.get("DPS_SOURCE_SESSION_TOKEN", ""),
        )
    except FileNotFoundError:
        return b"{}"


def _build_artifact_bytes(task_id: str, role: str, source_items: list) -> bytes:
    rows = []
    for index, item in enumerate(source_items):
        item_key = item.get("path") or item.get("key") or item.get("url") or str(index)
        digest = _sha256(f"{task_id}:{role}:{item_key}".encode("utf-8"))
        if role == "CAPTIONER":
            payload = {"item": item_key, "caption": f"dps caption {digest[:16]}"}
        else:
            payload = {
                "item": item_key,
                "encoding": digest,
                "resolution": os.environ.get("DPS_ARTIFACT_RESOLUTION", ""),
                "max_frames": os.environ.get("DPS_ARTIFACT_MAX_FRAMES", ""),
                "encoding_model": os.environ.get("DPS_ARTIFACT_ENCODING_MODEL", ""),
            }
        rows.append(json.dumps(payload, separators=(",", ":")))
    return ("\n".join(rows) + "\n").encode("utf-8")


def _artifact_spec() -> dict:
    spec = {
        "resolution": os.environ.get("DPS_ARTIFACT_RESOLUTION"),
        "max_frames": os.environ.get("DPS_ARTIFACT_MAX_FRAMES"),
        "encoding_model": os.environ.get("DPS_ARTIFACT_ENCODING_MODEL"),
    }
    return {key: value for key, value in spec.items() if value}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()
