import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


def main():
    task_id = os.environ["DPS_TASK_ID"]
    role = os.environ.get("DPS_ROLE", "ENCODER").upper()
    output_bucket = os.environ["DPS_OUTPUT_BUCKET"]
    output_prefix = os.environ["DPS_OUTPUT_PREFIX"].strip("/")
    output_endpoint = os.environ.get("DPS_OUTPUT_ENDPOINT_URL", "")

    source_items = _load_source_items()
    artifact_name = "encodings.jsonl" if role == "ENCODER" else "captions.jsonl"
    artifact_bytes = _build_artifact_bytes(task_id, role, source_items)
    artifact_sha = _sha256(artifact_bytes)

    output_dir = _output_dir(output_endpoint, output_bucket, output_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / artifact_name
    artifact_path.write_bytes(artifact_bytes)

    manifest = {
        "task_id": task_id,
        "role": role,
        "artifacts": [
            {
                "path": artifact_name,
                "sha256": artifact_sha,
                "work_units": max(1, len(source_items)),
                "format": "jsonl",
            }
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")

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
        raw = _read_url(manifest_url)
    else:
        source_dir = _source_dir()
        manifest_path = source_dir / manifest_key
        raw = manifest_path.read_bytes() if manifest_path.exists() else b"{}"

    try:
        manifest = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        manifest = {}
    items = manifest.get("items") or manifest.get("files") or manifest.get("artifacts") or []
    if not items:
        source_path = os.environ.get("DPS_SOURCE_PATH", "")
        items = [{"path": source_path or "source"}]
    return items


def _build_artifact_bytes(task_id: str, role: str, source_items: list) -> bytes:
    rows = []
    for index, item in enumerate(source_items):
        item_key = item.get("path") or item.get("key") or item.get("url") or str(index)
        digest = _sha256(f"{task_id}:{role}:{item_key}".encode("utf-8"))
        if role == "CAPTIONER":
            payload = {"item": item_key, "caption": f"dps caption {digest[:16]}"}
        else:
            payload = {"item": item_key, "encoding": digest}
        rows.append(json.dumps(payload, separators=(",", ":")))
    return ("\n".join(rows) + "\n").encode("utf-8")


def _source_dir() -> Path:
    endpoint = os.environ.get("DPS_SOURCE_ENDPOINT_URL", "")
    bucket = os.environ.get("DPS_SOURCE_BUCKET", "")
    source_path = os.environ.get("DPS_SOURCE_PATH", "").strip("/")
    if endpoint.startswith("file://"):
        return Path(urlparse(endpoint).path) / bucket / source_path
    return Path(bucket) / source_path


def _output_dir(endpoint: str, bucket: str, prefix: str) -> Path:
    if endpoint.startswith("file://"):
        return Path(urlparse(endpoint).path) / bucket / prefix
    return Path(bucket) / prefix


def _read_url(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return Path(parsed.path).read_bytes()
    if parsed.scheme in ("http", "https"):
        with urlopen(url, timeout=30) as response:
            return response.read()
    return Path(url).read_bytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()
