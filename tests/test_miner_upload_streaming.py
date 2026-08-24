"""Behavioral tests for discriminator model uploads."""

import hashlib
from types import SimpleNamespace

from gas.protocol import miner_requests


def test_file_hash_is_computed_incrementally(tmp_path):
    model = tmp_path / "model.zip"
    payload = b"streamed-model-data" * 100
    model.write_bytes(payload)

    assert miner_requests.calculate_file_sha256(
        model,
        chunk_size=7,
    ) == hashlib.sha256(payload).hexdigest()


def test_r2_upload_streams_file_and_reports_progress(tmp_path, monkeypatch, capsys):
    model = tmp_path / "model.zip"
    payload = b"0123456789" * 100
    model.write_bytes(payload)
    captured = {}

    def consume_upload(url, data, headers, timeout):
        captured["url"] = url
        captured["body_type"] = type(data)
        captured["headers"] = headers
        captured["timeout"] = timeout
        chunks = []
        while chunk := data.read(17):
            chunks.append(chunk)
        captured["payload"] = b"".join(chunks)
        return SimpleNamespace(status_code=200, headers={"ETag": "etag"}, text="")

    monkeypatch.setattr(miner_requests.requests, "put", consume_upload)

    result = miner_requests.upload_to_r2("https://r2.example/upload", model)

    assert result["success"] is True
    assert captured["payload"] == payload
    assert captured["body_type"] is miner_requests.UploadProgressReader
    assert captured["headers"]["Content-Length"] == str(len(payload))
    assert captured["timeout"] == 300
    assert "100.0%" in capsys.readouterr().out
