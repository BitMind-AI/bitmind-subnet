from types import SimpleNamespace

import pytest

from gas.protocol.miner_requests import generate_presigned_url, upload_single_modality
from gas.protocol.resubmit_burn import (
    BurnEvidence,
    ResubmitBurnError,
    alpha_rao_for_fee,
    evidence_from_response,
    is_submission_limit,
    offer_resubmit_burn,
)


def test_detects_free_slot_used():
    assert is_submission_limit(
        {
            "status_code": 403,
            "response": {
                "detail": "This registration has already used its free submission. Burn 0.5 TAO"
            },
        }
    )
    assert is_submission_limit(
        {
            "status_code": 403,
            "response": {"detail": "This registration has already used its one submission"},
        }
    )
    assert not is_submission_limit(
        {"status_code": 403, "response": {"detail": "Hotkey is not registered"}}
    )
    assert not is_submission_limit({"status_code": 409, "response": {"detail": "already used"}})


def test_alpha_amount_includes_price_slack():
    assert alpha_rao_for_fee(0.5) == 1_020_000_001
    with pytest.raises(ResubmitBurnError):
        alpha_rao_for_fee(0)


def test_evidence_from_successful_receipt():
    evidence = evidence_from_response(
        SimpleNamespace(
            success=True,
            extrinsic_receipt=SimpleNamespace(
                extrinsic_hash="0x" + "ab" * 32,
                block_number=99,
            ),
        )
    )
    assert evidence == BurnEvidence(tx_hash="ab" * 32, block_number=99)
    with pytest.raises(ResubmitBurnError, match="failed"):
        evidence_from_response(SimpleNamespace(success=False, message="no stake"))


def test_offer_declines_without_burning():
    answers = iter(["n"])
    evidence = offer_resubmit_burn(
        SimpleNamespace(),
        34,
        enabled=True,
        auto_confirm=False,
        confirm_fn=lambda prompt: next(answers),
        execute_fn=lambda *args, **kwargs: pytest.fail("should not burn"),
    )
    assert evidence is None


def test_offer_walks_through_when_confirmed(monkeypatch):
    monkeypatch.setattr("gas.protocol.resubmit_burn.sys.stdin.isatty", lambda: True)
    called = {}

    def execute(*args, **kwargs):
        called["ok"] = True
        return BurnEvidence(tx_hash="cd" * 32, block_number=7)

    evidence = offer_resubmit_burn(
        SimpleNamespace(),
        34,
        auto_confirm=False,
        confirm_fn=lambda prompt: "yes",
        execute_fn=execute,
    )
    assert called["ok"] is True
    assert evidence.block_number == 7


def test_presigned_payload_includes_burn_proof(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        def json(self):
            return {"success": True, "data": {"model_id": 1}}

    def post(url, data, headers, timeout):
        captured["data"] = data
        return Response()

    monkeypatch.setattr("gas.protocol.miner_requests.requests.post", post)
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Miner"))
    monkeypatch.setattr(
        "gas.protocol.miner_requests.generate_header",
        lambda *args, **kwargs: {},
    )
    result = generate_presigned_url(
        wallet,
        "https://upload.example/upload",
        "model.zip",
        10,
        "a" * 64,
        burn_tx_hash="b" * 64,
        burn_block=11,
    )
    assert result["success"] is True
    assert b'"burn_tx_hash":"' + b"b" * 64 in captured["data"]
    assert b'"burn_block":11' in captured["data"]


def test_upload_retries_after_interactive_burn(tmp_path, monkeypatch):
    model = tmp_path / "model.zip"
    model.write_bytes(b"zip")
    calls = []

    def presign(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("burn_tx_hash"):
            return {
                "success": True,
                "status_code": 200,
                "response": {
                    "data": {
                        "model_id": 9,
                        "presigned_url": "https://r2.example/put",
                        "r2_key": "miner/key",
                    }
                },
            }
        return {
            "success": False,
            "status_code": 403,
            "response": {"detail": "already used its free submission. Burn 0.5 TAO"},
        }

    monkeypatch.setattr("gas.protocol.miner_requests.generate_presigned_url", presign)
    monkeypatch.setattr(
        "gas.protocol.miner_requests.upload_to_r2",
        lambda *args, **kwargs: {"success": True, "response": {}},
    )
    monkeypatch.setattr(
        "gas.protocol.miner_requests.confirm_upload",
        lambda *args, **kwargs: {"success": True, "response": {"data": {}}},
    )
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Miner"))
    result = upload_single_modality(
        wallet,
        str(model),
        "image",
        "https://upload.example/upload",
        resubmit=lambda: BurnEvidence(tx_hash="e" * 64, block_number=42),
    )
    assert result["success"] is True
    assert calls[1]["burn_tx_hash"] == "e" * 64
    assert calls[1]["burn_block"] == 42
