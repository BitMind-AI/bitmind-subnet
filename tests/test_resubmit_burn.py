from types import SimpleNamespace

import pytest

from gas.protocol.miner_requests import generate_presigned_url, upload_single_modality
from gas.protocol.resubmit_burn import (
    BurnEvidence,
    ResubmitBurnError,
    alpha_rao_for_fee,
    clear_burn_receipt,
    evidence_from_response,
    execute_resubmit_burn,
    is_credit_used,
    is_submission_limit,
    load_burn_receipt,
    offer_resubmit_burn,
    save_burn_receipt,
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


def test_offer_requires_a_terminal(monkeypatch):
    monkeypatch.setattr("gas.protocol.resubmit_burn.sys.stdin.isatty", lambda: False)
    with pytest.raises(ResubmitBurnError, match="interactive"):
        offer_resubmit_burn(
            SimpleNamespace(),
            34,
            execute_fn=lambda *args, **kwargs: pytest.fail("should not burn"),
        )


def test_receipt_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("GAS_HOME", str(tmp_path))
    evidence = BurnEvidence(tx_hash="ab" * 32, block_number=11)
    save_burn_receipt("5Hot", 34, evidence)
    assert load_burn_receipt("5Hot", 34) == evidence
    clear_burn_receipt("5Hot", 34)
    assert load_burn_receipt("5Hot", 34) is None
    assert is_credit_used(
        {"status_code": 409, "response": {"detail": "This burn has already been used"}}
    )


def test_low_alpha_uses_add_stake_burn(tmp_path, monkeypatch):
    monkeypatch.setenv("GAS_HOME", str(tmp_path))
    monkeypatch.setattr("gas.protocol.resubmit_burn.sys.stdin.isatty", lambda: True)
    burned = SimpleNamespace(
        success=True,
        extrinsic_receipt=SimpleNamespace(
            extrinsic_hash="0x" + "ab" * 32, block_number=88
        ),
    )
    calls = []

    class Subtensor:
        def get_subnet_price(self, netuid):
            return SimpleNamespace(tao=0.5)

        def get_stake(self, coldkey, hotkey, netuid):
            return SimpleNamespace(rao=0)

        def get_balance(self, coldkey):
            return SimpleNamespace(tao=2.0)

        def add_stake_burn(self, wallet, netuid, hotkey, amount, **kwargs):
            calls.append((netuid, hotkey, amount))
            return burned

        def compose_call(self, *args, **kwargs):
            raise AssertionError("should use add_stake_burn, not burn_alpha")

    wallet = SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address="5Hot"),
        coldkeypub=SimpleNamespace(ss58_address="5Cold"),
    )
    evidence = execute_resubmit_burn(
        wallet,
        34,
        subtensor=Subtensor(),
        confirm_fn=lambda prompt: "y",
    )
    assert evidence.block_number == 88
    assert calls[0][0] == 34
    assert calls[0][1] == "5Hot"
    assert calls[0][2].tao == 0.5
    assert load_burn_receipt("5Hot", 34) == evidence


def test_offer_always_starts_the_burn_walkthrough(monkeypatch):
    monkeypatch.setattr("gas.protocol.resubmit_burn.sys.stdin.isatty", lambda: True)
    evidence = offer_resubmit_burn(
        SimpleNamespace(),
        34,
        execute_fn=lambda *args, **kwargs: BurnEvidence(
            tx_hash="cd" * 32, block_number=7
        ),
    )
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
    monkeypatch.setenv("GAS_HOME", str(tmp_path))
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
    assert load_burn_receipt("5Miner", 34) is None


def test_upload_reuses_saved_receipt_before_burning(tmp_path, monkeypatch):
    monkeypatch.setenv("GAS_HOME", str(tmp_path))
    model = tmp_path / "model.zip"
    model.write_bytes(b"zip")
    save_burn_receipt("5Miner", 34, BurnEvidence(tx_hash="f" * 64, block_number=9))
    calls = []

    def presign(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("burn_tx_hash") == "f" * 64:
            return {
                "success": True,
                "status_code": 200,
                "response": {
                    "data": {
                        "model_id": 3,
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
        resubmit=lambda: pytest.fail("should not burn again"),
    )
    assert result["success"] is True
    assert calls[1]["burn_tx_hash"] == "f" * 64
    assert load_burn_receipt("5Miner", 34) is None
