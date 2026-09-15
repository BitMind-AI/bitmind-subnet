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
    is_duplicate_upload,
    is_submission_limit,
    load_burn_receipt,
    min_alpha_rao_for_fee,
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


@pytest.mark.parametrize("status,detail,duplicate", [
    (409, "File with this hash already exists", True),
    (400, "File with this hash already exists", False),
    (409, "This burn already belongs to another registration", False),
    (409, "This burn does not match the current registration", False),
    (409, "This burn has already been used", False),
    (409, "Registration changed; please retry with a fresh registration lookup", False),
    (409, "A submission is already in progress for this registration", False),
    (409, "Upload reservation expired; please retry", False),
    (409, "Unrecognized conflict", False),
    (409, "", False),
])
def test_only_explicit_duplicate_response_can_skip_upload(status, detail, duplicate):
    assert is_duplicate_upload({"status_code": status, "response": {"detail": detail}}) is duplicate


@pytest.mark.parametrize("saved_receipt", [False, True])
@pytest.mark.parametrize("detail,duplicate", [
    ("File with this hash already exists", True),
    ("This burn already belongs to another registration", False),
    ("This burn does not match the current registration", False),
    ("Registration changed; please retry with a fresh registration lookup", False),
    ("A submission is already in progress for this registration", False),
    ("Upload reservation expired; please retry", False),
    ("Unrecognized conflict", False),
])
def test_upload_conflicts_fail_closed_before_or_after_receipt_retry(
    tmp_path, monkeypatch, saved_receipt, detail, duplicate,
):
    monkeypatch.setenv("GAS_HOME", str(tmp_path))
    model = tmp_path / "model.zip"
    model.write_bytes(b"zip")
    evidence = BurnEvidence(tx_hash="a" * 64, block_number=123)
    responses = []
    if saved_receipt:
        save_burn_receipt("5Miner", 34, evidence)
        responses.append({
            "success": False, "status_code": 403,
            "response": {"detail": "This registration has already used its free submission"},
        })
    responses.append({"success": False, "status_code": 409, "response": {"detail": detail}})
    iterator = iter(responses)
    monkeypatch.setattr("gas.protocol.miner_requests.generate_presigned_url", lambda *a, **kw: next(iterator))
    monkeypatch.setattr("gas.protocol.miner_requests.upload_to_r2", lambda *a, **kw: pytest.fail("must not upload"))
    monkeypatch.setattr("gas.protocol.miner_requests.confirm_upload", lambda *a, **kw: pytest.fail("must not confirm"))
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Miner"))
    result = upload_single_modality(
        wallet, str(model), "image", "https://upload.example/upload",
        resubmit=lambda: pytest.fail("conflict must not trigger another burn"),
    )
    assert result["success"] is False
    assert result.get("already_uploaded", False) is duplicate
    assert result["error"] == f"HTTP 409: {detail}"
    assert load_burn_receipt("5Miner", 34) == (evidence if saved_receipt else None)


def test_alpha_amount_includes_price_slack():
    assert min_alpha_rao_for_fee(0.5) == 1_000_000_000
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


def test_exact_fee_alpha_burns_without_slack_padding(tmp_path, monkeypatch):
    monkeypatch.setenv("GAS_HOME", str(tmp_path))
    monkeypatch.setattr("gas.protocol.resubmit_burn.sys.stdin.isatty", lambda: True)
    burned = SimpleNamespace(
        success=True,
        extrinsic_receipt=SimpleNamespace(
            extrinsic_hash="0x" + "cd" * 32, block_number=12
        ),
    )
    amounts = []

    class Subtensor:
        def get_subnet_price(self, netuid):
            return SimpleNamespace(tao=0.5)

        def get_stake(self, coldkey, hotkey, netuid):
            return SimpleNamespace(rao=1_000_000_000)

        def get_balance(self, coldkey):
            raise AssertionError("should not buy TAO when the fee is already staked")

        def compose_call(self, module, function, params):
            amounts.append(params["amount"])
            return "call"

        def sign_and_send_extrinsic(self, call, wallet, **kwargs):
            return burned

        def add_stake_burn(self, *args, **kwargs):
            raise AssertionError("should burn_alpha, not add_stake_burn")

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
    assert evidence.block_number == 12
    assert amounts == [1_000_000_000]


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


def test_used_receipt_walks_through_a_new_burn(tmp_path, monkeypatch):
    monkeypatch.setenv("GAS_HOME", str(tmp_path))
    model = tmp_path / "model.zip"
    model.write_bytes(b"zip")
    save_burn_receipt("5Miner", 34, BurnEvidence(tx_hash="a" * 64, block_number=1))
    calls = []

    def presign(*args, **kwargs):
        calls.append(kwargs)
        digest = kwargs.get("burn_tx_hash")
        if digest == "a" * 64:
            return {
                "success": False,
                "status_code": 409,
                "response": {"detail": "This burn has already been used"},
            }
        if digest == "b" * 64:
            return {
                "success": True,
                "status_code": 200,
                "response": {
                    "data": {
                        "model_id": 4,
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
        resubmit=lambda: BurnEvidence(tx_hash="b" * 64, block_number=2),
    )
    assert result["success"] is True
    assert result.get("already_uploaded") is None
    assert [c.get("burn_tx_hash") for c in calls] == [None, "a" * 64, "b" * 64]


def test_failed_r2_keeps_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("GAS_HOME", str(tmp_path))
    model = tmp_path / "model.zip"
    model.write_bytes(b"zip")
    save_burn_receipt("5Miner", 34, BurnEvidence(tx_hash="c" * 64, block_number=3))

    def presign(*args, **kwargs):
        if kwargs.get("burn_tx_hash") == "c" * 64:
            return {
                "success": True,
                "status_code": 200,
                "response": {
                    "data": {
                        "model_id": 5,
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
        lambda *args, **kwargs: {"success": False, "response": {"detail": "reset"}},
    )
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Miner"))
    result = upload_single_modality(
        wallet,
        str(model),
        "image",
        "https://upload.example/upload",
        resubmit=lambda: pytest.fail("should not burn again"),
    )
    assert result["success"] is False
    assert result["step"] == "r2_upload"
    assert load_burn_receipt("5Miner", 34) == BurnEvidence(
        tx_hash="c" * 64, block_number=3
    )
