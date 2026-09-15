"""Fault injection only: never connect to a chain or submit a real burn."""

import multiprocessing
import stat
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from gas.protocol import burn_journal as journal
from gas.protocol import resubmit_burn as burn
from gas.protocol.miner_requests import upload_single_modality


class FakeChain:
    def __init__(self):
        self.substrate = self
        self.head = 100
        self.genesis = "0x" + "11" * 32
        self.encoded = "0x01020304"
        self.signed = 0
        self.sent = 0
        self.blocks = {}
        self.events = {}
        self.error = None
        self.stake = 2_000_000_000
        self.calls = []
        self.before_submit = lambda: None

    def get_chain_finalised_head(self):
        return f"block-{self.head}"

    def get_block_number(self, block_hash):
        return int(block_hash.split("-")[1])

    def get_block_hash(self, number):
        return self.genesis if number == 0 else f"block-{number}"

    def get_subnet_price(self, netuid):
        return SimpleNamespace(tao=0.5)

    def get_stake(self, *args):
        return SimpleNamespace(rao=self.stake)

    def get_balance(self, *args):
        return SimpleNamespace(tao=10)

    def compose_call(self, module, function, params):
        self.calls.append((module, function, params))
        return self.calls[-1]

    def create_signed_extrinsic(self, **kwargs):
        self.signed += 1
        assert kwargs["era"] == {"period": 64, "current": self.head}
        return SimpleNamespace(data=self.encoded)

    def submit_extrinsic(self, extrinsic, **kwargs):
        self.before_submit()
        self.sent += 1
        assert kwargs == {"wait_for_inclusion": True, "wait_for_finalization": True}
        self.head += 1
        self.blocks[self.head] = [self.encoded]
        self.events[self.head] = [
            {"extrinsic_idx": 0, "event_module": "System", "event_id": "ExtrinsicSuccess"}
        ]
        if self.error:
            raise self.error
        return SimpleNamespace(
            finalized=True, is_success=True, extrinsic_hash="0x" + journal.transaction_hash(self.encoded),
            block_number=self.head,
        )

    def rpc_request(self, method, params):
        assert method == "chain_getBlock"
        number = self.get_block_number(params[0])
        return {"result": {"block": {"extrinsics": self.blocks.get(number, [])}}}

    def get_events(self, block_hash):
        return self.events.get(self.get_block_number(block_hash), [])

    def close(self):
        pass


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setenv("GAS_HOME", str(tmp_path))
    monkeypatch.setattr(burn.sys.stdin, "isatty", lambda: True)
    # Never permit tests to accidentally hit a live network.
    monkeypatch.setattr("bittensor.Subtensor", lambda **kw: pytest.fail("unexpected chain connection"))
    wallet = SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address="5Hot"),
        coldkeypub=SimpleNamespace(ss58_address="5Cold"), coldkey=object(),
    )
    return wallet, FakeChain()


def execute(context, **kwargs):
    wallet, chain = context
    return burn.execute_resubmit_burn(wallet, 34, subtensor=chain, confirm_fn=lambda _: "y", **kwargs)


@pytest.mark.parametrize("stake,kind,amount", [
    (2_000_000_000, "burn_alpha", 1_020_000_001),
    (0, "add_stake_burn", 500_000_000),
])
def test_persist_before_broadcast_and_show_actual_cost(context, stake, kind, amount, capsys):
    wallet, chain = context
    chain.stake = stake
    intent = {"file_hash": "ab" * 32, "modality": "image"}

    def before_submit():
        pending = journal.load_journal("5Hot", 34)
        assert pending["signed_extrinsic"] == chain.encoded
        assert pending["tx_hash"] == journal.transaction_hash(chain.encoded)
        assert pending["amount_rao"] == amount
        assert pending["status"] == "pending"
        assert pending["submission"] == intent
        assert pending["kind"] == kind

    chain.before_submit = before_submit
    result = execute(context, submission=intent)
    assert result == burn.load_burn_receipt("5Hot", 34)
    assert chain.sent == chain.signed == 1
    assert chain.calls[0][1] == kind
    assert chain.calls[0][2]["amount"] == amount
    text = capsys.readouterr().out
    assert "transaction fees are additional" in text
    if stake:
        assert "~0.510000 TAO" in text
        assert "2% price padding" in text
    assert stat.S_IMODE(journal.state_path("5Hot", 34, "pending.json").stat().st_mode) == 0o600


@pytest.mark.parametrize("failure", [OSError("disk full"), PermissionError("read only")])
def test_cannot_persist_never_broadcasts(context, monkeypatch, failure):
    def fail(*args):
        raise failure
    monkeypatch.setattr(burn, "save_journal", fail)
    with pytest.raises(burn.ResubmitBurnError, match="no transaction was sent"):
        execute(context)
    assert context[1].sent == 0


@pytest.mark.parametrize("failure", [TimeoutError("RPC timeout"), KeyboardInterrupt()])
def test_success_then_timeout_or_process_interruption_recovers_once(context, failure):
    wallet, chain = context
    chain.error = failure
    with pytest.raises((burn.ResubmitBurnError, KeyboardInterrupt)):
        execute(context)
    assert journal.load_journal("5Hot", 34)["status"] == "pending"
    chain.error = None
    recovered = burn.execute_resubmit_burn(
        wallet, 34, subtensor=chain,
        confirm_fn=lambda _: pytest.fail("must not ask for a fresh burn"),
    )
    assert recovered.block_number == 101
    assert recovered.tx_hash == journal.transaction_hash(chain.encoded)
    assert chain.sent == chain.signed == 1


def test_crash_before_broadcast_blocks_new_burn(context):
    _, chain = context
    def crash():
        raise KeyboardInterrupt()
    chain.before_submit = crash
    with pytest.raises(KeyboardInterrupt):
        execute(context)
    chain.head = 1000  # Even expired/missing transactions do not auto-trigger payment.
    with pytest.raises(burn.ResubmitBurnError, match="unresolved"):
        execute(context)
    assert chain.sent == 0
    assert chain.signed == 1


def test_receipt_save_failure_recovers_from_confirmed_journal(context, monkeypatch):
    original = burn.save_burn_receipt
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(burn, "save_burn_receipt", fail)
    with pytest.raises(burn.ResubmitBurnError, match="may have executed"):
        execute(context)
    monkeypatch.setattr(burn, "save_burn_receipt", original)
    execute(context)
    assert context[1].sent == context[1].signed == 1


@pytest.mark.parametrize("events", [
    [],
    [{"extrinsic_idx": 1, "event_module": "System", "event_id": "ExtrinsicSuccess"}],
    [{"extrinsic_idx": 0, "event_module": "Other", "event_id": "ExtrinsicSuccess"}],
    [
        {"extrinsic_idx": 0, "event_module": "System", "event_id": "ExtrinsicSuccess"},
        {"extrinsic_idx": 0, "event_module": "System", "event_id": "ExtrinsicFailed"},
    ],
])
def test_missing_success_fails_closed(context, events):
    _, chain = context
    chain.error = TimeoutError()
    with pytest.raises(burn.ResubmitBurnError):
        execute(context)
    chain.events[101] = events
    with pytest.raises(burn.ResubmitBurnError, match="Cannot reconcile"):
        execute(context)
    assert chain.sent == chain.signed == 1


def test_explicit_finalized_failure_only_allows_next_authorized_attempt(context):
    _, chain = context
    chain.error = TimeoutError()
    with pytest.raises(burn.ResubmitBurnError):
        execute(context)
    chain.events[101] = [{"extrinsic_idx": 0, "event_module": "System", "event_id": "ExtrinsicFailed"}]
    with pytest.raises(burn.ResubmitBurnError, match="failed on-chain"):
        execute(context)
    assert chain.sent == 1
    chain.error = None
    chain.encoded = "0x05060708"
    execute(context)
    assert chain.sent == 2


def test_rpc_failure_cannot_allow_new_burn(context, monkeypatch):
    _, chain = context
    chain.error = TimeoutError()
    with pytest.raises(burn.ResubmitBurnError):
        execute(context)
    monkeypatch.setattr(chain, "rpc_request", lambda *a: {"error": "unavailable"})
    with pytest.raises(burn.ResubmitBurnError, match="Cannot reconcile"):
        execute(context)
    assert chain.sent == 1


def test_different_chain_cannot_reuse_or_replace_burn(context):
    execute(context)
    context[1].genesis = "0x" + "22" * 32
    with pytest.raises(burn.ResubmitBurnError, match="different chain"):
        execute(context)
    assert context[1].sent == 1


@pytest.mark.parametrize("suffix", ["pending.json", "json"])
def test_corrupt_state_fails_closed(context, suffix):
    path = journal.state_path("5Hot", 34, suffix)
    path.parent.mkdir(parents=True)
    path.write_text("broken json")
    with pytest.raises(burn.ResubmitBurnError, match="refusing another burn"):
        execute(context)
    assert context[1].sent == context[1].signed == 0


def test_unknown_upload_state_stops_before_api(context, tmp_path, monkeypatch):
    wallet, chain = context
    def crash():
        raise KeyboardInterrupt()
    chain.before_submit = crash
    with pytest.raises(KeyboardInterrupt):
        execute(context)
    monkeypatch.setattr("bittensor.Subtensor", lambda **kw: chain)
    monkeypatch.setattr("gas.protocol.miner_requests.generate_presigned_url", lambda *a, **kw: pytest.fail("must recover first"))
    model = tmp_path / "model.zip"
    model.write_bytes(b"model")
    with pytest.raises(burn.ResubmitBurnError, match="unresolved"):
        upload_single_modality(wallet, str(model), "image", "https://upload.example")
    assert chain.sent == 0


def test_api_consumption_is_durable_and_next_submission_can_burn(context):
    execute(context)
    burn.clear_burn_receipt("5Hot", 34)
    assert journal.load_journal("5Hot", 34)["status"] == "used"
    context[1].encoded = "0x05060708"
    execute(context)
    assert context[1].sent == 2


def test_thread_lock_is_reentrant_but_excludes_competing_push(context):
    def compete():
        with pytest.raises(burn.ResubmitBurnError, match="in progress"):
            with journal.burn_lock("5Hot", 34):
                pytest.fail("must not enter")
    with journal.burn_lock("5Hot", 34):
        with journal.burn_lock("5Hot", 34):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(compete).result()
    with journal.burn_lock("5Hot", 34):
        pass


def _hold_process_lock(ready, release):
    with journal.burn_lock("5Hot", 34):
        ready.set()
        release.wait(10)


def test_process_lock_excludes_another_cli(context):
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    process = ctx.Process(target=_hold_process_lock, args=(ready, release))
    process.start()
    try:
        assert ready.wait(10)
        with pytest.raises(burn.ResubmitBurnError, match="in progress"):
            with journal.burn_lock("5Hot", 34):
                pytest.fail("must not enter")
    finally:
        release.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0


@pytest.mark.parametrize("price", [float("nan"), float("inf"), -1.0, 0.0])
def test_nonfinite_price_is_rejected(context, price):
    context[1].get_subnet_price = lambda _: SimpleNamespace(tao=price)
    with pytest.raises(burn.ResubmitBurnError):
        execute(context)
    assert context[1].sent == 0


def test_execution_event_shapes_and_contradictions_are_unknown():
    assert journal.execution_success([
        {"phase": {"ApplyExtrinsic": 2}, "event": {"module_id": "System", "event_id": "ExtrinsicSuccess"}}
    ], 2) is True
    assert journal.execution_success([
        {"extrinsic_idx": 2, "event_module": "System", "event_id": "ExtrinsicFailed"},
        {"extrinsic_idx": "2", "event_module": "System", "event_id": "ExtrinsicSuccess"},
    ], 2) is None


def test_hash_matches_real_sdk_scale_serialization():
    from scalecodec.base import ScaleBytes
    from scalecodec.types import GenericExtrinsic

    extrinsic = GenericExtrinsic(data=ScaleBytes("0x01020304"))
    assert journal.transaction_hash(str(extrinsic.data)) == extrinsic.extrinsic_hash.hex()


def test_fsync_failure_before_broadcast(context, monkeypatch):
    # Create the lock directory first so the injected failure hits the journal.
    with journal.burn_lock("5Hot", 34):
        pass
    def fail(*args):
        raise OSError("fsync failed")
    monkeypatch.setattr(journal.os, "fsync", fail)
    with pytest.raises(burn.ResubmitBurnError, match="no transaction was sent"):
        execute(context)
    assert context[1].sent == 0


def test_crash_after_api_ack_before_receipt_removal(context, monkeypatch):
    execute(context)
    from pathlib import Path
    original = Path.unlink
    def fail(self, *args, **kwargs):
        raise OSError("unlink failed")
    monkeypatch.setattr(Path, "unlink", fail)
    with pytest.raises(OSError):
        burn.clear_burn_receipt("5Hot", 34)
    assert journal.load_journal("5Hot", 34)["status"] == "used"
    monkeypatch.setattr(Path, "unlink", original)
    context[1].encoded = "0x05060708"
    execute(context)
    assert context[1].sent == 2


@pytest.mark.parametrize("field,value", [("finalized", False), ("is_success", None), ("extrinsic_hash", "0x" + "ff" * 32), ("block_number", 0)])
def test_ambiguous_receipt_keeps_pending_journal(context, field, value):
    chain = context[1]
    original = chain.submit_extrinsic
    def submit(*args, **kwargs):
        receipt = original(*args, **kwargs)
        setattr(receipt, field, value)
        return receipt
    chain.submit_extrinsic = submit
    with pytest.raises(burn.ResubmitBurnError, match="may have executed"):
        execute(context)
    assert journal.load_journal("5Hot", 34)["status"] == "pending"
    execute(context)
    assert chain.sent == 1


def test_full_upload_retry_reuses_durable_burn(context, tmp_path, monkeypatch):
    wallet, chain = context
    monkeypatch.setattr("bittensor.Subtensor", lambda **kw: chain)
    model = tmp_path / "model.zip"
    model.write_bytes(b"model")
    limit = {"success": False, "status_code": 403, "response": {"detail": "already used its free submission"}}
    responses = iter([
        limit,
        {"success": False, "status_code": 503, "response": {"detail": "temporary error"}},
        limit,
        {"success": True, "response": {"data": {"model_id": 1, "presigned_url": "https://r2.example", "r2_key": "model"}}},
    ])
    proofs = []
    def presign(*args, **kwargs):
        proofs.append(kwargs.get("burn_tx_hash"))
        return next(responses)
    monkeypatch.setattr("gas.protocol.miner_requests.generate_presigned_url", presign)
    monkeypatch.setattr("gas.protocol.miner_requests.upload_to_r2", lambda *a: {"success": True})
    monkeypatch.setattr("gas.protocol.miner_requests.confirm_upload", lambda *a: {"success": True})
    first = upload_single_modality(
        wallet, str(model), "image", "https://upload.example",
        resubmit=lambda: execute(context),
    )
    assert first["success"] is False
    second = upload_single_modality(
        wallet, str(model), "image", "https://upload.example",
        resubmit=lambda: pytest.fail("must reuse the first burn"),
    )
    assert second["success"] is True
    digest = journal.transaction_hash(chain.encoded)
    assert proofs == [None, digest, None, digest]
    assert chain.sent == chain.signed == 1
    assert journal.load_journal("5Hot", 34)["status"] == "used"
    assert burn.load_burn_receipt("5Hot", 34) is None
