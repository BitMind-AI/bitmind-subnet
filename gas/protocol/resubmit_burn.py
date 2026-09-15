"""Walk a miner through burning 0.5 TAO of SN34 alpha so they can push again."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from gas.protocol.burn_journal import (
    ResubmitBurnError, atomic_write, burn_lock, execution_success,
    load_journal, read_state, save_journal, state_path, transaction_hash,
)

RESUBMIT_FEE_TAO = 0.5
PRICE_SLACK = 0.02
RAO = 1_000_000_000
BURN_CALLS = (
    ("burn_alpha", {"amount": "amount"}),
    ("burn_alpha", {"amount": "alpha_amount"}),
)


@dataclass(frozen=True)
class BurnEvidence:
    tx_hash: str
    block_number: int


def _detail_text(result: dict) -> str:
    response = result.get("response") or {}
    detail = response.get("detail") or response.get("error") or response.get("message") or ""
    if isinstance(detail, list):
        detail = " ".join(str(item) for item in detail)
    return str(detail)


def is_submission_limit(result: dict) -> bool:
    if result.get("status_code") != 403:
        return False
    detail = _detail_text(result).lower()
    return any(
        token in detail
        for token in (
            "free submission",
            "one submission",
            "burn_tx_hash",
            "0.5 tao",
            "already used",
        )
    )


def is_credit_used(result: dict) -> bool:
    if result.get("status_code") != 409:
        return False
    return "already been used" in _detail_text(result).lower()


def is_duplicate_upload(result: dict) -> bool:
    """Only the API's explicit duplicate-file response permits skipping upload."""
    return (
        result.get("status_code") == 409
        and _detail_text(result).strip().lower() == "file with this hash already exists"
    )


def receipt_path(hotkey: str, netuid: int) -> Path:
    return state_path(hotkey, netuid, "json")


def load_burn_receipt(hotkey: str, netuid: int) -> Optional[BurnEvidence]:
    data = read_state(receipt_path(hotkey, netuid))
    if data is None:
        return None
    try:
        if type(data["block_number"]) is not int or data["block_number"] <= 0:
            raise ValueError("invalid block number")
        return BurnEvidence(
            tx_hash=normalize_tx_hash(data.get("tx_hash")),
            block_number=int(data["block_number"]),
        )
    except (KeyError, TypeError, ValueError, ResubmitBurnError) as exc:
        raise ResubmitBurnError("Invalid saved burn receipt; refusing another burn") from exc


def save_burn_receipt(hotkey: str, netuid: int, evidence: BurnEvidence) -> None:
    path = receipt_path(hotkey, netuid)
    atomic_write(path, {"tx_hash": evidence.tx_hash, "block_number": evidence.block_number})


def clear_burn_receipt(hotkey: str, netuid: int) -> None:
    journal = load_journal(hotkey, netuid)
    evidence = load_burn_receipt(hotkey, netuid)
    if journal and journal["status"] == "confirmed" and evidence:
        if journal["tx_hash"] == evidence.tx_hash:
            # Persist API acknowledgement before removing the legacy receipt.
            save_journal(hotkey, netuid, {**journal, "status": "used"})
    try:
        receipt_path(hotkey, netuid).unlink()
    except FileNotFoundError:
        pass


def min_alpha_rao_for_fee(tao_per_alpha: float) -> int:
    if not math.isfinite(tao_per_alpha) or tao_per_alpha <= 0:
        raise ResubmitBurnError("Could not read the SN34 alpha price")
    return int((RESUBMIT_FEE_TAO / tao_per_alpha) * RAO)


def alpha_rao_for_fee(tao_per_alpha: float) -> int:
    if not math.isfinite(tao_per_alpha) or tao_per_alpha <= 0:
        raise ResubmitBurnError("Could not read the SN34 alpha price")
    return int((RESUBMIT_FEE_TAO / tao_per_alpha) * (1 + PRICE_SLACK) * RAO) + 1


def normalize_tx_hash(value: Any) -> str:
    if value is None:
        raise ResubmitBurnError("Burn succeeded but the extrinsic hash was missing")
    if hasattr(value, "hex") and not isinstance(value, str):
        digest = value.hex()
    else:
        digest = str(value).strip()
    if digest.startswith("0x") or digest.startswith("0X"):
        digest = digest[2:]
    digest = digest.lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ResubmitBurnError("Burn succeeded but the extrinsic hash was invalid")
    return digest


def evidence_from_response(response: Any, subtensor: Any = None) -> BurnEvidence:
    if response is None or not getattr(response, "success", False):
        message = getattr(response, "message", None) or getattr(response, "error", None)
        raise ResubmitBurnError(f"Burn failed: {message or 'unknown error'}")
    receipt = getattr(response, "extrinsic_receipt", None)
    if receipt is None:
        raise ResubmitBurnError("Burn succeeded but the chain receipt was missing")
    tx_hash = normalize_tx_hash(
        getattr(receipt, "extrinsic_hash", None) or getattr(receipt, "extrinsic_idx", None)
    )
    block_number = getattr(receipt, "block_number", None)
    if block_number is None and subtensor is not None:
        block_hash = getattr(receipt, "block_hash", None)
        getter = getattr(getattr(subtensor, "substrate", None), "get_block_number", None)
        if block_hash is not None and callable(getter):
            block_number = getter(block_hash)
    if type(block_number) is not int or block_number <= 0:
        raise ResubmitBurnError("Burn succeeded but the block number was missing")
    return BurnEvidence(tx_hash=tx_hash, block_number=block_number)


def _confirm(prompt: str, confirm_fn: Callable[[str], str]) -> bool:
    if not sys.stdin.isatty():
        return False
    answer = confirm_fn(prompt).strip().lower()
    return answer in {"y", "yes"}


def _compose_burn(subtensor, hotkey: str, netuid: int, amount_rao: int):
    last_error = None
    for function, amount_key in BURN_CALLS:
        params = {"hotkey": hotkey, "netuid": netuid, amount_key["amount"]: amount_rao}
        try:
            return subtensor.compose_call(
                "SubtensorModule",
                function,
                params,
            )
        except Exception as exc:
            last_error = exc
            continue
    raise ResubmitBurnError(
        "This chain endpoint does not expose burn_alpha. "
        f"Cannot complete the resubmit burn ({last_error})."
    )


def execute_resubmit_burn(
    wallet,
    netuid: int,
    chain_endpoint: Optional[str] = None,
    *,
    subtensor=None,
    confirm_fn: Callable[[str], str] = input,
    submission: Optional[dict] = None,
) -> BurnEvidence:
    with burn_lock(wallet.hotkey.ss58_address, netuid):
        import bittensor as bt

        sub = subtensor if subtensor is not None else bt.Subtensor(
            network=chain_endpoint or ("finney" if netuid == 34 else "test")
        )
        try:
            return _execute_resubmit_burn(
                wallet, netuid, subtensor=sub,
                confirm_fn=confirm_fn, submission=submission,
            )
        finally:
            if subtensor is None:
                sub.close()


def recover_saved_burn(wallet, netuid: int, chain_endpoint=None):
    """Preflight uploads too: a server-side credit must not erase pending state."""
    journal = load_journal(wallet.hotkey.ss58_address, netuid)
    if journal is None:
        return
    import bittensor as bt

    sub = bt.Subtensor(network=chain_endpoint or ("finney" if netuid == 34 else "test"))
    try:
        _recover_burn(sub, wallet.hotkey.ss58_address, netuid, wallet.coldkeypub.ss58_address)
    finally:
        sub.close()


def _recover_burn(sub, hotkey: str, netuid: int, coldkey: str):
    journal = load_journal(hotkey, netuid)
    if journal is None:
        return load_burn_receipt(hotkey, netuid)
    if (
        journal["genesis_hash"] != sub.substrate.get_block_hash(0)
        or journal.get("coldkey") != coldkey
    ):
        raise ResubmitBurnError(
            "Saved burn belongs to a different chain or coldkey; refusing another burn"
        )
    if journal["status"] in {"used", "failed"}:
        # A crash during cleanup can leave the old receipt behind.
        clear_burn_receipt(hotkey, netuid)
        return None
    if journal["status"] == "pending":
        print(f"  Recovering pending burn {journal['tx_hash']} from finalized chain history...")
        substrate = sub.substrate
        try:
            finalized = substrate.get_block_number(substrate.get_chain_finalised_head())
            if type(finalized) is not int or finalized < journal["start_block"]:
                raise ValueError("finalized head unavailable")
            # Mortal transactions are signed with an explicit 64-block era.
            # Never infer failure from missing history, even after expiry.
            end = min(finalized, journal["start_block"] + journal["period"])
            for number in range(journal["start_block"], end + 1):
                block_hash = substrate.get_block_hash(number)
                if not block_hash:
                    raise ValueError("block hash unavailable")
                reply = substrate.rpc_request("chain_getBlock", [block_hash])
                extrinsics = reply["result"]["block"]["extrinsics"]
                if not isinstance(extrinsics, list):
                    raise ValueError("block contents unavailable")
                for index, encoded in enumerate(extrinsics):
                    if transaction_hash(encoded) != journal["tx_hash"]:
                        continue
                    success = execution_success(substrate.get_events(block_hash), index)
                    if success is False:
                        save_journal(hotkey, netuid, {**journal, "status": "failed"})
                        raise ResubmitBurnError(
                            "Previous burn failed on-chain (transaction fees may apply). "
                            "No new burn was sent. Run again to explicitly authorize a new attempt."
                        )
                    if success is not True:
                        raise ValueError("execution success unavailable")
                    journal = {**journal, "status": "confirmed", "block_number": number}
                    save_journal(hotkey, netuid, journal)
                    break
                if journal["status"] == "confirmed":
                    break
        except ResubmitBurnError:
            raise
        except Exception as exc:
            raise ResubmitBurnError(
                f"Cannot reconcile pending burn {journal['tx_hash']}; no new burn will be sent. "
                "Keep the burn journal and retry recovery when the chain RPC is available."
            ) from exc
        if journal["status"] != "confirmed":
            raise ResubmitBurnError(
                f"Burn {journal['tx_hash']} is unresolved; no new burn will be sent. "
                "Retry later; if it remains unresolved, investigate using the saved journal. "
                "Do not delete the journal or switch machines to bypass recovery."
            )
    evidence = BurnEvidence(journal["tx_hash"], journal["block_number"])
    save_burn_receipt(hotkey, netuid, evidence)
    return evidence


def _submit_durable_burn(sub, wallet, netuid: int, call, amount_rao: int, kind: str, submission):
    hotkey = wallet.hotkey.ss58_address
    substrate = sub.substrate
    start = substrate.get_block_number(substrate.get_chain_finalised_head())
    genesis = substrate.get_block_hash(0)
    if type(start) is not int or start < 0 or not genesis:
        raise ResubmitBurnError("Cannot read chain identity/finalized head; no burn was sent")
    # Creating/signing does not broadcast. Use a mortal, direct extrinsic for
    # both burn calls (not an SDK helper that signs and sends in one step).
    extrinsic = substrate.create_signed_extrinsic(
        call=call, keypair=wallet.coldkey, era={"period": 64, "current": start},
    )
    encoded = str(extrinsic.data)
    digest = transaction_hash(encoded)
    journal = {
        "version": 1, "status": "pending", "hotkey": hotkey, "netuid": netuid,
        "coldkey": wallet.coldkeypub.ss58_address, "genesis_hash": genesis,
        "start_block": start, "period": 64, "signed_extrinsic": encoded,
        "tx_hash": digest, "kind": kind, "amount_rao": amount_rao,
        "submission": submission,
    }
    try:
        save_journal(hotkey, netuid, journal)
    except OSError as exc:
        raise ResubmitBurnError("Cannot persist pending burn; no transaction was sent") from exc
    try:
        receipt = substrate.submit_extrinsic(
            extrinsic, wait_for_inclusion=True, wait_for_finalization=True,
        )
        if receipt.finalized is not True or receipt.is_success is not True:
            raise ValueError("success not confirmed")
        if normalize_tx_hash(receipt.extrinsic_hash) != digest:
            raise ValueError("receipt transaction mismatch")
        number = getattr(receipt, "block_number", None)
        if number is None:
            number = substrate.get_block_number(receipt.block_hash)
        if type(number) is not int or number <= 0:
            raise ValueError("missing block number")
        save_journal(hotkey, netuid, {**journal, "status": "confirmed", "block_number": number})
        evidence = BurnEvidence(digest, number)
        save_burn_receipt(hotkey, netuid, evidence)
    except Exception as exc:
        raise ResubmitBurnError(
            f"Burn {digest} may have executed; its journal is saved. "
            "Run again to recover this transaction, not to burn again."
        ) from exc
    print(f"  Burn included in block {evidence.block_number} ({evidence.tx_hash[:12]}…).")
    return evidence


def _execute_resubmit_burn(
    wallet, netuid, *, subtensor, confirm_fn, submission,
) -> BurnEvidence:
    sub = subtensor
    hotkey = wallet.hotkey.ss58_address
    coldkey = wallet.coldkeypub.ss58_address
    recovered = _recover_burn(sub, hotkey, netuid, coldkey)
    if recovered is not None:
        print(f"  Reusing existing burn {recovered.tx_hash[:12]}…; no new burn sent.")
        return recovered
    price = sub.get_subnet_price(netuid)
    tao_per_alpha = float(getattr(price, "tao", 0) or 0)
    min_rao = min_alpha_rao_for_fee(tao_per_alpha)
    amount_rao = alpha_rao_for_fee(tao_per_alpha)
    stake = sub.get_stake(coldkey, hotkey, netuid)
    stake_rao = int(getattr(stake, "rao", 0) or 0)
    print(f"  Resubmit fee: {RESUBMIT_FEE_TAO:g} TAO worth of subnet {netuid} alpha.")
    print(f"  Alpha staked to this hotkey: {stake_rao / RAO:.4f} α")
    if stake_rao >= min_rao:
        amount_rao = min(stake_rao, amount_rao)
        amount_alpha = amount_rao / RAO
        print(
            f"  Actual burn: {amount_rao} alpha rao ({amount_alpha:.9f} α), "
            f"worth ~{amount_alpha * tao_per_alpha:.6f} TAO at the quoted price. "
            "Includes up to 2% price padding (plus rounding); transaction fees are additional."
        )
        if not _confirm(
            "  Burn that alpha now? This cannot be undone. [y/N] ",
            confirm_fn,
        ):
            raise ResubmitBurnError(
                "Cannot submit another model without burning 0.5 TAO of SN34 alpha."
            )
        print("  Unlock the coldkey if prompted, then wait for the burn to land...")
        call = _compose_burn(sub, hotkey, netuid, amount_rao)
        kind = "burn_alpha"
    else:
        tao_free = sub.get_balance(coldkey)
        tao_available = float(getattr(tao_free, "tao", 0) or 0)
        print(
            f"  Not enough alpha. Free TAO on this coldkey: {tao_available:.4f}. "
            "We can spend 0.5 TAO in one add_stake_burn (buy α and burn it)."
        )
        if tao_available < RESUBMIT_FEE_TAO:
            raise ResubmitBurnError(
                "Need 0.5 TAO of SN34 alpha, or 0.5 free TAO for add_stake_burn."
            )
        print("  Actual spend: 0.5 TAO via add_stake_burn; transaction fees are additional.")
        if not _confirm(
            "  Spend 0.5 TAO via add_stake_burn now? This cannot be undone. [y/N] ",
            confirm_fn,
        ):
            raise ResubmitBurnError(
                "Cannot submit another model without burning 0.5 TAO of SN34 alpha."
            )
        print("  Unlock the coldkey if prompted, then wait for add_stake_burn to land...")
        amount_rao = int(RESUBMIT_FEE_TAO * RAO)
        kind = "add_stake_burn"
        call = sub.compose_call(
            "SubtensorModule", kind,
            {"hotkey": hotkey, "netuid": netuid, "amount": amount_rao, "limit": None},
        )
    return _submit_durable_burn(sub, wallet, netuid, call, amount_rao, kind, submission)


def offer_resubmit_burn(
    wallet,
    netuid: int,
    chain_endpoint: Optional[str] = None,
    *,
    confirm_fn: Callable[[str], str] = input,
    execute_fn=execute_resubmit_burn,
    submission: Optional[dict] = None,
) -> BurnEvidence:
    print()
    print("  This hotkey already used its free submission.")
    print(
        f"  Another model on this key requires burning "
        f"{RESUBMIT_FEE_TAO:g} TAO of SN34 alpha."
    )
    print("  Recycle does not count. The CLI will submit the burn, then retry the upload.")
    if not sys.stdin.isatty():
        raise ResubmitBurnError(
            "A second submission needs an interactive 0.5 TAO SN34 alpha burn. "
            "Re-run `gascli d push` in a terminal."
        )
    return execute_fn(
        wallet,
        netuid,
        chain_endpoint,
        confirm_fn=confirm_fn,
        submission=submission,
    )
