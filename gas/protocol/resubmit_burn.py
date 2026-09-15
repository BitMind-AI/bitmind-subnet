"""Walk a miner through burning 0.5 TAO of SN34 alpha so they can push again."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

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


class ResubmitBurnError(RuntimeError):
    """The interactive burn could not be completed."""


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


def receipt_dir() -> Path:
    root = os.environ.get("GAS_HOME") or str(Path.home() / ".gas")
    return Path(root) / "resubmit_burns"


def receipt_path(hotkey: str, netuid: int) -> Path:
    return receipt_dir() / f"{int(netuid)}-{hotkey}.json"


def load_burn_receipt(hotkey: str, netuid: int) -> Optional[BurnEvidence]:
    path = receipt_path(hotkey, netuid)
    try:
        data = json.loads(path.read_text())
        return BurnEvidence(
            tx_hash=normalize_tx_hash(data.get("tx_hash")),
            block_number=int(data["block_number"]),
        )
    except (OSError, KeyError, TypeError, ValueError, ResubmitBurnError):
        return None


def save_burn_receipt(hotkey: str, netuid: int, evidence: BurnEvidence) -> None:
    path = receipt_path(hotkey, netuid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"tx_hash": evidence.tx_hash, "block_number": evidence.block_number}
        )
    )


def clear_burn_receipt(hotkey: str, netuid: int) -> None:
    try:
        receipt_path(hotkey, netuid).unlink()
    except FileNotFoundError:
        pass


def alpha_rao_for_fee(tao_per_alpha: float) -> int:
    if tao_per_alpha <= 0:
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
    if len(digest) != 64:
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
) -> BurnEvidence:
    import bittensor as bt

    network = chain_endpoint or ("finney" if netuid == 34 else "test")
    sub = subtensor or bt.Subtensor(network=network)
    price = sub.get_subnet_price(netuid)
    tao_per_alpha = float(getattr(price, "tao", 0) or 0)
    amount_rao = alpha_rao_for_fee(tao_per_alpha)
    amount_alpha = amount_rao / RAO
    hotkey = wallet.hotkey.ss58_address
    coldkey = wallet.coldkeypub.ss58_address
    stake = sub.get_stake(coldkey, hotkey, netuid)
    stake_rao = int(getattr(stake, "rao", 0) or 0)
    print(
        f"  Resubmit fee: {RESUBMIT_FEE_TAO:g} TAO of SN34 alpha "
        f"(~{amount_alpha:.4f} α at the current pool price)."
    )
    print(f"  Alpha staked to this hotkey: {stake_rao / RAO:.4f} α")
    if stake_rao >= amount_rao:
        if not _confirm(
            "  Burn that alpha now? This cannot be undone. [y/N] ",
            confirm_fn,
        ):
            raise ResubmitBurnError(
                "Cannot submit another model without burning 0.5 TAO of SN34 alpha."
            )
        print("  Unlock the coldkey if prompted, then wait for the burn to land...")
        call = _compose_burn(sub, hotkey, netuid, amount_rao)
        burned = sub.sign_and_send_extrinsic(
            call,
            wallet,
            sign_with="coldkey",
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
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
        if not _confirm(
            "  Spend 0.5 TAO via add_stake_burn now? This cannot be undone. [y/N] ",
            confirm_fn,
        ):
            raise ResubmitBurnError(
                "Cannot submit another model without burning 0.5 TAO of SN34 alpha."
            )
        print("  Unlock the coldkey if prompted, then wait for add_stake_burn to land...")
        burned = sub.add_stake_burn(
            wallet,
            netuid,
            hotkey,
            bt.Balance.from_tao(RESUBMIT_FEE_TAO),
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
    evidence = evidence_from_response(burned, sub)
    try:
        save_burn_receipt(hotkey, netuid, evidence)
    except OSError:
        pass
    print(f"  Burn included in block {evidence.block_number} ({evidence.tx_hash[:12]}…).")
    return evidence


def offer_resubmit_burn(
    wallet,
    netuid: int,
    chain_endpoint: Optional[str] = None,
    *,
    confirm_fn: Callable[[str], str] = input,
    execute_fn=execute_resubmit_burn,
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
    )
