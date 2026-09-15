"""Durable local state and process/thread exclusion for irreversible burns.

Locks deliberately cover all endpoints for a hotkey/netuid. The journal also
binds to the chain genesis hash so changing endpoints cannot replay stale proof.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading


class ResubmitBurnError(RuntimeError):
    """The interactive burn could not be completed safely."""


_locks = {}
_guard = threading.Lock()
_held = threading.local()


def receipt_dir() -> Path:
    root = os.environ.get("GAS_HOME") or str(Path.home() / ".gas")
    return Path(root) / "resubmit_burns"


def state_path(hotkey: str, netuid: int, suffix: str) -> Path:
    # Wallet addresses are normally SS58; reject path traversal in callers too.
    if not hotkey or not hotkey.isalnum():
        raise ResubmitBurnError("Invalid hotkey for burn state")
    return receipt_dir() / f"{int(netuid)}-{hotkey}.{suffix}"


def read_state(path: Path):
    try:
        with path.open() as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("expected an object")
        return data
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ResubmitBurnError(
            f"Cannot read burn state at {path}; refusing another burn. "
            "Preserve this file and recover the existing transaction first."
        ) from exc


def _sync_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _ensure_directory(path: Path) -> None:
    if path.is_dir():
        return
    _ensure_directory(path.parent)
    path.mkdir(exist_ok=True, mode=0o700)
    _sync_directory(path.parent)


def atomic_write(path: Path, data: dict) -> None:
    """Private file, atomic replacement, and fsync of both file and directory."""
    _ensure_directory(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=".burn-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def burn_lock(hotkey: str, netuid: int):
    """Nonblocking, reentrant locally; OS lock survives until process exit.

    Keep the lock inode: unlinking a lock file would allow competing owners.
    Upload holds this through proof acceptance and receipt cleanup, not just
    through transaction submission.
    """
    path = state_path(hotkey, netuid, "lock").absolute()
    with _guard:
        lock = _locks.setdefault(str(path), threading.RLock())
    if not lock.acquire(blocking=False):
        raise ResubmitBurnError("Another push/burn is in progress for this hotkey")
    try:
        held = getattr(_held, "paths", set())
        if str(path) in held:
            yield
            return
        _ensure_directory(path.parent)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ResubmitBurnError(
                    "Another push/burn is in progress for this hotkey"
                ) from exc
            _held.paths = held | {str(path)}
            try:
                yield
            finally:
                _held.paths = held
        finally:
            os.close(fd)
    finally:
        lock.release()


def transaction_hash(encoded: str) -> str:
    if not isinstance(encoded, str) or not encoded.startswith("0x"):
        raise ValueError("missing encoded transaction")
    raw = bytes.fromhex(encoded[2:])
    if not raw:
        raise ValueError("empty encoded transaction")
    return hashlib.blake2b(raw, digest_size=32).hexdigest()


def load_journal(hotkey: str, netuid: int):
    path = state_path(hotkey, netuid, "pending.json")
    data = read_state(path)
    if data is None:
        return None
    try:
        if (
            data["version"] != 1
            or data["hotkey"] != hotkey
            or data["netuid"] != netuid
            or data["status"] not in {"pending", "confirmed", "used", "failed"}
            or transaction_hash(data["signed_extrinsic"]) != data["tx_hash"]
            or type(data["start_block"]) is not int
            or data["start_block"] < 0
            or data["period"] != 64
            or not isinstance(data["genesis_hash"], str)
            or not data["genesis_hash"]
        ):
            raise ValueError("invalid pending transaction")
        if data["status"] == "confirmed" and (
            type(data["block_number"]) is not int or data["block_number"] <= 0
        ):
            raise ValueError("invalid confirmation")
        return data
    except (KeyError, TypeError, ValueError) as exc:
        raise ResubmitBurnError(
            f"Invalid burn journal at {path}; refusing another burn. Preserve it for recovery."
        ) from exc


def save_journal(hotkey: str, netuid: int, data: dict) -> None:
    atomic_write(state_path(hotkey, netuid, "pending.json"), data)


def execution_success(events, index: int):
    """Require an explicit System event for this exact extrinsic index."""
    outcomes = set()
    for event in events:
        value = getattr(event, "value", event)
        if not isinstance(value, dict):
            continue
        found_index = value.get("extrinsic_idx")
        if found_index is None and isinstance(value.get("phase"), dict):
            found_index = value["phase"].get("ApplyExtrinsic")
        if isinstance(found_index, str) and found_index.isdecimal():
            found_index = int(found_index)
        if type(found_index) is not int or found_index != index:
            continue
        details = value.get("event")
        if not isinstance(details, dict):
            details = value
        module = details.get("event_module") or details.get("module_id")
        name = details.get("event_id") or details.get("event_name")
        if module == "System" and name in {"ExtrinsicSuccess", "ExtrinsicFailed"}:
            outcomes.add(name == "ExtrinsicSuccess")
    # Contradictory RPC events are unknown, not a confirmed failure that could
    # authorize another payment.
    return next(iter(outcomes)) if len(outcomes) == 1 else None
