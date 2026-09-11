"""Reclaim file cache after sequential reads on memory-capped hosts.

Linux keeps closed-file pages in RAM until the *host* is short on memory.
Inside a cgroup (Runpod, Docker) that cache is charged to the container
limit even when the host still has hundreds of GiB free, so the pod OOMs
while `free` looks fine.

Off by default (SN34_CACHE_RECLAIM=0). Set SN34_CACHE_RECLAIM=1 to enable.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

try:
    import bittensor as bt
except Exception:  # pragma: no cover
    bt = None

_POLL_SECONDS = 1.0
_POSIX_FADV_DONTNEED = 4
_MODEL_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".ggml",
    ".onnx",
    ".msgpack",
    ".h5",
)

_started = False
_lock = threading.Lock()


def _log(level: str, msg: str) -> None:
    logger = getattr(bt, "logging", None) if bt is not None else None
    if logger is not None and hasattr(logger, level):
        getattr(logger, level)(msg)


def cgroup_memory_limit_bytes() -> Optional[int]:
    """Return the cgroup memory cap, or None if unlimited / unknown."""
    v1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1.exists():
        try:
            limit = int(v1.read_text().strip())
        except (OSError, ValueError):
            return None
        # cgroup v1 uses a huge sentinel for "unlimited"
        if limit >= 2**60:
            return None
        return limit

    v2 = Path("/sys/fs/cgroup/memory.max")
    if v2.exists():
        raw = v2.read_text().strip()
        if raw == "max":
            return None
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def cache_reclaim_enabled() -> bool:
    """True only when SN34_CACHE_RECLAIM is explicitly enabled. Default off."""
    try:
        from dotenv import load_dotenv

        load_dotenv(".env.validator")
    except Exception:
        pass
    flag = os.environ.get("SN34_CACHE_RECLAIM", "0").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _advise_dontneed(path: str) -> bool:
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        return libc.posix_fadvise(fd, 0, 0, _POSIX_FADV_DONTNEED) == 0
    finally:
        os.close(fd)


def _is_model_weight(path: str) -> bool:
    lower = path.lower()
    return any(lower.endswith(suffix) for suffix in _MODEL_SUFFIXES)


def _watch_prefixes(cache_dir: Optional[str]) -> tuple[str, ...]:
    prefixes = []
    for raw in (
        cache_dir,
        os.environ.get("SN34_CACHE_DIR"),
        os.environ.get("HF_HOME"),
        os.environ.get("HUGGINGFACE_HOME"),
    ):
        if not raw:
            continue
        prefixes.append(str(Path(raw).expanduser()))
    seen = set()
    out = []
    for prefix in prefixes:
        if prefix not in seen:
            seen.add(prefix)
            out.append(prefix)
    return tuple(out)


def _open_watched_files(prefixes: Iterable[str]) -> set[str]:
    found: set[str] = set()
    fd_dir = "/proc/self/fd"
    try:
        names = os.listdir(fd_dir)
    except OSError:
        return found
    for name in names:
        try:
            target = os.readlink(os.path.join(fd_dir, name))
        except OSError:
            continue
        if any(target.startswith(prefix) for prefix in prefixes):
            found.add(target)
    return found


def _reclaim_loop(prefixes: tuple[str, ...]) -> None:
    prev: set[str] = set()
    while True:
        current = _open_watched_files(prefixes)
        for path in prev - current:
            _advise_dontneed(path)
        # Under pressure, also drop open media (not in-use model weights).
        limit = cgroup_memory_limit_bytes()
        if limit is not None:
            try:
                usage = int(
                    Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text()
                )
            except (OSError, ValueError):
                usage = 0
            if usage >= 0.7 * limit:
                for path in current:
                    if not _is_model_weight(path):
                        _advise_dontneed(path)
        prev = current
        time.sleep(_POLL_SECONDS)


def start_cache_reclaim(cache_dir: Optional[str] = None) -> bool:
    """Reclaim this process's file cache when SN34_CACHE_RECLAIM=1.

    No-op by default. Returns True if the reclaim thread is running.
    """
    global _started
    with _lock:
        if _started:
            return True
        if not cache_reclaim_enabled():
            _log("debug", "cache_reclaim idle (SN34_CACHE_RECLAIM=0)")
            return False
        prefixes = _watch_prefixes(cache_dir)
        if not prefixes:
            _log("warning", "cache_reclaim skipped: no cache directories")
            return False
        thread = threading.Thread(
            name="cache-reclaim",
            target=_reclaim_loop,
            args=(prefixes,),
            daemon=True,
        )
        thread.start()
        _started = True
        _log("info", "cache_reclaim on (SN34_CACHE_RECLAIM=1)")
        return True
