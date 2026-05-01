"""
Lightweight durable task records for generative miner restarts.

Writes one JSON file per task under MINER_STATE_DIR (default: <output_dir>/.gen_miner_state).
Preserves pending queue + processing tasks that have an external job checkpoint (Runway, Sora).

Large binary fields are never persisted (validators do not need them in the snapshot).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import bittensor as bt

from .task_manager import GenerationTask, TaskStatus

RECORD_VERSION = 1
STATE_SUBDIR = ".gen_miner_state"


def resolve_state_dir(output_dir: Path) -> Path:
    env = os.getenv("MINER_STATE_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path(output_dir) / STATE_SUBDIR


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def serialize_task(task: GenerationTask) -> Dict[str, Any]:
    from dataclasses import asdict

    d = asdict(task)
    d["status"] = task.status.value
    d["input_data"] = None
    d["result_data"] = None
    if getattr(task, "checkpoint", None) is None:
        d["checkpoint"] = None
    return d


def deserialize_task(data: Dict[str, Any]) -> GenerationTask:
    d = dict(data)
    d["status"] = TaskStatus(d["status"])
    d["input_data"] = None
    d["result_data"] = None
    if d.get("checkpoint") is None:
        d["checkpoint"] = None
    fields = set(GenerationTask.__dataclass_fields__.keys())
    return GenerationTask(**{k: d[k] for k in fields if k in d})


class TaskCheckpointStore:
    """Filesystem-backed task snapshots for crash/restart recovery."""

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)

    def path_for(self, task_id: str) -> Path:
        return self.state_dir / f"{task_id}.json"

    def save(self, task: GenerationTask) -> None:
        payload = {
            "version": RECORD_VERSION,
            "task": serialize_task(task),
        }
        raw = json.dumps(payload, indent=2).encode("utf-8")
        path = self.path_for(task.task_id)
        _atomic_write(path, raw)
        bt.logging.debug(f"CheckpointStore saved task {task.task_id} ({task.status.value})")

    def delete(self, task_id: str) -> None:
        path = self.path_for(task_id)
        try:
            path.unlink(missing_ok=True)
        except TypeError:
            if path.exists():
                path.unlink()

    def load_all(self) -> List[GenerationTask]:
        if not self.state_dir.is_dir():
            return []
        out: List[GenerationTask] = []
        for p in sorted(self.state_dir.glob("*.json")):
            try:
                raw = p.read_text(encoding="utf-8")
                payload = json.loads(raw)
                if payload.get("version") != RECORD_VERSION:
                    bt.logging.warning(f"Skipping unknown checkpoint version in {p.name}")
                    continue
                out.append(deserialize_task(payload["task"]))
            except Exception as e:
                bt.logging.warning(f"Could not load checkpoint {p}: {e}")
        return out