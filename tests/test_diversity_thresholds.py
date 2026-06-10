"""Threshold canary tests for the prompt diversity report (Phase 4).

Marked slow-adjacent but actually fast: builds tiny fixture DBs in tmp_path.
Verifies the --check mode catches monoculture and passes diverse data.
"""

import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "prompt_diversity_report.py"


def _make_db(path: Path, prompts: list[str]) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE prompts (
            id TEXT PRIMARY KEY, content TEXT NOT NULL,
            content_type TEXT NOT NULL, modality TEXT,
            created_at REAL NOT NULL, used_count INTEGER DEFAULT 0,
            last_used REAL, source_media_id TEXT)"""
    )
    now = time.time()
    for i, p in enumerate(prompts):
        conn.execute(
            "INSERT INTO prompts VALUES (?,?,?,?,?,0,NULL,NULL)",
            (str(uuid.uuid4()), p, "prompt", "video", now - i),
        )
    conn.commit()
    conn.close()


def _run_check(db: Path) -> int:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(db), "--check"],
        capture_output=True,
        text=True,
    )
    return proc.returncode


def test_monoculture_fails_check(tmp_path):
    base = (
        "A tight close-up frames the subject as if holding a breath, "
        "monochrome light over charcoal shadows while the camera stays "
        "locked and nothing moves but a faint tremor of stillness"
    )
    prompts = [base + f" take {i}." for i in range(40)]
    db = tmp_path / "mono.db"
    _make_db(db, prompts)
    assert _run_check(db) == 1


def test_diverse_passes_check(tmp_path):
    import random

    rng = random.Random(7)
    subjects = ["a forklift driver", "two kids", "a vendor", "a stray dog",
                "a barista", "a traffic cop", "an old fisherman", "a courier"]
    verbs = ["stacks crates", "cross the street", "arranges fruit", "trots past",
             "steams milk", "waves cars on", "mends a net", "locks a bike"]
    places = ["in a warehouse", "at a crosswalk", "at a night market",
              "near the harbor", "in a cafe", "downtown", "on a pier", "by a wall"]
    tails = ["rain hits the lens", "a bus passes", "a door slams", "wind lifts dust",
             "someone enters frame", "the light turns green", "gulls scatter",
             "a phone buzzes"]
    prompts = []
    for i in range(60):
        parts = [rng.choice(subjects), rng.choice(verbs), rng.choice(places) + ",",
                 rng.choice(tails) + ","] + rng.sample(tails, k=rng.randint(1, 3))
        prompts.append((" ".join(parts)).capitalize() + f" clip {i}.")
    db = tmp_path / "diverse.db"
    _make_db(db, prompts)
    assert _run_check(db) == 0
