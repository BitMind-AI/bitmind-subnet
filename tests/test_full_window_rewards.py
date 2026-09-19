"""The reward lookback is time-bounded, not a global newest-N sample."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gas.cache.db import ChallengeStore, ConnectionManager
from gas.cache.db.connection import create_schema
from gas.evaluation.rewards import (
    GeneratorQualification,
    combine_generator_rewards,
    get_generator_base_rewards,
)


NOW = 2_000_000_000.0


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("gas.cache.db.challenge_store.time.time", lambda: NOW)
    db = ConnectionManager(tmp_path / "prompts.db")
    with db.connect() as conn:
        create_schema(conn)
        conn.execute(
            "INSERT INTO prompts (id, content, content_type, modality, created_at) "
            "VALUES ('prompt', 'test', 'prompt', 'image', ?)", (NOW,),
        )
        # Newer work from another miner must not crowd out the target's image.
        rows = [
            (f"busy-{i}", 1, "busy", "video", "verified", None, NOW - 60 - i)
            for i in range(1100)
        ] + [
            ("older-image", 200, "target", "image", "verified", None, NOW - 23 * 3600),
            ("failed-image", 200, "target", "image", "failed", "C2PA verification failed", NOW - 22 * 3600),
            ("older-video", 200, "target", "video", "verified", None, NOW - 21 * 3600),
            ("failed-video", 200, "target", "video", "failed", "challenge_timeout", NOW - 20 * 3600),
            ("too-old", 200, "target", "image", "verified", None, NOW - 24 * 3600 - 1),
            ("no-answer", 200, "target", "image", "failed", "no_answer", NOW),
            ("pending", 200, "target", "image", "pending", None, NOW),
            ("stored", 200, "target", "image", "stored", None, NOW),
            ("at-cutoff", 2, "boundary", "image", "verified", None, NOW - 24 * 3600),
        ]
        conn.executemany(
            "INSERT INTO generator_challenge_outcomes "
            "(task_id,uid,hotkey,modality,status,failure_reason,updated_at,prompt_id,created_at) "
            "VALUES (?,?,?,?,?,?,?,'prompt',?)",
            # Scoring uses completion/update time, not creation time.
            [(*row, NOW - 25 * 3600) for row in rows],
        )
        conn.commit()
    return ChallengeStore(db)


def manager_for(store):
    # Execute the actual thin wrapper without importing optional GPU/HF modules,
    # matching the lightweight method harness used by test_generator_scores.
    source = Path(__file__).parents[1] / "gas/cache/content_manager.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ContentManager")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_verification_stats_last_n_hours")
    namespace = {"bt": SimpleNamespace(logging=Mock())}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    harness = type("ContentManagerHarness", (), {method.name: namespace[method.name]})
    manager = harness()
    manager.challenges = store
    return manager


@pytest.mark.parametrize("kwargs", [{}, {"limit": None}])
def test_default_and_none_include_every_terminal_outcome_in_window(store, kwargs):
    outcomes = store.get_outcomes_last_n_hours(24, **kwargs)
    assert len(outcomes) == 1105
    assert outcomes[-1].task_id == "at-cutoff"
    assert all(outcome.updated_at >= NOW - 86400 for outcome in outcomes)
    assert {"too-old", "no-answer", "pending", "stored"}.isdisjoint(
        outcome.task_id for outcome in outcomes
    )
    stats = store.get_outcome_stats_last_n_hours(24, **kwargs)
    assert stats["busy"]["video_verified"] == 1100
    assert stats["target"]["image_verified"] == 1
    assert stats["target"]["image_failed"] == 1
    assert stats["target"]["video_verified"] == 1
    assert stats["target"]["video_failed"] == 1
    assert stats["target"]["image_pass_rate"] == 0.5
    assert stats["target"]["video_pass_rate"] == 0.5
    assert stats["boundary"]["image_verified"] == 1


@pytest.mark.parametrize("kwargs", [{}, {"limit": None}])
def test_reward_facing_wrapper_keeps_older_qualified_images_payable(store, kwargs):
    manager = manager_for(store)
    stats = manager.get_verification_stats_last_n_hours(lookback_hours=24, **kwargs)
    assert stats["target"]["total_evaluated"] == 4
    base, _ = get_generator_base_rewards(stats)
    paid = combine_generator_rewards(
        base, {200: GeneratorQualification(qualified_image=True, qualified_video=False)},
    )
    assert paid[200] == pytest.approx(0.3 * 0.5)
    # Reproduce the former cap: all of this miner's eligible work disappears.
    truncated = manager.get_verification_stats_last_n_hours(lookback_hours=24, limit=1000)
    assert "target" not in truncated


@pytest.mark.parametrize("limit", [0, 1, 1000])
def test_explicit_limits_are_preserved(store, limit):
    outcomes = store.get_outcomes_last_n_hours(24, limit=limit)
    assert len(outcomes) == limit
    if limit:
        assert outcomes[0].task_id == "busy-0"
    stats = manager_for(store).get_verification_stats_last_n_hours(24, limit=limit)
    assert sum(s["total_evaluated"] for s in stats.values()) == limit


def test_shorter_lookback_still_excludes_older_work(store):
    stats = manager_for(store).get_verification_stats_last_n_hours(lookback_hours=2)
    assert set(stats) == {"busy"}
    assert stats["busy"]["total_verified"] == 1100
