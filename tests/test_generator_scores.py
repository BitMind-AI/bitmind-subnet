"""Modality EMA regressions, including the real validator update/load paths."""

import ast
import asyncio
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from gas.evaluation.generator_scores import GeneratorScoreState
from gas.evaluation.rewards import (
    GeneratorQualification,
    combine_generator_rewards,
    get_generator_qualification,
    resolve_generator_qualification,
)
from gas.utils.state_manager import load_validator_state, save_validator_state
from gas.koth_weights import build_koth_weights


def _qualified(image=True, video=True):
    return GeneratorQualification(qualified_image=image, qualified_video=video)


def test_unchanged_qualification_matches_original_scalar_ema():
    state = GeneratorScoreState()
    scalar = 0.0
    for image, video in [(10, 30), (40, 20), (3, 9)]:
        scalar = 0.5 * (0.3 * image + 0.7 * video) + 0.5 * scalar
        scores = state.update({0: {"image": image, "video": video}}, {"A": _qualified()}, ["A"])
        assert scores[0] == pytest.approx(scalar)


@pytest.mark.parametrize("lost", ["image", "video"])
def test_losing_one_modality_discards_only_its_history(lost):
    state = GeneratorScoreState()
    base = {0: {"image": 10, "video": 100}}
    state.update(base, {"A": _qualified()}, ["A"])
    scores = state.update(base, {"A": _qualified(image=lost != "image", video=lost != "video")}, ["A"])
    assert state.by_hotkey["A"][lost] == 0
    assert scores[0] == pytest.approx(0.3 * 7.5 if lost == "video" else 0.7 * 75)


def test_empty_pay_epoch_clears_disqualification_and_requalification_starts_over():
    state = GeneratorScoreState()
    base = {0: {"image": 10, "video": 10}}
    state.update(base, {"A": _qualified()}, ["A"])
    assert state.update({}, {"A": _qualified(False, False)}, ["A"]) == {}
    assert state.by_hotkey == {}
    assert state.update(base, {"A": _qualified()}, ["A"])[0] == 5


def test_qualified_lanes_decay_when_base_rewards_are_empty():
    state = GeneratorScoreState()
    state.update({0: {"image": 10, "video": 10}}, {"A": _qualified()}, ["A"])
    assert state.update({}, {"A": _qualified()}, ["A"])[0] == 2.5


def test_history_follows_hotkey_not_reused_uid():
    state = GeneratorScoreState()
    state.update({0: {"image": 100}}, {"A": _qualified()}, ["A"])
    scores = state.update(
        {0: {"image": 10}, 1: {"image": 10}},
        {"A": _qualified(), "replacement": _qualified()},
        ["replacement", "A"],
    )
    assert scores == {0: 1.5, 1: 9.0}
    state.update({}, {"replacement": _qualified()}, ["replacement"])
    assert "A" not in state.by_hotkey


def test_inactivity_clears_history_before_reactivation():
    state = GeneratorScoreState()
    base = {0: {"image": 10, "video": 10}}
    state.update(base, {"A": _qualified()}, ["A"])
    assert state.update(base, {"A": _qualified()}, ["A"], last_seen={"A": 1}, inactive_cutoff=2) == {}
    assert state.by_hotkey == {}
    assert state.update(base, {"A": _qualified()}, ["A"], last_seen={"A": 3}, inactive_cutoff=2)[0] == 5


def test_new_state_roundtrip_preserves_hotkeys_and_lanes(tmp_path):
    state = GeneratorScoreState()
    state.update({0: {"image": 10, "video": 20}}, {"A": _qualified()}, ["A"])
    state.save_state(tmp_path, "ema.json")
    restored = GeneratorScoreState()
    assert restored.load_state(tmp_path, "ema.json")
    assert restored.by_hotkey == state.by_hotkey
    assert restored.update({0: {"image": 10}}, {"A": _qualified(True, False)}, ["A"])[0] == 2.25


@pytest.mark.parametrize("payload", [
    "not json", "null", "[]", '{"version": 2, "by_hotkey": {}}',
    '{"version": 1, "by_hotkey": {"A": {"image": 1}}}',
    '{"version": 1, "by_hotkey": {"A": {"image": -1, "video": 0}}}',
    '{"version": 1, "by_hotkey": {"A": {"image": NaN, "video": 0}}}',
])
def test_invalid_state_cannot_reuse_previous_history(tmp_path, payload):
    state = GeneratorScoreState()
    state.by_hotkey = {"A": {"image": 100, "video": 100}}
    (tmp_path / "ema.json").write_text(payload)
    assert not state.load_state(tmp_path, "ema.json")
    assert state.by_hotkey == {}


@pytest.fixture
def validator(tmp_path):
    # Execute the actual methods without importing GPU/chain startup dependencies.
    source = Path(__file__).parents[1] / "neurons/validator/validator.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Validator")
    methods = [node for node in cls.body if isinstance(node, ast.AsyncFunctionDef)
               and node.name in {"update_scores", "save_state", "load_state"}]
    namespace = {
        "np": np, "time": time, "traceback": traceback,
        "bt": SimpleNamespace(logging=Mock()),
        "get_benchmark_results": AsyncMock(),
        "get_generator_base_rewards": lambda stats: (stats, []),
        "get_generator_qualification": get_generator_qualification,
        "resolve_generator_qualification": resolve_generator_qualification,
        "combine_generator_rewards": combine_generator_rewards,
        "load_validator_state": load_validator_state,
        "save_validator_state": save_validator_state,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    harness = type("ValidatorHarness", (), {method.name: namespace[method.name] for method in methods})
    instance = harness()
    instance.config = SimpleNamespace(
        scoring=SimpleNamespace(image_fool_cutoff=0.02, video_fool_cutoff=0.01,
                                min_fool_samples=20, image_weight=0.3, video_weight=0.7),
        benchmark_api_url="unused", neuron=SimpleNamespace(full_path=str(tmp_path)),
    )
    instance.wallet = SimpleNamespace(hotkey=None)
    instance.metagraph = SimpleNamespace(hotkeys=["A"])
    instance.generator_qualification = {}
    instance.generator_score_state = GeneratorScoreState()
    instance.scores = np.array([1000.0])  # Legacy scalar must never seed modality history.
    instance._state_lock = asyncio.Lock()
    instance.content_manager = Mock()
    instance.content_manager.get_verification_stats_last_n_hours.return_value = {0: {"image": 10, "video": 10}}
    instance.generative_challenge_manager = Mock()
    instance.generative_challenge_manager.get_all_generator_last_seen.return_value = {}
    instance.kings_state = Mock()
    instance.api = namespace["get_benchmark_results"]
    instance.api.return_value = [
        {"ss58_address": "A", "modality": modality, "fooled_count": 10, "not_fooled_count": 10}
        for modality in ("image", "video")
    ]
    return instance


def test_validator_outage_uses_cached_gates_then_clears_lost_lane(validator):
    assert asyncio.run(validator.update_scores()) == [0]
    assert validator.scores[0] == 5
    payload = validator.api.return_value
    validator.api.return_value = None
    assert asyncio.run(validator.update_scores()) == [0]
    assert validator.scores[0] == 7.5
    validator.generative_challenge_manager.set_qualification.assert_called_with(None, fresh=False)
    payload[1]["fooled_count"] = 0
    validator.api.return_value = payload
    assert asyncio.run(validator.update_scores()) == [0]
    assert validator.scores[0] == pytest.approx(0.3 * 8.75)
    assert validator.generator_score_state.by_hotkey["A"]["video"] == 0


def test_validator_all_disqualified_epoch_clears_history(validator):
    asyncio.run(validator.update_scores())
    for row in validator.api.return_value:
        row["fooled_count"] = 0
    assert asyncio.run(validator.update_scores()) == []
    assert validator.scores.tolist() == [0]
    assert validator.generator_score_state.by_hotkey == {}


def test_validator_no_current_reward_does_not_pay_historical_ema(validator):
    asyncio.run(validator.update_scores())
    validator.content_manager.get_verification_stats_last_n_hours.return_value = {}
    assert asyncio.run(validator.update_scores()) == []
    assert validator.scores[0] == 2.5  # Retained history is not an eligible payout UID.


def test_validator_legacy_snapshot_migration_does_not_seed_ema(validator, tmp_path):
    assert save_validator_state(str(tmp_path), {"scores.npy": np.array([999.0])})
    assert asyncio.run(validator.load_state())
    assert validator.scores.tolist() == [0]
    assert validator.generator_score_state.by_hotkey == {}
    asyncio.run(validator.update_scores())
    assert validator.scores[0] == 5


def test_validator_new_snapshot_restores_separate_lanes(validator):
    asyncio.run(validator.update_scores())
    asyncio.run(validator.save_state())
    validator.generator_score_state = GeneratorScoreState()
    assert asyncio.run(validator.load_state())
    assert validator.generator_score_state.by_hotkey == {"A": {"image": 5, "video": 5}}
    validator.api.return_value[1]["fooled_count"] = 0
    asyncio.run(validator.update_scores())
    assert validator.scores[0] == 2.25


def test_disqualified_video_history_cannot_inflate_actual_generator_share(validator):
    validator.metagraph.hotkeys = ["A", "B", "burn"]
    validator.content_manager.get_verification_stats_last_n_hours.return_value[1] = {"image": 10, "video": 10}
    validator.api.return_value.extend([
        {"ss58_address": "B", "modality": modality, "fooled_count": 10, "not_fooled_count": 10}
        for modality in ("image", "video")
    ])
    asyncio.run(validator.update_scores())
    validator.api.return_value[1]["fooled_count"] = 0
    eligible = asyncio.run(validator.update_scores())
    weights = build_koth_weights(
        n=3, scores=validator.scores, generator_uids=eligible, kings={},
        uid_for_hotkey=lambda hotkey: validator.metagraph.hotkeys.index(hotkey), burn_uid=2,
    )
    assert weights[0] == pytest.approx(0.16 * 2.25 / (2.25 + 7.5))
    assert weights[1] == pytest.approx(0.16 * 7.5 / (2.25 + 7.5))
    assert weights[2] == pytest.approx(0.84)


def test_missing_new_state_clears_history(tmp_path):
    state = GeneratorScoreState()
    state.by_hotkey = {"A": {"image": 100, "video": 100}}
    assert not state.load_state(tmp_path, "missing.json")
    assert state.by_hotkey == {}
