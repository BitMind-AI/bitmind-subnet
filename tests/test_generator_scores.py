"""Modality EMA regressions, including the real validator update/load paths."""

import ast
import asyncio
from dataclasses import asdict
import json
import sqlite3
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
from gas.koth_weights import (
    build_koth_weights, chains_by_modality, discriminator_emissions_enabled,
    kings_by_modality,
)


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
               and node.name in {"update_scores", "save_state", "load_state", "set_weights"}]
    for method in methods:
        method.decorator_list = []
    namespace = {
        "np": np, "time": time, "traceback": traceback,
        "bt": SimpleNamespace(logging=Mock(), Subtensor=Mock()),
        "BURN_PERCENTAGE": 0.0, "BURN_SS58": "burn",
        "get_current_kings": AsyncMock(return_value={"kings": []}),
        "kings_by_modality": kings_by_modality,
        "chains_by_modality": chains_by_modality,
        "discriminator_emissions_enabled": discriminator_emissions_enabled,
        "build_koth_weights": build_koth_weights,
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
    instance.weight_globals = namespace
    instance.set_weights_fn = Mock()
    instance.api.return_value = [
        {"ss58_address": "A", "modality": modality, "fooled_count": 10, "not_fooled_count": 10}
        for modality in ("image", "video")
    ]
    return instance


def prepare_weight_submission(validator):
    validator.metagraph.hotkeys = ["A", "burn"]
    validator.metagraph.n = 2
    validator.config.netuid = 34
    validator.config.subtensor = SimpleNamespace(chain_endpoint="unused")
    validator.weight_globals["bt"].Subtensor.return_value.get_uid_for_hotkey_on_subnet.return_value = 1


@pytest.mark.parametrize("message", [
    "database disk image is malformed", "disk I/O error", "database is locked",
])
def test_database_outage_resubmits_payout_preserves_state_and_recovers(validator, message):
    prepare_weight_submission(validator)
    assert asyncio.run(validator.set_weights(100)) is True
    previous_scores = validator.scores.copy()
    previous_history = json.loads(json.dumps(validator.generator_score_state.by_hotkey))
    previous_qualification = dict(validator.generator_qualification)
    previous_payout = dict(validator.generator_score_state.last_payout)
    previous_weights = validator.set_weights_fn.call_args.args[3][1].copy()
    validator.set_weights_fn.reset_mock()
    validator.api.reset_mock()
    kings = validator.weight_globals["get_current_kings"]
    kings.reset_mock()
    validator.content_manager.get_verification_stats_last_n_hours.side_effect = sqlite3.DatabaseError(message)

    assert asyncio.run(validator.set_weights(460)) is True
    validator.set_weights_fn.assert_called_once()
    assert np.array_equal(validator.set_weights_fn.call_args.args[3][1], previous_weights)
    validator.api.assert_not_awaited()
    kings.assert_awaited_once()
    assert np.array_equal(validator.scores, previous_scores)
    assert validator.generator_score_state.by_hotkey == previous_history
    assert validator.generator_qualification == previous_qualification
    assert validator.generator_score_state.last_payout == previous_payout

    validator.set_weights_fn.reset_mock()
    validator.content_manager.get_verification_stats_last_n_hours.side_effect = None
    assert asyncio.run(validator.set_weights(820)) is True
    validator.set_weights_fn.assert_called_once()
    _, weights = validator.set_weights_fn.call_args.args[3]
    assert weights[0] == pytest.approx(0.16)


def test_missing_score_result_skips_weights(validator):
    validator.update_scores = AsyncMock(return_value=None)
    assert asyncio.run(validator.set_weights(100)) is False
    validator.set_weights_fn.assert_not_called()
    validator.weight_globals["get_current_kings"].assert_not_awaited()


def test_repeated_missing_score_results_resubmit_without_decaying_snapshot(validator):
    prepare_weight_submission(validator)
    assert asyncio.run(validator.set_weights(100)) is True
    previous_scores = validator.scores.copy()
    previous_payout = dict(validator.generator_score_state.last_payout)
    previous_weights = validator.set_weights_fn.call_args.args[3][1].copy()
    validator.update_scores = AsyncMock(return_value=None)
    validator.set_weights_fn.reset_mock()
    for block in (460, 820, 1180):
        assert asyncio.run(validator.set_weights(block)) is True
        assert np.array_equal(validator.set_weights_fn.call_args.args[3][1], previous_weights)
        assert np.array_equal(validator.scores, previous_scores)
        assert validator.generator_score_state.last_payout == previous_payout
    assert validator.set_weights_fn.call_count == 3


def test_real_empty_window_retains_intended_burn_behavior(validator):
    prepare_weight_submission(validator)
    validator.content_manager.get_verification_stats_last_n_hours.return_value = {}
    assert asyncio.run(validator.set_weights(100)) is True
    _, weights = validator.set_weights_fn.call_args.args[3]
    assert weights.tolist() == [0.0, 1.0]


def test_full_burn_does_not_require_generator_database(validator):
    prepare_weight_submission(validator)
    validator.weight_globals["BURN_PERCENTAGE"] = 1.0
    validator.content_manager.get_verification_stats_last_n_hours.side_effect = sqlite3.DatabaseError("disk I/O error")
    assert asyncio.run(validator.set_weights(100)) is True
    validator.content_manager.get_verification_stats_last_n_hours.assert_not_called()
    _, weights = validator.set_weights_fn.call_args.args[3]
    assert weights.tolist() == [0.0, 1.0]


def test_cold_start_database_outage_has_no_safe_fallback(validator):
    prepare_weight_submission(validator)
    validator.content_manager.get_verification_stats_last_n_hours.side_effect = sqlite3.DatabaseError("disk I/O error")
    assert asyncio.run(validator.set_weights(100)) is False
    validator.set_weights_fn.assert_not_called()
    assert validator.generator_score_state.last_payout is None


@pytest.mark.parametrize("empty", [False, True])
def test_payout_fallback_survives_restart_including_known_empty(validator, empty):
    prepare_weight_submission(validator)
    if empty:
        validator.content_manager.get_verification_stats_last_n_hours.return_value = {}
    assert asyncio.run(validator.set_weights(100)) is True
    expected = validator.set_weights_fn.call_args.args[3][1].copy()
    # set_weights itself must persist the snapshot, without a later challenge save.
    _restart(validator)
    validator.content_manager.get_verification_stats_last_n_hours.side_effect = sqlite3.DatabaseError("disk I/O error")
    assert asyncio.run(validator.set_weights(460)) is True
    assert np.array_equal(validator.set_weights_fn.call_args.args[3][1], expected)
    assert validator.scores.tolist() == [0.0, 0.0]  # Fallback does not rebuild EMA.


@pytest.mark.parametrize("still_registered", [False, True])
def test_payout_fallback_never_transfers_to_reused_uid(validator, still_registered):
    prepare_weight_submission(validator)
    assert asyncio.run(validator.set_weights(100)) is True
    validator.metagraph.hotkeys = ["replacement", "burn"] + (["A"] if still_registered else [])
    validator.metagraph.n = len(validator.metagraph.hotkeys)
    validator.content_manager.get_verification_stats_last_n_hours.side_effect = sqlite3.DatabaseError("disk I/O error")
    assert asyncio.run(validator.set_weights(460)) is True
    _, weights = validator.set_weights_fn.call_args.args[3]
    assert weights[0] == 0
    assert weights[1] == pytest.approx(0.84 if still_registered else 1.0)
    if still_registered:
        assert weights[2] == pytest.approx(0.16)


def test_ineligible_positive_ema_is_not_in_fallback(validator):
    prepare_weight_submission(validator)
    validator.metagraph.hotkeys.append("B")
    validator.metagraph.n = 3
    stats = validator.content_manager.get_verification_stats_last_n_hours.return_value
    stats[2] = {"image": 10, "video": 10}
    validator.api.return_value.extend([
        {"ss58_address": "B", "modality": m, "fooled_count": 10, "not_fooled_count": 10}
        for m in ("image", "video")
    ])
    assert asyncio.run(validator.set_weights(100)) is True
    del stats[0]
    assert asyncio.run(validator.set_weights(460)) is True
    assert validator.scores[0] > 0
    assert set(validator.generator_score_state.last_payout) == {"B"}
    validator.content_manager.get_verification_stats_last_n_hours.side_effect = sqlite3.DatabaseError("disk I/O error")
    assert asyncio.run(validator.set_weights(820)) is True
    _, weights = validator.set_weights_fn.call_args.args[3]
    assert weights[0] == 0
    assert weights[2] == pytest.approx(0.16)


def test_successful_disqualification_replaces_fallback_with_empty(validator):
    prepare_weight_submission(validator)
    assert asyncio.run(validator.set_weights(100)) is True
    for row in validator.api.return_value:
        row["fooled_count"] = 0
    assert asyncio.run(validator.set_weights(460)) is True
    assert validator.generator_score_state.last_payout == {}
    validator.content_manager.get_verification_stats_last_n_hours.side_effect = sqlite3.DatabaseError("disk I/O error")
    assert asyncio.run(validator.set_weights(820)) is True
    assert validator.set_weights_fn.call_args.args[3][1].tolist() == [0.0, 1.0]


def test_fallback_refreshes_kings_instead_of_replaying_old_final_weights(validator):
    prepare_weight_submission(validator)
    validator.metagraph.hotkeys.append("king")
    validator.metagraph.n = 3
    assert asyncio.run(validator.set_weights(100)) is True
    validator.weight_globals["get_current_kings"].return_value = {
        "kings": [{"modality": "image", "ss58_address": "king"}],
        "emissions_enabled": True, "emissions_start_at": "2020-01-01T00:00:00Z",
    }
    validator.weight_globals["bt"].Subtensor.return_value.get_uid_for_hotkey_on_subnet.side_effect = (
        lambda hotkey_ss58, netuid: validator.metagraph.hotkeys.index(hotkey_ss58)
    )
    validator.content_manager.get_verification_stats_last_n_hours.side_effect = sqlite3.DatabaseError("disk I/O error")
    assert asyncio.run(validator.set_weights(460)) is True
    _, weights = validator.set_weights_fn.call_args.args[3]
    assert weights.tolist() == pytest.approx([0.16, 0.44, 0.4])


def test_legacy_snapshot_needs_successful_scoring_before_outage_fallback(validator, tmp_path):
    prepare_weight_submission(validator)
    assert asyncio.run(validator.set_weights(100)) is True
    path = tmp_path / "state_current" / "generator_scores.json"
    payload = json.loads(path.read_text())
    del payload["last_payout"]
    path.write_text(json.dumps(payload))
    _restart(validator)
    assert validator.generator_score_state.last_payout is None
    assert validator.generator_score_state.by_hotkey  # Existing EMA is preserved.
    validator.set_weights_fn.reset_mock()
    validator.content_manager.get_verification_stats_last_n_hours.side_effect = sqlite3.DatabaseError("disk I/O error")
    assert asyncio.run(validator.set_weights(460)) is False
    validator.set_weights_fn.assert_not_called()
    validator.content_manager.get_verification_stats_last_n_hours.side_effect = None
    assert asyncio.run(validator.set_weights(820)) is True
    assert validator.generator_score_state.last_payout


@pytest.mark.parametrize("payout", [[], {"": 1}, {"A": True}, {"A": "1"},
                                     {"A": -1}, {"A": 0}, {"A": float("nan")},
                                     {"A": float("inf")}])
def test_invalid_payout_snapshot_cannot_leave_stale_fallback(tmp_path, payout):
    state = GeneratorScoreState()
    state.last_payout = {"old": 5.0}
    (tmp_path / "ema.json").write_text(json.dumps({
        "version": 1, "by_hotkey": {}, "qualification": {}, "last_payout": payout,
    }))
    assert not state.load_state(tmp_path, "ema.json")
    assert state.last_payout is None


def test_missing_snapshot_clears_old_payout(tmp_path):
    state = GeneratorScoreState()
    state.last_payout = {"old": 5.0}
    assert not state.load_state(tmp_path, "missing.json")
    assert state.last_payout is None


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


def _restart(validator):
    validator.generator_qualification = {}
    validator.generator_score_state = GeneratorScoreState()
    validator.scores = np.zeros(len(validator.metagraph.hotkeys))
    validator.generative_challenge_manager.reset_mock()
    assert asyncio.run(validator.load_state())
    validator.generative_challenge_manager.set_qualification.assert_called_once_with(None, fresh=False)


@pytest.mark.parametrize("outage", [None, [], {"data": []}])
def test_restart_outage_restores_pay_and_ema_but_not_fresh_challenge_cache(validator, outage):
    asyncio.run(validator.update_scores())
    expected = dict(validator.generator_qualification)
    asyncio.run(validator.save_state())
    _restart(validator)
    assert validator.generator_qualification == expected
    validator.api.return_value = outage
    assert asyncio.run(validator.update_scores()) == [0]
    assert validator.scores.tolist() == [7.5]
    assert validator.generator_score_state.by_hotkey == {"A": {"image": 7.5, "video": 7.5}}
    validator.generative_challenge_manager.set_qualification.assert_called_with(None, fresh=False)


def test_restart_outage_does_not_transfer_eligibility_or_ema_to_replacement(validator):
    asyncio.run(validator.update_scores())
    asyncio.run(validator.save_state())
    validator.metagraph.hotkeys = ["replacement", "A"]
    validator.content_manager.get_verification_stats_last_n_hours.return_value[1] = {"image": 10, "video": 10}
    _restart(validator)
    validator.api.return_value = None
    assert asyncio.run(validator.update_scores()) == [1]
    assert validator.scores.tolist() == [0, 7.5]
    assert set(validator.generator_score_state.by_hotkey) == {"A"}


def test_successful_fetch_replaces_restored_gate_and_refreshes_challenges(validator):
    asyncio.run(validator.update_scores())
    asyncio.run(validator.save_state())
    _restart(validator)
    for row in validator.api.return_value:
        row["fooled_count"] = 0
    assert asyncio.run(validator.update_scores()) == []
    assert validator.scores.tolist() == [0]
    assert validator.generator_score_state.by_hotkey == {}
    assert not validator.generator_qualification["A"].qualified_image
    assert not validator.generator_qualification["A"].qualified_video
    validator.generative_challenge_manager.set_qualification.assert_called_with(
        validator.generator_qualification, fresh=True,
    )


def test_restored_unqualified_hotkeys_stay_unpaid_during_outage(validator):
    for row in validator.api.return_value:
        row["fooled_count"] = 0
    asyncio.run(validator.update_scores())
    asyncio.run(validator.save_state())
    _restart(validator)
    assert "A" in validator.generator_qualification
    validator.api.return_value = None
    assert asyncio.run(validator.update_scores()) == []
    assert validator.scores.tolist() == [0]


def test_older_ema_snapshot_without_gate_does_not_invent_eligibility(validator, tmp_path):
    asyncio.run(validator.update_scores())
    asyncio.run(validator.save_state())
    path = tmp_path / "state_current" / "generator_scores.json"
    payload = json.loads(path.read_text())
    del payload["qualification"]
    path.write_text(json.dumps(payload))
    _restart(validator)
    assert validator.generator_score_state.by_hotkey == {"A": {"image": 5, "video": 5}}
    assert validator.generator_qualification == {}
    validator.api.return_value = None
    assert asyncio.run(validator.update_scores()) == []
    assert validator.scores.tolist() == [0]


def test_backup_snapshot_restores_matching_ema_and_gate(validator, tmp_path):
    asyncio.run(validator.update_scores())
    asyncio.run(validator.save_state())
    for row in validator.api.return_value:
        row["fooled_count"] = 0
    asyncio.run(validator.update_scores())
    asyncio.run(validator.save_state())
    # Simulate an incomplete current snapshot; the previous one qualified A.
    (tmp_path / "state_current" / "complete").unlink()
    _restart(validator)
    validator.api.return_value = None
    assert asyncio.run(validator.update_scores()) == [0]
    assert validator.scores.tolist() == [7.5]


@pytest.mark.parametrize("bad_cache", [None, [], {"A": {}}, {"A": {
    "image_n": 20, "image_fooled": 10, "video_n": 20, "video_fooled": 10,
    "qualified_image": "false", "qualified_video": True,
}}, {"A": {
    "image_n": 20, "image_fooled": 21, "video_n": 20, "video_fooled": 10,
    "qualified_image": True, "qualified_video": True,
}}])
def test_invalid_persisted_gate_fails_closed_without_partial_restore(tmp_path, bad_cache):
    state = GeneratorScoreState()
    state.by_hotkey = {"A": {"image": 5, "video": 5}}
    state.qualification = {"A": _qualified()}
    (tmp_path / "ema.json").write_text(json.dumps({
        "version": 1, "by_hotkey": state.by_hotkey, "qualification": bad_cache,
    }))
    assert not state.load_state(tmp_path, "ema.json")
    assert state.by_hotkey == {}
    assert state.qualification == {}


def test_qualification_roundtrip_preserves_all_counts_and_flags(tmp_path):
    state = GeneratorScoreState()
    state.qualification = {"A": GeneratorQualification(
        image_n=100, image_fooled=10, video_n=40, video_fooled=0,
        qualified_image=True, qualified_video=False,
    )}
    state.save_state(tmp_path, "ema.json")
    restored = GeneratorScoreState()
    assert restored.load_state(tmp_path, "ema.json")
    assert asdict(restored.qualification["A"]) == asdict(state.qualification["A"])
