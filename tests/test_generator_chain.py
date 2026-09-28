import copy
import json
import subprocess
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gas.protocol.generator_chain import (collect_snapshot, fetch_generator_chain,
    add_recent_reveals, PREFIX, MAX_REVEAL_PAGES)


def chain():
    graph = SimpleNamespace(block=100, hotkeys=["miner", "val1", "val2", "nonval"],
        I=[.001, 0, 0, 0], S=[0, 20, 10, 40], validator_permit=[False, True, True, False],
        last_update=[0, 90, 95, 96])
    s = Mock()
    s.metagraph.return_value = graph
    s.get_timestamp.side_effect = lambda block: datetime.fromtimestamp(block * 12, timezone.utc)
    s.commit_reveal_enabled.return_value = True
    s.weights.return_value = [(1, [(0, 10), (2, 90)]), (2, [(1, 100)]), (3, [(0, 100)])]
    return s


def test_snapshot_is_block_consistent_and_targets_miner_not_validator_update():
    s = chain()
    result = collect_snapshot(s, "miner", 34, "finney")
    assert result["incentive"] == .001
    assert result["positive_weight_count"] == 1 and result["validator_count"] == 2
    assert [r["submitted_block"] for r in result["validators"]] == [90, 95]
    assert [r["weight"] for r in result["validators"]] == [.1, 0]
    s.weights.assert_called_once_with(34, block=100)
    s.commit_reveal_enabled.assert_called_once_with(34, block=100)
    assert result["commit_reveal_enabled"] is True
    assert result["status"] == "ok"


def test_missing_hotkey_is_not_zero_incentive():
    s = chain()
    result = collect_snapshot(s, "not-registered", 34, "finney")
    assert result["status"] == "not_registered" and result["incentive"] is None
    s.weights.assert_not_called()


def test_rpc_failure_preserves_incentive_without_fake_zero_weights():
    s = chain()
    s.weights.side_effect = RuntimeError("unavailable")
    result = collect_snapshot(s, "miner", 34, "finney")
    assert result["status"] == "partial" and result["incentive"] == .001
    assert "positive_weight_count" not in result


def test_unknown_timestamp_has_no_estimated_event():
    s = chain()
    s.get_timestamp.side_effect = [datetime.now(timezone.utc), RuntimeError("pruned")]
    result = collect_snapshot(s, "miner", 34, "finney")
    assert result["status"] == "partial"
    assert all(v["submitted_at"] is None for v in result["validators"])


def test_never_submitted_or_future_block_is_not_plotted():
    s = chain()
    s.metagraph.return_value.last_update = [0, 0, 101, 0]
    result = collect_snapshot(s, "miner", 34, "finney")
    assert all(v["submitted_block"] is None for v in result["validators"])
    assert s.get_timestamp.call_count == 1


def test_limits_timestamp_queries_and_orders_validators_by_stake():
    s = chain()
    g = s.metagraph.return_value
    g.hotkeys = ["miner"] + [f"val{i}" for i in range(10)]
    g.I = [0] * 11
    g.S = list(range(11))
    g.validator_permit = [False] + [True] * 10
    g.last_update = [0] + [90] * 10
    result = collect_snapshot(s, "miner", 34, "finney")
    assert [v["uid"] for v in result["validators"]] == [10, 9, 8, 7, 6]
    assert result["validator_count"] == 10
    assert s.get_timestamp.call_count == 2  # shared timestamps fetched once


def test_stale_validators_do_not_trigger_archive_timestamp_lookups():
    s = chain()
    s.metagraph.return_value.block = 10000
    s.metagraph.return_value.last_update = [0, 9990, 50, 0]
    result = collect_snapshot(s, "miner", 34, "finney")
    assert result["status"] == "ok"
    assert result["validators"][1]["submitted_block"] == 50
    assert result["validators"][1]["submitted_at"] is None
    assert result["validators"][1]["timing_note"] == ">7200 blocks ago"
    assert s.get_timestamp.call_count == 2


def test_publishes_partial_progress_before_optional_queries():
    snapshots = []
    collect_snapshot(chain(), "miner", 34, "finney", lambda x: snapshots.append(copy.deepcopy(x)))
    assert snapshots[0]["incentive"] == .001
    assert snapshots[0]["validators"] == []
    assert snapshots[-1]["validators"][0]["submitted_at"]


def test_subprocess_is_bounded_and_stdout_is_isolated(monkeypatch):
    runner = Mock(return_value=SimpleNamespace(stdout='SDK log\n'+PREFIX+'{"status":"ok","incentive":0}\n', returncode=0))
    monkeypatch.setattr(subprocess, "run", runner)
    result = fetch_generator_chain("miner", network="test", netuid=379, timeout=12)
    assert result["incentive"] == 0 and result["status"] == "ok"
    assert runner.call_args.kwargs["timeout"] == 12
    assert runner.call_args.args[0][-3:] == ["miner", "test", "379"]


@pytest.mark.parametrize("output", [None, b"no result"])
def test_timeout_is_unknown_not_zero(monkeypatch, output):
    runner = Mock(side_effect=subprocess.TimeoutExpired("chain", 1, output=output))
    monkeypatch.setattr(subprocess, "run", runner)
    result = fetch_generator_chain("miner")
    assert result["incentive"] is None and result["status"] == "unavailable"


def test_timeout_retains_completed_snapshot_fields(monkeypatch):
    output = (PREFIX + json.dumps(dict(status="partial", incentive=.1, validators=[]))).encode()
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired("chain", 1, output=output)))
    result = fetch_generator_chain("miner")
    assert result["status"] == "partial" and result["incentive"] == .1


@pytest.mark.parametrize("output", ["", PREFIX + "not json"])
def test_broken_worker_does_not_crash_cli(monkeypatch, output):
    monkeypatch.setattr(subprocess, "run", Mock(return_value=SimpleNamespace(stdout=output, returncode=1)))
    assert fetch_generator_chain("miner")["incentive"] is None


def indexed(block=92, netuid=34, hotkey="val1"):
    return dict(block_number=block, pallet="SubtensorModule", method="TimelockedWeightsRevealed",
                args=[[netuid], hotkey], observed_at=0)


def page(events=(), cursor=None, degraded=False):
    return Mock(headers={"x-metagraph-degraded": "tier_unavailable"} if degraded else {},
                json=Mock(return_value=dict(ok=True, data=dict(events=list(events), next_cursor=cursor))))


def revealed(hotkey="val1", netuid=34):
    return dict(event=dict(module_id="SubtensorModule", event_id="TimelockedWeightsRevealed",
                           attributes=(netuid, hotkey)))


def test_reveal_locator_is_verified_on_chain_and_never_uses_index_timestamp():
    s = chain()
    snapshot = collect_snapshot(s, "miner", 34, "finney")
    get = Mock(return_value=page([indexed(), indexed(hotkey="val2")]))
    s.get_block_hash.return_value = "block-hash"
    s.substrate.get_events.return_value = [revealed(), revealed("val2"), revealed(netuid=99)]
    add_recent_reveals(s, snapshot, get=get)
    first, second = snapshot["validators"]
    assert first["revealed_block"] == second["revealed_block"] == 92
    assert first["revealed_at"] == datetime.fromtimestamp(92 * 12, timezone.utc).isoformat()
    # Latest val2 commit is newer than its reveal: don't invent a pairing.
    assert second["submitted_block"] == 95
    s.substrate.get_events.assert_called_once_with(block_hash="block-hash")
    assert "miner" not in str(get.call_args)


@pytest.mark.parametrize("events", [[], [revealed(netuid=99)], [revealed("unrelated")]])
def test_index_cannot_fabricate_a_reveal(events):
    s = chain()
    snapshot = collect_snapshot(s, "miner", 34, "finney")
    s.substrate.get_events.return_value = events
    add_recent_reveals(s, snapshot, get=Mock(return_value=page([indexed()])))
    assert all(not r.get("revealed_at") for r in snapshot["validators"])


def test_reveal_search_skips_future_stale_and_wrong_subnet_events():
    s = chain()
    snapshot = collect_snapshot(s, "miner", 34, "finney")
    events = [indexed(101), indexed(-8000), indexed(netuid=99), indexed(hotkey="unrelated")]
    add_recent_reveals(s, snapshot, get=Mock(return_value=page(events)))
    s.substrate.get_events.assert_not_called()


def test_reveal_search_has_a_page_limit_and_uses_opaque_cursors():
    s = chain()
    snapshot = collect_snapshot(s, "miner", 34, "finney")
    get = Mock(side_effect=[page(cursor=f"cursor-{i}") for i in range(MAX_REVEAL_PAGES)])
    add_recent_reveals(s, snapshot, get=get)
    assert get.call_count == MAX_REVEAL_PAGES
    assert get.call_args_list[1].kwargs["params"]["cursor"] == "cursor-0"
    assert get.call_args.kwargs["timeout"] == 3


def test_reveal_search_stops_repeated_cursor():
    s = chain()
    snapshot = collect_snapshot(s, "miner", 34, "finney")
    get = Mock(return_value=page(cursor="same"))
    add_recent_reveals(s, snapshot, get=get)
    assert get.call_count == 2


@pytest.mark.parametrize("response", [page(degraded=True), RuntimeError("offline")])
def test_reveal_index_failure_is_unknown_and_preserves_chain_snapshot(response):
    s = chain()
    snapshot = collect_snapshot(s, "miner", 34, "finney")
    get = Mock(side_effect=response) if isinstance(response, Exception) else Mock(return_value=response)
    add_recent_reveals(s, snapshot, get=get)
    assert snapshot["incentive"] == .001
    assert snapshot["validators"][0]["submitted_block"] == 90
    assert "incomplete/unavailable" in snapshot["reveal_note"]


@pytest.mark.parametrize("network, enabled", [("test", True), ("wss://custom", True), ("finney", False)])
def test_index_not_used_for_other_networks_or_non_commit_reveal(network, enabled):
    s = chain()
    s.commit_reveal_enabled.return_value = enabled
    snapshot = collect_snapshot(s, "miner", 34, network)
    get = Mock()
    add_recent_reveals(s, snapshot, get=get)
    get.assert_not_called()


def test_partial_reveal_results_survive_later_index_failure():
    s = chain()
    snapshot = collect_snapshot(s, "miner", 34, "finney")
    s.substrate.get_events.return_value = [revealed()]
    get = Mock(side_effect=[page([indexed()], cursor="next"), RuntimeError("offline")])
    checkpoints = []
    add_recent_reveals(s, snapshot, lambda x: checkpoints.append(copy.deepcopy(x)), get=get)
    assert snapshot["validators"][0]["revealed_block"] == 92
    assert checkpoints[-1]["validators"][0]["revealed_block"] == 92
    assert not snapshot["validators"][1].get("revealed_block")
