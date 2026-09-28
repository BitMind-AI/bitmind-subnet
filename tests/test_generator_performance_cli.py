import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from gas.cli import cli
from gas.utils.generator_performance import line_chart, render_fool_history, time_axis


def payload():
    current = dict(fool_rate=.0272, total_samples=257, fooled_count=7,
                   not_fooled_count=250, benchmark_run_count=20)
    empty = dict(fool_rate=None, total_samples=0, fooled_count=0,
                 not_fooled_count=0, benchmark_run_count=0)
    at = "2026-09-28T16:00:00+00:00"
    return dict(as_of=at, window_days=7, verification={}, fool_aggregate=current,
        history={
            "image": dict(current=current, threshold=.02, minimum_samples=20,
                          status="qualified", last_evaluated_at=at,
                          points=[dict(at=at, **current)] * 29),
            "video": dict(current=empty, threshold=.01, minimum_samples=20,
                          status="no_data", last_evaluated_at=None,
                          points=[dict(at=at, **empty)] * 29),
        })


@pytest.fixture
def run(monkeypatch):
    import bittensor
    from gas.protocol import miner_requests
    monkeypatch.setattr("gas.cli.load_miner_env", lambda: None)
    monkeypatch.setattr(bittensor, "Wallet", lambda **kwargs: SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address="test-hotkey")))
    fetch = Mock(return_value=dict(success=True, data=payload()))
    monkeypatch.setattr(miner_requests, "fetch_generator_performance", fetch)
    return lambda *args: CliRunner().invoke(cli, ["g", "perf", *args]), fetch


def test_default_charts_and_eligibility_caveat(run):
    invoke, fetch = run
    result = invoke()
    assert result.exit_code == 0, result.output
    assert "IMAGE  2.72%" in result.output
    assert "Fool-rate eligible" in result.output
    assert "VIDEO  —" in result.output
    assert "No recent data" in result.output
    assert "●" in result.output and "┄" in result.output
    assert "On-chain incentive is not checked" in result.output
    assert fetch.call_args.kwargs["lookback_days"] == 7


def test_json_is_pure_json_with_new_api(run):
    invoke, fetch = run
    result = invoke("--json")
    assert result.exit_code == 0
    assert json.loads(result.output) == payload()
    assert "⛽" not in result.output


def test_old_api_keeps_aggregate_and_json(run):
    invoke, fetch = run
    old = dict(verification={}, fool_aggregate=payload()["fool_aggregate"])
    fetch.return_value = dict(success=True, data=old)
    result = invoke()
    assert result.exit_code == 0
    assert "History unavailable" in result.output
    assert "samples=257" in result.output
    assert json.loads(invoke("--json").output) == old


def test_modality_filter_and_custom_window(run):
    invoke, fetch = run
    result = invoke("--modality", "image", "--lookback-days", "3")
    assert result.exit_code == 0
    assert "VIDEO" not in result.output
    assert "rolling 7-day" in result.output
    assert "last 3d" in result.output


@pytest.mark.parametrize("days", ["0", "91", "-1"])
def test_invalid_window_never_calls_api(run, days):
    invoke, fetch = run
    assert invoke("--lookback-days", days).exit_code == 2
    fetch.assert_not_called()


def test_api_failure_has_no_fake_chart(run):
    invoke, fetch = run
    fetch.return_value = dict(success=False, error="unavailable")
    result = invoke()
    assert result.exit_code == 1
    assert "unavailable" in result.output
    assert "eligible" not in result.output


def test_missing_data_is_not_zero_and_zero_is_plotted():
    assert line_chart([dict(fool_rate=None)] * 3, .02) == []
    chart = line_chart([dict(fool_rate=0), dict(fool_rate=None), dict(fool_rate=.01)], .02)
    assert sum(line.count("●") for line in chart) == 2


def test_single_point_and_narrow_terminal(monkeypatch, capsys):
    monkeypatch.setattr("gas.utils.generator_performance.shutil.get_terminal_size",
                        lambda _: SimpleNamespace(columns=40))
    render_fool_history(payload())
    assert max(len(line) for line in capsys.readouterr().out.splitlines() if "│" in line) <= 40
    assert line_chart([dict(fool_rate=.05)], .02)


@pytest.mark.parametrize("width", [28, 68, 108, 148])
def test_sparse_points_use_full_width_with_connected_lines(width):
    chart = line_chart([dict(fool_rate=.01), dict(fool_rate=.03)], .02, width=width)
    assert all(len(line) == width + 10 for line in chart)
    assert any(line[10] == "●" for line in chart[:-1])
    assert any(line[-1] == "●" for line in chart[:-1])
    assert any("─" in line[11:-1] for line in chart[:-1])


def test_missing_point_leaves_a_gap_even_on_wide_plot():
    chart = line_chart([dict(fool_rate=.01), dict(fool_rate=None), dict(fool_rate=.03)], None, width=80)
    assert all(line[11:-1].strip() == "" for line in chart[:-1])


@pytest.mark.parametrize("width", [2, 12, 28, 68, 108])
def test_time_axis_fits_and_has_aligned_ticks(width):
    axis = time_axis("2026-09-21T16:00:00Z", "2026-09-28T16:00:00Z", width)
    assert all(len(line) == width + 10 for line in axis)
    if width >= 20:
        assert "Sep 21" in axis[1] and "Sep 28" in axis[1]
        assert axis[0][10] == "┬" and axis[0][-1] == "┬"
        assert "16:00" in axis[2]
    if width == 68:
        assert axis[2].count("16:00") == 8


def test_wide_terminal_is_not_capped_at_sixty_columns(monkeypatch, capsys):
    monkeypatch.setattr("gas.utils.generator_performance.shutil.get_terminal_size",
                        lambda _: SimpleNamespace(columns=120))
    render_fool_history(payload())
    chart_rows = [line for line in capsys.readouterr().out.splitlines() if "│" in line]
    assert all(len(line) == 118 for line in chart_rows)
