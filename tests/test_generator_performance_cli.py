import json
from datetime import datetime, timedelta, timezone
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from gas.cli import cli
from rich.console import Console
from gas.utils.generator_performance import line_chart, render_fool_history, _segments


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
    assert "IMAGE" in result.output and "2.72%" in result.output
    assert "Fool-rate eligible" in result.output
    assert "VIDEO" in result.output and "—" in result.output
    assert "No recent data" in result.output
    assert "●" in result.output and "threshold" in result.output
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


def points(rates):
    now = datetime(2026, 9, 28, 16, tzinfo=timezone.utc)
    return [dict(at=(now-timedelta(days=len(rates)-i-1)).isoformat(), fool_rate=rate)
            for i, rate in enumerate(rates)]


def test_missing_data_is_not_zero_and_splits_lines():
    sample = points([0, None, .01])
    assert line_chart(points([None] * 3), .02) is None
    segments = list(_segments(sample))
    assert len(segments) == 2
    assert segments[0][0][1] == 0
    assert segments[1][0][1] == 1


def test_plot_calls_do_not_bridge_missing_samples(monkeypatch):
    from gas.utils import generator_performance as renderer
    plot = Mock(wraps=renderer.plt.plot)
    monkeypatch.setattr(renderer.plt, "plot", plot)
    line_chart(points([.01, .02, None, .03, .04]), .02)
    assert plot.call_count == 2
    assert plot.call_args_list[0].args[1] == [1, 2]
    assert plot.call_args_list[1].args[1] == [3, 4]
    assert all(call.kwargs["marker"] == "braille" for call in plot.call_args_list)


@pytest.mark.parametrize("width", [28, 68, 108, 148])
def test_plot_fills_requested_width(width):
    chart = line_chart(points([.01, .03]), .02, width=width)
    lines = chart.plain.splitlines()
    assert max(len(line) for line in lines) == width
    assert all(len(line) <= width for line in lines)
    assert "\x1b" not in chart.plain


@pytest.mark.parametrize("width", [20, 40, 80, 120])
def test_dashboard_fits_terminal(width):
    output = StringIO()
    render_fool_history(payload(), console=Console(file=output, width=width, color_system=None))
    lines = output.getvalue().splitlines()
    assert all(len(line) <= width for line in lines)
    assert any(len(line) == width for line in lines)


def test_short_window_has_time_ticks_and_long_window_has_dates():
    sample = points([.01, .02])
    assert "16:00" in line_chart(sample, .02).plain
    sample = points([.01, .02, .03, .02, .04, .02, .03, .04])
    chart = line_chart(sample, .02, width=100).plain
    assert "Sep 21" in chart and "Sep 28" in chart


def test_repeated_render_has_no_state_leakage():
    first = line_chart(points([.01, .03]), .02).plain
    line_chart(points([.5, .9]), .01, modality="video")
    assert line_chart(points([.01, .03]), .02).plain == first
    assert line_chart(points([.05]), .02) is not None


def test_redirected_cli_has_no_ansi(run):
    result = run[0]()
    assert result.exit_code == 0
    assert "\x1b" not in result.output


@pytest.mark.parametrize("status,label", [
    ("below_threshold", "Below threshold"),
    ("insufficient_samples", "Insufficient samples"),
    ("not_applicable", "No generator reward gate"),
])
def test_status_labels(run, status, label):
    invoke, fetch = run
    fetch.return_value["data"]["history"]["image"]["status"] = status
    assert label in invoke().output


def test_verification_details_are_preserved(run):
    invoke, fetch = run
    fetch.return_value["data"]["verification"] = dict(
        validator_count=1, aggregate_pass_rate=.9, total_verified=9,
        total_failed=1, total_evaluated=10, by_validator=[dict(
            validator_hotkey="validator-123456", pass_rate=.9,
            total_verified=9, total_failed=1, lookback_hours=24)])
    output = invoke().output
    assert "VERIFICATION" in output and "90.0%" in output
    assert "validator-12" in output and "24h" in output


def test_external_text_is_not_rich_markup():
    output = StringIO()
    render_fool_history(payload(), hotkey="[red]not markup[/red]",
                        console=Console(file=output, width=100, color_system=None))
    assert "[red]not markup[/red]" in output.getvalue()
