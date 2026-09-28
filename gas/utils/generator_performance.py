"""Rich performance cards with Plotext's high-resolution terminal curves."""

import math
import os
from datetime import datetime, timezone

import plotext as plt
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ACCENTS = {"image": "#5eead4", "video": "#c4b5fd", "audio": "#7dd3fc"}
PLOT_COLORS = {"image": (94, 234, 212), "video": (196, 181, 253), "audio": (125, 211, 252)}
MUTED = "#94a3b8"
THRESHOLD = (148, 163, 184)
STATUS = {
    "qualified": ("Fool-rate eligible", "#86efac"),
    "below_threshold": ("Below threshold", "#fcd34d"),
    "insufficient_samples": ("Insufficient samples", "#fcd34d"),
    "no_data": ("No recent data", MUTED),
    "not_applicable": ("No generator reward gate", MUTED),
}


def _datetime(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _date(value):
    return _datetime(value).strftime("%b %d %H:%M UTC") if value else "none in available history"


def _segments(points):
    """Never join across missing data, or reinterpret it as zero."""
    segment = []
    for point in points:
        rate = point.get("fool_rate")
        if rate is None:
            if segment:
                yield segment
                segment = []
        else:
            segment.append((_datetime(point["at"]).timestamp(), rate * 100))
    if segment:
        yield segment


def line_chart(points, threshold, width=80, height=12, modality="image", submissions=()):
    """Build colored text without printing ANSI codes directly to stdout."""
    segments = list(_segments(points))
    if not segments or width < 24:
        return None
    start, end = _datetime(points[0]["at"]), _datetime(points[-1]["at"])
    left, right = start.timestamp(), end.timestamp()
    if left == right:
        left, right = left - 3600, right + 3600
    max_rate = max(rate for segment in segments for _, rate in segment)
    ceiling = max(1, max_rate * 1.15, (threshold or 0) * 125)
    step = next(s for s in (.25, .5, 1, 2, 5, 10, 20, 25, 50) if s >= ceiling / 4)
    ceiling = math.ceil(ceiling / step) * step
    # Plotext uses global state. Each synchronous render starts/ends clean.
    # Never call show(), which would bypass Rich's redirect/color handling.
    plt.clear_figure()
    try:
        plt.limit_size(False, False)
        plt.plot_size(width, height)
        plt.theme("clear")
        plt.canvas_color("default")
        plt.axes_color("default")
        plt.ticks_color(THRESHOLD)
        plt.xaxes(True, False)
        plt.yaxes(True, False)
        plt.grid(False, False)
        plt.xlim(left, right)
        plt.ylim(0, ceiling)
        ticks = [step * i for i in range(round(ceiling / step) + 1)]
        plt.yticks(ticks, [f"{t:.1f}%" for t in ticks])
        count = min(8, max(2, (width - 9) // 10))
        positions = [left + (right - left) * i / (count - 1) for i in range(count)]
        fmt = "%H:%M" if (end - start).total_seconds() <= 86400 else "%b %d"
        plt.xticks(positions, [datetime.fromtimestamp(t, timezone.utc).strftime(fmt) for t in positions])
        if threshold is not None:
            plt.horizontal_line(threshold * 100, color=(100, 116, 139))
        for segment in segments:
            xs, ys = zip(*segment)
            plt.plot(list(xs), list(ys), marker="braille",
                     color=PLOT_COLORS.get(modality, PLOT_COLORS["image"]))
        chart = Text.from_ansi(plt.build().rstrip("\n"))
        # Separate event tracks preserve both types even when their timestamps
        # round to the same terminal column. Align to the rendered plot axis,
        # whose left margin varies with the percentage tick labels.
        axis = next((line for line in chart.plain.splitlines() if "└" in line), None)
        if axis is not None:
            first = axis.index("└") + 1
            last = len(axis.rstrip()) - 1
            for field, label, color in (("submitted_at", "S", "#fbbf24"),
                                        ("revealed_at", "R", "#60a5fa")):
                columns = set()
                for row in submissions:
                    if row.get(field):
                        timestamp = _datetime(row[field]).timestamp()
                        if left <= timestamp <= right:
                            columns.add(round((timestamp - left) / (right - left) * (last - first)))
                if columns:
                    track = "".join("│" if i in columns else " " for i in range(last - first + 1))
                    chart.append("\n" + (label + " ").rjust(first) + track, style=color)
        return chart
    finally:
        plt.clear_figure()


def _modality_panel(lane, entry, width, submissions=()):
    accent = ACCENTS.get(lane, ACCENTS["image"])
    current = entry["current"]
    rate = current.get("fool_rate")
    label, status_color = STATUS.get(entry.get("status"), ("Unknown", MUTED))
    heading = Table.grid(expand=True)
    heading.add_column(ratio=1)
    heading.add_column(justify="right")
    heading.add_row(Text(f"{rate:.2%}" if rate is not None else "—", style=f"bold {accent}"),
                    Text(f"● {label}", style=status_color))
    counts = Text(f"{current['fooled_count']:,} fooled / {current['total_samples']:,} evaluations"
                  f"   ·   {current['benchmark_run_count']:,} runs", style=MUTED)
    threshold = entry.get("threshold")
    gate = (f"Gate >{threshold:.0%}  +  {entry['minimum_samples']} evaluations minimum"
            if threshold is not None else "No generator reward gate for this modality")
    points = entry.get("points") or []
    chart = line_chart(points, threshold, width=max(0, width - 6), modality=lane, submissions=submissions)
    body = [heading, counts, Text("")]
    if chart is not None:
        body.extend([chart, Text("")])
    elif not any(p.get("fool_rate") is not None for p in points):
        body.extend([Text("No benchmark observations in this history window.", style=MUTED), Text("")])
    else:
        body.extend([Text("Widen the terminal to show the chart.", style=MUTED), Text("")])
    legend = Text("━ rolling 7-day fool rate", style=accent)
    if threshold is not None:
        legend.append(f"    ─ {threshold:.0%} threshold", style=MUTED)
    legend.append("    UTC", style=MUTED)
    if chart is not None:
        body.append(legend)
        events = Text()
        for field, label, color in (("submitted_at", "S latest submissions", "#fbbf24"),
                                    ("revealed_at", "R recent reveals", "#60a5fa")):
            if any(row.get(field) and _datetime(points[0]["at"]) <= _datetime(row[field]) <= _datetime(points[-1]["at"]) for row in submissions):
                if events.plain:
                    events.append("    ")
                events.append(label, style=color)
        if events.plain:
            body.extend([events, Text("Latest events only, not 7-day event history; separate tracks share the chart's time axis.", style=MUTED)])
    body.append(Text(gate, style=MUTED))
    return Panel(Group(*body), title=Text(f" {lane.upper()} ", style=f"bold {accent}"),
                 subtitle=Text(f"Last eval: {_date(entry.get('last_evaluated_at'))}", style=MUTED),
                 title_align="left", subtitle_align="right", box=box.ROUNDED,
                 border_style="#475569", padding=(1, 2), width=width)


def _verification_panel(verification):
    rate = verification.get("aggregate_pass_rate")
    summary = Text(f"{verification.get('validator_count', 0)} validators reporting", style=MUTED)
    summary.append(f"   ·   {rate:.1%} pass" if rate is not None else "   ·   pass rate unavailable")
    summary.append(f"   ·   {verification.get('total_verified', 0):,} verified"
                   f" / {verification.get('total_failed', 0):,} failed"
                   f" / {verification.get('total_evaluated', 0):,} evaluated")
    parts = [summary]
    rows = verification.get("by_validator") or []
    if rows:
        table = Table(box=None, expand=True, padding=(0, 1), header_style=MUTED)
        for heading in ("Validator", "Pass", "Verified", "Failed", "Window"):
            table.add_column(heading, overflow="fold")
        for row in rows:
            pr = row.get("pass_rate")
            table.add_row(Text((row.get("validator_hotkey") or "")[:12] + "…"),
                          f"{pr:.1%}" if pr is not None else "—",
                          str(row.get("total_verified", 0)), str(row.get("total_failed", 0)),
                          f"{row.get('lookback_hours', '?')}h")
        parts.extend([Text(""), table])
    return Panel(Group(*parts), title=" VERIFICATION ", title_align="left",
                 border_style="#475569", box=box.ROUNDED, padding=(0, 2))


def _console():
    import click
    context = click.get_current_context(silent=True)
    color = context.color if context is not None else None
    return Console(no_color="NO_COLOR" in os.environ or color is False, force_terminal=color)


def render_chain_context(chain, console=None):
    console = console or _console()
    chain = chain or {"status": "disabled"}
    if chain.get("status") == "disabled":
        return
    incentive = chain.get("incentive")
    parts = []
    if incentive is not None:
        color = "#86efac" if incentive > 0 else "#fcd34d"
        parts.append(Text(f"Incentive  {incentive:.6g}  ({incentive:.4%})   ·   UID {chain['uid']}", style=f"bold {color}"))
        network = chain.get("network") if chain.get("network") in ("finney", "test", "local") else "custom RPC"
        at = _date(chain["as_of"]) if chain.get("as_of") else "timestamp unavailable"
        parts.append(Text(f"{network} / SN{chain['netuid']} · block {chain['block']:,} · {at}", style=MUTED))
    else:
        parts.append(Text("Incentive  —  not registered" if chain.get("status") == "not_registered"
                          else "Incentive  —  unavailable", style="#fcd34d"))
    if "positive_weight_count" in chain:
        parts.append(Text(f"{chain['positive_weight_count']}/{chain['validator_count']} permitted validators have positive revealed weight for you.", style=MUTED))
    validators = chain.get("validators") or []
    if validators:
        count = chain.get("validator_count", len(validators))
        parts.extend([Text(""), Text(f"Most recent observed events · showing {len(validators)}/{count} validators, largest by stake", style=MUTED)])
        table = Table(box=None, expand=True, padding=(0, 1), header_style=MUTED)
        for heading in ("Validator UID", "Revealed weight", "Latest event UTC", "Block"):
            table.add_column(heading, overflow="fold")
        for row in validators:
            submitted = _date(row["submitted_at"]) if row.get("submitted_at") else row.get("timing_note") or "unknown"
            label = "Commit" if chain.get("commit_reveal_enabled") else "Submit"
            table.add_row(str(row["uid"]), f"{row['weight']:.4%}",
                          Text(f"{label}  {submitted}", style="#fbbf24"),
                          str(row.get("submitted_block") or "—"))
            if chain.get("commit_reveal_enabled"):
                revealed = _date(row["revealed_at"]) if row.get("revealed_at") else "unknown"
                table.add_row("", "", Text(f"Reveal  {revealed}", style="#60a5fa"),
                              str(row.get("revealed_block") or "—"))
        parts.append(table)
        parts.append(Text("Weights are current row-normalized shares, not incentive. Events are validator-wide, not miner-specific.", style=MUTED))
    if chain.get("commit_reveal_enabled"):
        parts.append(Text("Submission ≠ reveal: these may belong to different commits. Neither marks an incentive change.", style="#fbbf24"))
    if chain.get("reveal_note"):
        parts.append(Text(chain["reveal_note"], style=MUTED))
    if chain.get("warning"):
        parts.append(Text(chain["warning"], style="#fcd34d"))
    console.print(Panel(Group(*parts), title=" ON-CHAIN SNAPSHOT ", title_align="left",
                        box=box.ROUNDED, border_style="#475569", padding=(1, 2)))
    console.print()


def render_fool_history(data, modality=None, *, hotkey=None, lookback_days=7, console=None, chain=None):
    """Old APIs return False; JSON callers bypass this renderer entirely."""
    history = data.get("history") or {}
    if not history:
        return False
    console = console or _console()
    console.print()
    console.print(Text.assemble(("GAS", "bold #5eead4"), ("  /  GENERATOR PERFORMANCE", "bold")))
    if hotkey:
        console.print(Text(hotkey, style=MUTED))
    console.print(Text(f"rolling 7-day · sample-weighted · snapshot {_date(data.get('as_of'))}", style=MUTED))
    lanes = [modality] if modality else ["image", "video"]
    all_points = next((history[lane].get("points") for lane in lanes
                       if lane in history and history[lane].get("points")), [])
    if all_points:
        console.print(Text(f"History: {_date(all_points[0]['at'])} → {_date(all_points[-1]['at'])}", style=MUTED))
    console.print()
    render_chain_context(chain, console)
    submissions = (chain or {}).get("validators") or []
    for lane in lanes:
        if lane in history:
            console.print(_modality_panel(lane, history[lane], console.width, submissions))
            console.print()
    console.print(_verification_panel(data.get("verification") or {}))
    if lookback_days != 7:
        aggregate = data.get("fool_aggregate") or {}
        rate = aggregate.get("fool_rate")
        display = f"{rate:.2%}" if rate is not None else "—"
        console.print(Text(f"Requested aggregate · last {lookback_days}d: {display}"
                           f" · {aggregate.get('total_samples', 0):,} evaluations", style=MUTED))
    console.print()
    console.print(Text("Counts are evaluations, not unique media. Missing data is not zero.", style=MUTED))
    console.print(Text("Eligibility is per modality; no video data does not block image rewards.", style=MUTED))
    note = ("Chain snapshot shown above; fool-rate eligibility does not guarantee payment."
            if (chain or {}).get("incentive") is not None else
            "On-chain incentive is not checked or unavailable; verified activity and weight reveal/epoch timing also matter.")
    console.print(Text(note, style=MUTED))
    console.print()
    return True
