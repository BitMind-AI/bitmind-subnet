"""Dependency-free terminal charts for generator benchmark performance."""

import math
import shutil
from datetime import datetime

import click


def _date(value):
    if not value:
        return "unknown"
    return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%b %d %H:%M")


def line_chart(points, threshold, width=48, height=7):
    """Evenly spaced rolling points; missing observations remain gaps, not zeros."""
    if not points or not any(p.get("fool_rate") is not None for p in points):
        return []
    width = max(2, min(width, len(points)))
    selected = [points[round(i * (len(points) - 1) / (width - 1))]
                for i in range(width)]
    values = [p.get("fool_rate") for p in selected]
    ceiling = max([.01, (threshold or 0) * 1.25] +
                  [v for v in values if v is not None])
    ceiling = math.ceil(ceiling * 100) / 100
    grid = [[" " for _ in values] for _ in range(height)]

    def level(value):
        return max(0, min(height - 1, round((1 - value / ceiling) * (height - 1))))

    if threshold is not None:
        grid[level(threshold)] = ["┄"] * width
    previous = None
    for x, value in enumerate(values):
        if value is None:
            previous = None
            continue
        y = level(value)
        if previous is not None:
            for between in range(min(previous, y) + 1, max(previous, y)):
                grid[between][x] = "│"
        grid[y][x] = "●"
        previous = y
    lines = [f"  {ceiling * (1 - y / (height - 1)) * 100:5.1f}% │{''.join(row)}"
             for y, row in enumerate(grid)]
    lines.append("         └" + "─" * width)
    return lines


def render_fool_history(data, modality=None):
    """Return False for an older API, allowing the caller to retain its summary."""
    history = data.get("history") or {}
    if not history:
        return False
    width = max(12, min(60, shutil.get_terminal_size((80, 24)).columns - 14))
    labels = {
        "qualified": ("Fool-rate eligible", "green"),
        "below_threshold": ("Below threshold", "yellow"),
        "insufficient_samples": ("Insufficient samples", "yellow"),
        "no_data": ("No recent data", "bright_black"),
        "not_applicable": ("No generator reward gate", "bright_black"),
    }
    click.echo("  FOOL RATE  ·  rolling 7-day, sample-weighted")
    click.echo(f"  Snapshot: {_date(data.get('as_of'))} UTC")
    click.echo()
    for lane in ([modality] if modality else ["image", "video"]):
        if lane not in history:
            continue
        entry = history[lane]
        current = entry["current"]
        rate = current.get("fool_rate")
        display = f"{rate:.2%}" if rate is not None else "—"
        label, color = labels.get(entry.get("status"), ("Unknown", "yellow"))
        click.echo("  " + click.style(f"{lane.upper():5}  {display}", bold=True)
                   + "  ·  " + click.style(label, fg=color))
        click.echo(f"  {current['fooled_count']:,}/{current['total_samples']:,} fooled evaluations"
                   f"  ·  {current['benchmark_run_count']:,} runs")
        threshold = entry.get("threshold")
        if threshold is not None:
            click.echo(f"  Gate: >{threshold:.0%} and ≥{entry['minimum_samples']} evaluations")
        points = entry.get("points") or []
        chart = line_chart(points, threshold, width)
        for line in chart:
            click.echo(click.style(line, fg="cyan" if lane == "image" else "magenta"))
        if points:
            click.echo(f"  {_date(points[0]['at'])} → {_date(points[-1]['at'])} UTC")
        if chart and threshold is not None:
            click.echo(f"  ┄ threshold {threshold:.0%}  ·  ● rolling fool rate")
        latest = entry.get("last_evaluated_at")
        click.echo(f"  Last evaluation: {_date(latest) + ' UTC' if latest else 'none in available history'}")
        click.echo()
    click.echo("  Counts are benchmark evaluations, not unique media.")
    click.echo("  Eligibility is per modality; no video data does not block image rewards.")
    click.echo("  On-chain incentive is not checked here. Verified activity and weight")
    click.echo("  reveal/epoch timing also affect payment.")
    click.echo()
    return True
