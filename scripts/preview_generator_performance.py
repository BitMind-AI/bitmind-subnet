"""Render synthetic CLI performance data, without a wallet or API request.

Run from the repo: python -m scripts.preview_generator_performance --width 100
Optionally export the exact terminal render with --svg /tmp/gascli-preview.svg.
"""

import argparse
from datetime import datetime, timedelta, timezone

from rich.console import Console
from rich.terminal_theme import TerminalTheme

from gas.utils.generator_performance import render_fool_history


def demo_data():
    now = datetime(2026, 9, 28, 16, tzinfo=timezone.utc)
    history = {}
    examples = [
        ("image", [.004, .006, .005, .009, .014, .012, .017, .019, .016, .021, .024, .022, 7 / 257], 7, 257, .02),
        ("video", [.003, .004, .006, .005, .008, .011, .009, .012, .014, .011, .013, .014, .015], 3, 200, .01),
    ]
    for lane, rates, fooled, total, threshold in examples:
        points = [dict(at=(now - timedelta(hours=(12-i)*14)).isoformat(), fool_rate=rate)
                  for i, rate in enumerate(rates)]
        history[lane] = dict(
            current=dict(fool_rate=fooled/total, fooled_count=fooled, total_samples=total, benchmark_run_count=20),
            threshold=threshold, minimum_samples=20, status="qualified",
            last_evaluated_at="2026-09-28T15:02:02Z", points=points)
    return dict(as_of=now.isoformat(), history=history, verification=dict(
        validator_count=3, aggregate_pass_rate=.975, total_verified=117,
        total_failed=3, total_evaluated=120, by_validator=[]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--svg")
    args = parser.parse_args()
    console = Console(width=args.width, height=60, record=True, force_terminal=True,
                      color_system="truecolor", no_color=False)
    console.print("$ gascli g perf   [dim]# illustrative demo data[/dim]")
    render_fool_history(demo_data(), hotkey="5GKGN7…DmWuGS  ·  DEMO", console=console)
    if args.svg:
        theme = TerminalTheme((15, 23, 42), (226, 232, 240),
                              [(0, 0, 0), (255, 85, 85), (134, 239, 172), (252, 211, 77),
                               (125, 211, 252), (196, 181, 253), (94, 234, 212), (226, 232, 240)])
        console.save_svg(args.svg, title="gascli g perf  ·  preview", theme=theme)
