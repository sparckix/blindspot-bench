"""Analysis + dashboard layer for Goodhart on the Bridge.

Turns many `RunResult`s into a seed-averaged Goodhart curve, a P1-P6 prediction
scorecard, and a self-contained HTML dashboard. Pure stdlib; deterministic.
"""

from __future__ import annotations

from .curve import evaluate_predictions, goodhart_curve
from .report import render_html, write_report

__all__ = [
    "goodhart_curve",
    "evaluate_predictions",
    "render_html",
    "write_report",
]
