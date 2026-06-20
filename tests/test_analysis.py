"""Analysis-layer tests: the seed-averaged curve, the P1-P6 scorecard, and the
self-contained HTML/JSON report. Uses genuine RunResults from a small real sweep.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.analysis import (evaluate_predictions, goodhart_curve,
                               render_html, write_report)
from goodhart.bridge.channel import Architecture
from goodhart.runner.config import RunConfig
from goodhart.runner.experiment import sweep

_LAMS = (0.0, 0.4)
_SEEDS = (7, 8)
_VALID = {"supported", "refuted", "inconclusive", "n/a"}


def _results():
    """A small but genuine sweep: arch A x {lam 0.0, 0.4} x 2 seeds, epochs=6."""
    configs = [
        RunConfig(lam=lam, architecture=Architecture.TYPED_STATIC,
                  epochs=6, seed=seed)
        for lam in _LAMS for seed in _SEEDS
    ]
    return sweep(configs)


def test_goodhart_curve_shape():
    curve = goodhart_curve(_results())
    assert set(curve.keys()) == {"A"}
    rec = curve["A"]
    for key in ("lams", "terminal_mean", "terminal_std", "meanG_mean", "n"):
        assert key in rec
    assert rec["lams"] == sorted(_LAMS)
    # Each lambda cell averaged across both seeds.
    assert rec["n"] == [len(_SEEDS), len(_SEEDS)]
    # Parallel arrays are aligned and numeric.
    assert len(rec["terminal_mean"]) == len(rec["lams"]) == len(rec["meanG_mean"])
    assert all(isinstance(v, float) for v in rec["terminal_mean"])


def test_goodhart_curve_averages_match():
    results = _results()
    curve = goodhart_curve(results)
    # Recompute the lam=0.0 terminal mean by hand and compare.
    cell = [r for r in results
            if r.config["architecture"] == "A" and r.config["lam"] == 0.0]
    expected = sum(r.terminal_gap for r in cell) / len(cell)
    idx = curve["A"]["lams"].index(0.0)
    assert abs(curve["A"]["terminal_mean"][idx] - expected) < 1e-6


def test_evaluate_predictions_complete_and_valid():
    preds = evaluate_predictions(_results())["predictions"]
    ids = [p["id"] for p in preds]
    assert ids == ["P1", "P2", "P3", "P4", "P5", "P6"]
    for p in preds:
        assert p["verdict"] in _VALID
        assert isinstance(p["claim"], str) and p["claim"]
        assert isinstance(p["evidence"], str) and p["evidence"]


def test_evaluate_predictions_never_raises_on_empty():
    # Robust to missing cells: empty input yields all-n/a, no crash.
    preds = evaluate_predictions([])["predictions"]
    assert len(preds) == 6
    assert all(p["verdict"] in _VALID for p in preds)


def test_render_html_self_contained():
    html = render_html(_results(), title="Test Run")
    assert isinstance(html, str)
    assert "<svg" in html                       # inline SVG charts present
    assert "MOCK BACKEND" in html               # mock caveat surfaced
    assert "Test Run" in html
    # The lambda grid values appear on the page.
    for lam in _LAMS:
        assert f"{lam:g}" in html


def test_write_report_creates_html_and_json(tmp_path):
    path = tmp_path / "report.html"
    returned = write_report(_results(), str(path), title="Sweep")
    assert returned == str(path)
    assert path.exists()
    sidecar = Path(str(path) + ".json")
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert "curve" in data and "predictions" in data and "runs" in data
    assert len(data["predictions"]) == 6
    assert len(data["runs"]) == len(_LAMS) * len(_SEEDS)
