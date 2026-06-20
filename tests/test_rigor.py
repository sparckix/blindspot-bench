"""Tests for goodhart.analysis.rigor - statistically honest verdicts.

Runs a SMALL real mock sweep (lambda in {0.0,0.15,0.4,0.8} x arch A x 3 seeds,
short epochs) via goodhart.runner.experiment.run, then asserts that:
  * curve_with_ci returns a bootstrap CI per lambda;
  * p2/p3 verdicts are produced with valid verdict strings and never crash;
  * honest_scorecard returns all keys;
  * power_planning reports a required-N;
  * the stats-unavailable fallback path is exercised (by simulating the import
    failure flag) and degrades to 'stats_unavailable' rather than crashing.

sys.path bootstrap follows tests/test_spine.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from goodhart.analysis import rigor
from goodhart.bridge.channel import Architecture
from goodhart.runner.config import RunConfig
from goodhart.runner.experiment import run

_LAMS = (0.0, 0.15, 0.4, 0.8)
_SEEDS = (7, 11, 13)
_VALID = {
    "h1_supported", "h0_kept", "inconclusive_underpowered", "invalid_run",
    "inconclusive", "refuted", "n/a", "stats_unavailable", "descriptive",
}


def _results():
    """A small but genuine mock sweep: arch A x {lambda grid} x 3 seeds, epochs=4
    (2 no-feedback + 2 pressure-on). Kept tiny so the suite stays fast."""
    return [
        run(RunConfig(lam=lam, architecture=Architecture.TYPED_STATIC,
                      epochs=4, no_feedback_epochs=2, n_agents=6,
                      seed=seed, backend="mock"))
        for lam in _LAMS for seed in _SEEDS
    ]


# Module-scoped fixture: the sweep is deterministic, so compute it once.
@pytest.fixture(scope="module")
def results():
    return _results()


# -- toolkit availability (the whole point: stats must be wired in) --------
def test_stats_toolkit_available():
    # The ztare toolkit is expected to be importable in this environment; if it
    # were not, every verdict would be 'stats_unavailable' (also tested below).
    assert rigor.stats_available() is True


# -- 1. curve_with_ci ------------------------------------------------------
def test_curve_with_ci_returns_ci_per_lambda(results):
    curve = rigor.curve_with_ci(results, arch="A", metric="terminal")
    assert set(curve.keys()) == set(_LAMS)
    for lam, cell in curve.items():
        for key in ("point", "ci_lo", "ci_hi", "n", "values"):
            assert key in cell
        assert cell["n"] == len(_SEEDS)
        assert len(cell["values"]) == len(_SEEDS)
        assert isinstance(cell["point"], float)
        # With >=2 seeds a bootstrap CI is defined and brackets the point.
        assert cell["ci_lo"] is not None and cell["ci_hi"] is not None
        assert cell["ci_lo"] <= cell["point"] <= cell["ci_hi"]


def test_curve_with_ci_mean_metric(results):
    curve = rigor.curve_with_ci(results, arch="A", metric="mean")
    assert set(curve.keys()) == set(_LAMS)
    assert all(c["ci_lo"] is not None for c in curve.values())


# -- 2. P2 verdict ---------------------------------------------------------
def test_p2_verdict_valid_and_never_crashes(results):
    p2 = rigor.p2_goodhart_verdict(results, arch="A")
    assert p2["verdict"] in _VALID
    assert "cells" in p2 and set(p2["cells"].keys()) == set(_LAMS)
    assert "note" in p2 and isinstance(p2["note"], str)
    # min_lam (if computed) must be one of the grid lambdas.
    if p2["min_lam"] is not None:
        assert p2["min_lam"] in _LAMS
    # At 3 seeds the paired permutation test cannot resolve (needs >=5 paired),
    # so an interior dip should NOT be called 'supported'.
    assert p2["verdict"] != "h1_supported" or p2["n_paired"] >= 5


# -- 3. P3 verdict ---------------------------------------------------------
def test_p3_verdict_power_aware(results):
    p3 = rigor.p3_gaming_verdict(results)
    assert p3["verdict"] in _VALID
    assert p3["n"] == len(_LAMS) * len(_SEEDS)
    assert "rho" in p3 and "ci" in p3 and "detectable_rho_at_n" in p3
    assert isinstance(p3["note"], str)
    # Honesty guard: if the design is underpowered for this rho, it must not be
    # called supported. detectable_rho_at_n at n=12 is large (~0.74), so a small
    # observed rho cannot clear h1_supported here.
    if p3["verdict"] == "h1_supported":
        # Only allowed if the CI genuinely excludes 0.
        lo, hi = p3["ci"]
        assert (lo is not None and hi is not None) and (lo > 0 or hi < 0)


# -- 4. power planning -----------------------------------------------------
def test_power_planning_reports_required_n():
    plan = rigor.power_planning(target_rho=0.3)
    assert plan["target_rho"] == 0.3
    assert isinstance(plan["n_required"], int) and plan["n_required"] > 0
    # Sanity: detecting rho=0.3 needs many more than our 12-run mock sweep.
    assert plan["n_required"] > 12


# -- 5. honest scorecard ---------------------------------------------------
def test_honest_scorecard_has_all_keys(results):
    card = rigor.honest_scorecard(results)
    assert card["stats_available"] is True
    assert "power_planning" in card
    assert set(card["predictions"].keys()) == {"P2", "P3", "P5"}
    for pid in ("P2", "P3", "P5"):
        assert card["predictions"][pid]["verdict"] in _VALID
    # One human summary line per prediction, each prefixed with its id.
    assert len(card["summary"]) == 3
    assert [s.split(":")[0] for s in card["summary"]] == ["P2", "P3", "P5"]


def test_scorecard_never_says_supported_when_underpowered(results):
    # P2 uses a seed-paired permutation test that genuinely cannot resolve at 3
    # seeds (<5 paired), so it must NEVER be 'h1_supported' here - it should fall
    # back to 'inconclusive_underpowered' or another non-support verdict.
    card = rigor.honest_scorecard(results)
    p2v = card["predictions"]["P2"]["verdict"]
    assert p2v != "h1_supported", f"P2 claimed support while underpowered: {p2v}"

    # P3 is only allowed to claim support when its 95% CI on rho excludes 0 -
    # that is the honest bar. (The mock's gaming response is a strong, clean
    # function of lambda, so a real, CI-backed correlation is expected.)
    p3 = card["predictions"]["P3"]
    if p3["verdict"] == "h1_supported":
        lo, hi = p3["ci"]
        assert (lo is not None and hi is not None) and (lo > 0 or hi < 0), \
            "P3 claimed support without a CI that excludes 0"


# -- 6. stats-unavailable fallback (degrade gracefully, never crash) -------
def test_stats_unavailable_fallback(results, monkeypatch):
    # Simulate the toolkit being absent by flipping the module's availability
    # flag (what happens when the import in rigor.py fails on a machine without
    # the ztare repo). Every public function must return a result dict whose
    # verdict is 'stats_unavailable' rather than raising.
    monkeypatch.setattr(rigor, "_STATS_OK", False)
    assert rigor.stats_available() is False

    curve = rigor.curve_with_ci(results, arch="A")
    assert curve["verdict"] == "stats_unavailable"

    p2 = rigor.p2_goodhart_verdict(results, arch="A")
    assert p2["verdict"] == "stats_unavailable"

    p3 = rigor.p3_gaming_verdict(results)
    assert p3["verdict"] == "stats_unavailable"

    plan = rigor.power_planning()
    assert plan["verdict"] == "stats_unavailable"

    card = rigor.honest_scorecard(results)
    assert card["stats_available"] is False
    assert set(card["predictions"].keys()) == {"P2", "P3", "P5"}
    assert all(card["predictions"][pid]["verdict"] == "stats_unavailable"
               for pid in ("P2", "P3", "P5"))
    assert len(card["summary"]) == 3
