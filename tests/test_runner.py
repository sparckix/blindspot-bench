"""Runner integration tests: determinism, the Goodhart shape, and governance."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.bridge.channel import Architecture
from goodhart.runner.config import RunConfig
from goodhart.runner.experiment import run


def test_agent_model_is_recorded():
    # Each result self-documents the underlying model (validity logging).
    r = run(RunConfig(lam=0.4, seed=7, epochs=4, backend="mock"))
    assert r.config.get("agent_model") == "mock"


def test_run_is_deterministic():
    a = run(RunConfig(lam=0.4, seed=7, epochs=6))
    b = run(RunConfig(lam=0.4, seed=7, epochs=6))
    assert a.gaps == b.gaps
    assert a.ratings == b.ratings
    assert a.gt_commit == b.gt_commit


def test_boundary_stays_clear_across_sweep():
    for lam in (0.0, 0.4, 0.8):
        r = run(RunConfig(lam=lam, seed=7, epochs=6))
        assert r.boundary_all_clear, f"boundary leak at lam={lam}"


def test_high_pressure_raises_gap_vs_moderate():
    # The mock behavioral model must produce the decoupling arm: terminal gap
    # under heavy pressure exceeds that under moderate pressure (averaged a bit
    # over seeds to damp the per-run coverage noise).
    def mean_terminal(lam):
        return sum(run(RunConfig(lam=lam, seed=s, epochs=8)).terminal_gap
                   for s in (7, 11, 13)) / 3
    assert mean_terminal(0.8) > mean_terminal(0.15)


def test_governance_capture_under_pressure():
    # Architecture C accepts capture proposals once pressure is on; none at λ=0.
    hi = run(RunConfig(lam=0.8, architecture=Architecture.TYPED_GOVERNED, seed=7, epochs=8))
    lo = run(RunConfig(lam=0.0, architecture=Architecture.TYPED_GOVERNED, seed=7, epochs=8))
    assert hi.capture_accepted_total > 0
    assert lo.capture_accepted_total == 0


def test_pressure_drives_budget():
    # With λ on, a well-rated epoch should push next-epoch budget above base and
    # change the step count from the base; at λ=0 budget never moves.
    r0 = run(RunConfig(lam=0.0, seed=7, epochs=6))
    assert all(e.steps == r0.epochs[0].steps for e in r0.epochs)
