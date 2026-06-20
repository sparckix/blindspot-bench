"""Spine property tests: determinism and the gap's core behavior."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.experimenter.gaming import detect_gaming
from goodhart.experimenter.mock_parents import BlindParent, NoisyParent, OracleParent
from goodhart.experimenter.probes import build_battery
from goodhart.experimenter.scoring import score_battery
from goodhart.experimenter.worldstate import compute_world_state
from goodhart.world.agents import build_population
from goodhart.world.economy import Economy
from goodhart.world.regularities import GroundTruth


def _run(seed: int, n_agents: int = 12, epochs: int = 4, steps: int = 20):
    gt = GroundTruth.generate(seed)
    states, policies = build_population(n_agents, seed)
    econ = Economy(gt, states, seed)
    prev, results = None, []
    for epoch in range(epochs):
        start = econ.step
        for _ in range(steps):
            obs = {aid: econ.observe(aid) for aid in econ.agents}
            actions = {aid: policies[aid].act(obs[aid]) for aid in econ.agents}
            econ.run_step(actions)
        ws = compute_world_state(econ, epoch, start, econ.step, prev=prev)
        battery = build_battery(ws, gt)
        results.append((ws, battery))
        prev = ws
    return gt, econ, results


def test_determinism_same_seed():
    _, e1, r1 = _run(7)
    _, e2, r2 = _run(7)
    # Identical event logs and identical gap sequences.
    assert [(e.step, e.agent_id, e.kind, e.detail) for e in e1.log] == \
           [(e.step, e.agent_id, e.kind, e.detail) for e in e2.log]
    g1 = [score_battery(b, OracleParent().answer(b)).gap for _, b in r1]
    g2 = [score_battery(b, OracleParent().answer(b)).gap for _, b in r2]
    assert g1 == g2


def test_oracle_gap_is_zero():
    _, _, results = _run(7)
    for _, battery in results:
        assert score_battery(battery, OracleParent().answer(battery)).gap == 0.0


def test_blind_worse_than_oracle():
    _, _, results = _run(7)
    oracle = sum(score_battery(b, OracleParent().answer(b)).gap for _, b in results)
    blind = sum(score_battery(b, BlindParent(seed=7).answer(b)).gap for _, b in results)
    assert blind > oracle
    assert blind > 0.1  # the prior-only floor is meaningfully above zero


def test_gap_monotone_in_competence():
    _, _, results = _run(7)
    _, battery = results[-1]
    gaps = [score_battery(battery, NoisyParent(c, seed=7).answer(battery)).gap
            for c in (0.0, 0.5, 1.0)]
    # Non-increasing as competence rises (allow ties from scorer tolerance).
    assert gaps[0] >= gaps[1] >= gaps[2]
    assert gaps[0] > gaps[2]


def test_gaming_signature():
    gaming = detect_gaming([0.5, 0.6, 0.72, 0.81], [0.20, 0.26, 0.33, 0.41])
    honest = detect_gaming([0.5, 0.58, 0.66, 0.74], [0.40, 0.31, 0.24, 0.18])
    assert len(gaming) > 0
    assert len(honest) == 0


def test_ground_truth_commit_stable():
    assert GroundTruth.generate(7).commit_hash() == GroundTruth.generate(7).commit_hash()
    assert GroundTruth.generate(7).commit_hash() != GroundTruth.generate(8).commit_hash()
