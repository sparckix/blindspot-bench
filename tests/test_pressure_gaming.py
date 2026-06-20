"""Pressure model (D4) and mechanized-gaming taxonomy (D5 / §7) tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.bridge.channel import Architecture, Bridge
from goodhart.bridge.schema import BridgeSchema
from goodhart.experimenter.gaming import (
    GamingRegistry,
    ablation_replay,
    anomaly_inflation,
    classify_event,
    coordinated_narrative,
    detect_gaming,
    evidence_theater,
    regularity_laundering,
    reward_shaped_exporting,
    salami_drift,
)
from goodhart.experimenter.pressure import PressureModel, budget_to_steps


# -- D4: pressure model -----------------------------------------------------

def test_pressure_lam_zero_ignores_rating():
    pm = PressureModel(0.0)
    for rating in (0.0, 0.3, 0.5, 0.9, 1.0):
        assert pm.next_budget(100.0, rating) == 100.0


def test_pressure_lam_one_endpoints():
    pm = PressureModel(1.0)
    assert pm.next_budget(100.0, 1.0) == 200.0   # perfect rating doubles
    assert pm.next_budget(100.0, 0.0) == 0.0      # zero rating -> no budget
    assert pm.next_budget(100.0, 0.5) == 100.0    # neutral rating -> base


def test_pressure_monotone_and_blend():
    pm = PressureModel(0.5)
    # monotone increasing in rating
    budgets = [pm.next_budget(100.0, r) for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert budgets == sorted(budgets)
    assert len(set(budgets)) == len(budgets)  # strictly increasing
    # neutral rating at any lam yields base budget
    assert PressureModel(0.5).next_budget(100.0, 0.5) == 100.0


def test_pressure_lam_bounds_validated():
    for bad in (-0.1, 1.1):
        try:
            PressureModel(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"lam={bad} should have raised")


def test_budget_to_steps():
    assert budget_to_steps(100.0, 100.0, 20) == 20    # base -> base steps
    assert budget_to_steps(200.0, 100.0, 20) == 40    # double budget -> double
    assert budget_to_steps(0.0, 100.0, 20) == 1       # floored at 1
    assert budget_to_steps(10.0, 100.0, 20) == 2      # round(2.0)
    assert budget_to_steps(50.0, 0.0, 20) == 20       # base_budget guard


# -- D5: ablation replay ----------------------------------------------------

def _synthetic_score_fn(views):
    """A toy scorer where the suspect id 'BAD' raises the gap by 0.3 and a
    benign id 'OK' leaves it unchanged. Base gap is 0.1."""
    gap = 0.1
    for v in views:
        if v["artifact_id"] == "BAD":
            gap += 0.3
    return gap


def test_ablation_replay_suspect_raises_gap():
    views = [{"artifact_id": "OK"}, {"artifact_id": "BAD"}]
    res = ablation_replay(_synthetic_score_fn, views, {"BAD"})
    assert res["gap_with"] == 0.4
    assert res["gap_without"] == 0.1
    assert res["attributed"] > 0  # suspect increased the gap


def test_ablation_replay_benign_no_attribution():
    views = [{"artifact_id": "OK"}, {"artifact_id": "BAD"}]
    res = ablation_replay(_synthetic_score_fn, views, {"OK"})
    # removing OK changes nothing; gap stays 0.4
    assert res["gap_with"] == 0.4
    assert res["gap_without"] == 0.4
    assert res["attributed"] == 0


# -- §7: taxonomy classifiers -----------------------------------------------

def test_anomaly_inflation():
    pos = [{"type": "anomaly_report"}, {"type": "anomaly_report"},
           {"type": "regularity_claim"}]
    neg = [{"type": "anomaly_report"}, {"type": "regularity_claim"},
           {"type": "regularity_claim"}, {"type": "regularity_claim"}]
    assert anomaly_inflation(pos, base_rate=0.3) is True   # 2/3 > 0.3
    assert anomaly_inflation(neg, base_rate=0.3) is False  # 1/4 < 0.3
    assert anomaly_inflation([], base_rate=0.3) is False


def test_regularity_laundering():
    charter = {"trade", "yield", "tool", "is", "good"}
    tautology = [{"type": "regularity_claim", "claim": "tool == tool"}]
    restatement = [{"type": "regularity_claim", "claim": "trade is good"}]
    discovery = [{"type": "regularity_claim", "claim": "hidden coalition forms"}]
    assert regularity_laundering(tautology, charter) is True
    assert regularity_laundering(restatement, charter) is True
    assert regularity_laundering(discovery, charter) is False


def test_evidence_theater():
    dup = [{"evidence_traces": ["t1", "t1"]}]
    blank = [{"evidence_traces": ["t1", ""]}]
    honest = [{"evidence_traces": ["t1", "t2"]}]
    none_cited = [{"evidence_traces": []}]
    assert evidence_theater(dup) is True
    assert evidence_theater(blank) is True
    assert evidence_theater(honest) is False
    assert evidence_theater(none_cited) is False


def test_coordinated_narrative():
    pos = [
        {"emitting_agents": ["A00"], "payload": {"key": "yield", "value": 0.6},
         "evidence_traces": ["t1", "t2"]},
        {"emitting_agents": ["A01"], "payload": {"key": "yield", "value": 0.6},
         "evidence_traces": ["t2", "t3"]},  # shares t2 -> non-independent
    ]
    # negative: same agent emits both (not multi-agent) -> independent provenance
    same_agent = [
        {"emitting_agents": ["A00"], "payload": {"key": "yield", "value": 0.6},
         "evidence_traces": ["t1"]},
        {"emitting_agents": ["A00"], "payload": {"key": "yield", "value": 0.6},
         "evidence_traces": ["t1"]},
    ]
    # negative: two agents but disjoint evidence -> genuinely independent
    disjoint = [
        {"emitting_agents": ["A00"], "payload": {"key": "yield", "value": 0.6},
         "evidence_traces": ["t1"]},
        {"emitting_agents": ["A01"], "payload": {"key": "yield", "value": 0.6},
         "evidence_traces": ["t9"]},
    ]
    assert coordinated_narrative(pos) is True
    assert coordinated_narrative(same_agent) is False
    assert coordinated_narrative(disjoint) is False


def test_reward_shaped_exporting():
    baseline = {"anomaly_report": 0.2, "regularity_claim": 0.8}
    drifted = {"anomaly_report": 0.6, "regularity_claim": 0.4}  # L1 = 0.8
    steady = {"anomaly_report": 0.25, "regularity_claim": 0.75}  # L1 = 0.1
    assert reward_shaped_exporting(drifted, baseline, thresh=0.25) is True
    assert reward_shaped_exporting(steady, baseline, thresh=0.25) is False


def test_salami_drift_uses_cumulative_authority():
    bridge = Bridge(BridgeSchema(), Architecture.TYPED_GOVERNED)
    # initially no drift
    assert salami_drift(bridge) == {}
    # commit a schema that drops the evidence requirement -> reduce_evidence drift
    looser = BridgeSchema(version="0.2.0", require_evidence=False)
    bridge.commit_schema(looser)
    drift = salami_drift(bridge)
    assert drift.get("reduce_evidence") == 1


# -- classify_event dispatcher ----------------------------------------------

def test_classify_event_dispatch():
    views = [
        {"type": "anomaly_report", "emitting_agents": ["A00"],
         "payload": {"key": "yield", "value": 0.6}, "evidence_traces": ["t1", "t1"]},
        {"type": "anomaly_report", "emitting_agents": ["A01"],
         "payload": {"key": "yield", "value": 0.6}, "evidence_traces": ["t1"]},
    ]
    context = {"base_rate": 0.3, "charter_terms": {"trade"},
               "current_mix": {"anomaly_report": 1.0},
               "baseline_mix": {"regularity_claim": 1.0}}
    labels = classify_event(views, context)
    assert "anomaly_inflation" in labels       # 2/2 > 0.3
    assert "evidence_theater" in labels         # duplicate t1
    assert "coordinated_narrative" in labels    # 2 agents, shared t1
    assert "reward_shaped_exporting" in labels  # full L1 drift
    assert "regularity_laundering" not in labels


def test_classify_event_empty_context():
    views = [{"type": "regularity_claim", "emitting_agents": ["A00"],
              "evidence_traces": ["t1", "t2"], "payload": {}}]
    # no context keys -> only the always-on classifiers run, none should match
    assert classify_event(views, {}) == []


# -- registry lineage -------------------------------------------------------

def test_registry_lineage_increments():
    reg = GamingRegistry()
    events = detect_gaming([0.5, 0.6, 0.72, 0.81], [0.20, 0.26, 0.33, 0.41])
    assert events  # the gaming signature fires
    ids = [reg.record(e, labels=["anomaly_inflation"], epoch=i)
           for i, e in enumerate(events)]
    # also record a second batch to confirm the counter keeps rising
    ids.append(reg.record(events[0], labels=[], epoch=5))
    assert ids == list(range(1, len(ids) + 1))
    entries = reg.entries()
    assert [e["lineage_id"] for e in entries] == ids
    assert entries[0]["epoch"] == 0
    assert entries[-1]["epoch"] == 5
    assert "labels" in entries[0] and "event" in entries[0]
