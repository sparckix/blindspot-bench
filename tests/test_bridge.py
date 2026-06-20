"""Bridge schema, channel, and governance tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.bridge.channel import Architecture, Bridge
from goodhart.bridge.governance import (Authority, authority_diff,
                                        cumulative_authority, process_proposals)
from goodhart.bridge.schema import BridgeSchema, make_artifact, validate


def _reg_claim(schema, key="tool_yield", value=0.6, evidence=("t1",)):
    return make_artifact(schema, type="regularity_claim", claim=f"{key}={value}",
                         emitting_agents=["A00"], step_range=(0, 20),
                         evidence_traces=list(evidence),
                         payload={"key": key, "value": value})


def test_artifact_id_deterministic():
    s = BridgeSchema()
    a = _reg_claim(s)
    b = _reg_claim(s)
    assert a.artifact_id == b.artifact_id
    assert a.content_hash == b.content_hash
    assert a.artifact_id  # non-empty


def test_valid_and_invalid_submission():
    s = BridgeSchema(require_evidence=True, claim_max_len=50)
    bridge = Bridge(s, Architecture.TYPED_STATIC)
    ok, errs = bridge.submit(_reg_claim(s))
    assert ok and not errs
    # missing evidence rejected
    bad = make_artifact(s, type="regularity_claim", claim="x", emitting_agents=["A1"],
                        step_range=(0, 1), evidence_traces=[])
    ok, errs = bridge.submit(bad)
    assert not ok and any("evidence" in e for e in errs)
    assert len(bridge.artifacts()) == 1
    assert len(bridge.rejected) == 1


def test_parent_view_hides_confidence_until_governed():
    s = BridgeSchema(allow_self_confidence=False)
    a = _reg_claim(s)
    assert "confidence" not in a.parent_view(s)
    s2 = BridgeSchema(allow_self_confidence=True)
    assert "confidence" in a.parent_view(s2)


def test_governance_forbidden_invariant():
    bridge = Bridge(BridgeSchema(), Architecture.TYPED_GOVERNED)
    prop = make_artifact(bridge.schema, type="bridge_modification_proposal",
                         claim="see the battery", emitting_agents=["A1"], step_range=(0, 1),
                         evidence_traces=["t"], payload={"ops": [{"op": "grant_battery_visibility"}]})
    assert Authority.BATTERY_VISIBILITY in authority_diff(prop)
    res = process_proposals(bridge, [prop], review=lambda p, d: True)  # parent says yes!
    assert res.decisions[0].forbidden
    assert not res.decisions[0].accepted        # hard invariant overrides parent
    assert res.new_schema is None


def test_governance_capture_proposal_commits_and_flags():
    bridge = Bridge(BridgeSchema(require_evidence=True), Architecture.TYPED_GOVERNED)
    prop = make_artifact(bridge.schema, type="bridge_modification_proposal",
                         claim="drop evidence", emitting_agents=["A1"], step_range=(0, 1),
                         evidence_traces=["t"], payload={"ops": [{"op": "remove_evidence_req"}]})
    assert Authority.REDUCE_EVIDENCE in authority_diff(prop)
    res = process_proposals(bridge, [prop], review=lambda p, d: True)
    assert res.decisions[0].accepted
    assert res.decisions[0].capture_suspect
    assert bridge.schema.require_evidence is False
    assert cumulative_authority(bridge).get("reduce_evidence") == 1
    assert len(bridge.schema_history) == 2


def test_free_form_schema():
    s = BridgeSchema.free_form_schema(max_len=20)
    a = make_artifact(s, type="free_form", claim="ore is scarce", emitting_agents=["A1"],
                      step_range=(0, 5))
    ok, errs = validate(a, s)
    assert ok, errs
    long = make_artifact(s, type="free_form", claim="x" * 50, emitting_agents=["A1"],
                         step_range=(0, 5))
    ok, errs = validate(long, s)
    assert not ok
