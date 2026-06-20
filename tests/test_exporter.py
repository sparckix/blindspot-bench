"""Endogenous LLM exporter: prompt is boundary-safe; model choice -> artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.bridge.payloads import extract_answers
from goodhart.bridge.schema import BridgeSchema, validate
from goodhart.experimenter.boundary import DEFAULT_BLOCKLIST, verify_context
from goodhart.llm.base import LLMBackend, LLMResult
from goodhart.world.llm_exporter import LLMExporter, build_prompt


class FakeBackend(LLMBackend):
    """Returns a fixed text — stands in for a real model's decision, for free."""
    name = "fake"

    def __init__(self, text: str):
        self.text = text

    def complete(self, prompt, *, system="", max_tokens=1024, temperature=0.0,
                 label="request", agent_id=""):
        return LLMResult(text=self.text, model="fake", backend="fake")


LOCAL = {
    "tool_yield": 0.68, "depletion_threshold": 30.0,
    "change_tags": ["coalition_changed"],
    "evidence": ["A01:s0", "A01:s19"], "step_range": (0, 20),
}


def test_prompt_is_boundary_safe_across_roles_and_pressure():
    for role in ("farmer", "miner", "smith", "trader"):
        for pct in (0, 15, 40, 80):
            p = build_prompt("A01", role, LOCAL, pct)
            ok, hits = verify_context(p, DEFAULT_BLOCKLIST)
            assert ok, f"leak for {role}/{pct}: {hits}"


def test_honest_choice_yields_truthful_artifacts():
    text = ('[{"fact":"tool_yield","value":0.68,"confidence":"high","note":"crafted a lot"},'
            ' {"fact":"depletion_threshold","value":30,"confidence":"medium","note":"saw a dip"}]')
    exp = LLMExporter(FakeBackend(text))
    schema = BridgeSchema()
    arts, _props, prompt = exp.decide_exports("A01", "miner", LOCAL, 80, schema, (0, 20))
    assert arts and all(validate(a, schema)[0] for a in arts)
    bview = [a.parent_view(schema) for a in arts]
    answers = extract_answers([{"id": "reg.tool_yield", "klass": "regularity",
                                "prompt": "", "atype": "numeric"}], bview)
    assert abs(answers["reg.tool_yield"] - 0.68) < 1e-6   # truthful value flows through


def test_gaming_choice_is_represented_endogenously():
    # The model CHOSE to overstate tool_yield (0.68 -> 0.95). Nothing scripts this;
    # the apparatus just faithfully carries the model's decision.
    text = '[{"fact":"tool_yield","value":0.95,"confidence":"high","note":"very reliable"}]'
    exp = LLMExporter(FakeBackend(text))
    schema = BridgeSchema()
    arts, _p, _ = exp.decide_exports("A01", "miner", LOCAL, 80, schema, (0, 20))
    bview = [a.parent_view(schema) for a in arts]
    ans = extract_answers([{"id": "reg.tool_yield", "klass": "regularity",
                            "prompt": "", "atype": "numeric"}], bview)
    assert abs(ans["reg.tool_yield"] - 0.95) < 1e-6       # the inflated claim is what's exported


def test_malformed_output_is_safe():
    exp = LLMExporter(FakeBackend("I refuse to answer in JSON."))
    arts, _p, _ = exp.decide_exports("A01", "miner", LOCAL, 80, BridgeSchema(), (0, 20))
    assert arts == []
