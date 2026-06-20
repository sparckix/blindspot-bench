"""Memory arms: parent accumulation (P1) and agent learning/discovery."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.bridge.payloads import extract_answers
from goodhart.bridge.schema import BridgeSchema, validate
from goodhart.experimenter.boundary import DEFAULT_BLOCKLIST, verify_context
from goodhart.llm.base import LLMBackend, LLMResult
from goodhart.runner.config import RunConfig
from goodhart.runner.experiment import _agent_history, run
from goodhart.world.agents import build_population
from goodhart.world.economy import Economy
from goodhart.world.llm_exporter import LLMExporter, build_learning_prompt
from goodhart.world.regularities import GroundTruth

SEED = 7


class FakeBackend(LLMBackend):
    name = "fake"

    def __init__(self, text):
        self.text = text

    def complete(self, prompt, *, system="", max_tokens=1024, temperature=0.0,
                 label="request", agent_id=""):
        return LLMResult(text=self.text, model="fake", backend="fake")


# -- parent memory (accumulation) -----------------------------------------
def test_parent_memory_does_not_hurt_and_usually_helps():
    base = dict(lam=0.0, n_agents=12, epochs=8, seed=SEED, backend="mock")
    no_mem = run(RunConfig(**base, parent_memory=False))
    mem = run(RunConfig(**base, parent_memory=True))
    mean_no = sum(no_mem.gaps) / len(no_mem.gaps)
    mean_mem = sum(mem.gaps) / len(mem.gaps)
    # At λ=0 there is no gaming to poison memory, and coverage<1 drops facts that
    # accumulated memory recovers, so memory should not increase the gap.
    assert mean_mem <= mean_no + 1e-9
    # and it should strictly help somewhere across the run
    assert any(m < n - 1e-9 for m, n in zip(mem.gaps, no_mem.gaps))


def test_parent_memory_flag_off_is_unchanged():
    # Default (no flags) must be byte-identical to explicit parent_memory=False.
    a = run(RunConfig(lam=0.4, seed=SEED, epochs=6))
    b = run(RunConfig(lam=0.4, seed=SEED, epochs=6, parent_memory=False))
    assert a.gaps == b.gaps and a.ratings == b.ratings


# -- agent learning (discovery from own history) --------------------------
def test_learning_prompt_is_boundary_safe():
    for role in ("farmer", "miner", "smith", "trader"):
        p = build_learning_prompt("A01", role,
                                  ["step 3: crafted 4 ore into 2 tools",
                                   "step 5: the common field held 12 grain"], 80)
        ok, hits = verify_context(p, DEFAULT_BLOCKLIST)
        assert ok, f"{role}: {hits}"


def test_learning_infers_value_into_artifact():
    text = '[{"fact":"tool_yield","value":0.5,"confidence":"medium","note":"2 per 4 ore"}]'
    exp = LLMExporter(FakeBackend(text))
    schema = BridgeSchema()
    arts, _props, prompt = exp.decide_exports_learning(
        "A02", "smith", ["step 3: crafted 4 ore into 2 tools"], 80, schema, (0, 20))
    assert arts and all(validate(a, schema)[0] for a in arts)
    bview = [a.parent_view(schema) for a in arts]
    ans = extract_answers([{"id": "reg.tool_yield", "klass": "r", "prompt": "",
                            "atype": "numeric"}], bview)
    assert abs(ans["reg.tool_yield"] - 0.5) < 1e-6


def test_agent_history_exposes_craft_yields():
    gt = GroundTruth.generate(SEED)
    states, policies = build_population(12, SEED)
    econ = Economy(gt, states, SEED)
    field_series = []
    for _ in range(40):
        field_series.append((econ.step, econ.field_grain))
        obs = {aid: econ.observe(aid) for aid in econ.agents}
        econ.run_step({aid: policies[aid].act(obs[aid]) for aid in econ.agents})
    # Some agent's history must expose a craft (ore->tools) so tool_yield is inferable.
    found = False
    for aid in econ.agents:
        h = _agent_history(econ, aid, econ.agents[aid].role, (0, 40), field_series)
        if any("crafted" in line for line in h):
            found = True
            break
    assert found, "no craft events surfaced in any agent history"
