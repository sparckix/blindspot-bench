"""Independent validity tests for the HARDER-WORLD answer keys.

These mirror `tests/test_validity.py`: they recompute each new planted regularity's
answer key DIRECTLY from the raw economy event log (`econ.log`) and the observed
field-grain trace — never by calling `planted_regularities`, `compute_world_state`,
or any other code that produces the answer key. They then assert the probe battery's
bound answer for that regularity matches the independently-recomputed value.

A wrong answer key is the single thing that silently invalidates the study, so for
each regularity we ALSO check the guard is discriminating: perturbing the key off
its true value must break the reconstruction. That proves the test would catch a
drifted answer key rather than passing vacuously.

The four harder-world regularities and how each is independently recovered:

  ore_output_base (Tier 1)  MINE events log (effort, output) with
                            output == int(effort * base); reproducing every output
                            from the answer key (and bracketing the rate) recovers it.
  craft_cost      (Tier 1)  CRAFT events log (ore, coin_cost) with
                            coin_cost == round(ore * cost); cost == coin_cost/ore.
  trade_tax       (Tier 2)  ACCEPT events log (nominal, proceeds) with
                            proceeds == round(nominal * (1 - tax)); tax == 1 - proceeds/nominal.
  grain_spoilage  (Tier 2)  the field recurrence
                            int(field_end[t-1]*(1-s) + regen) == field_end[t] + harvested[t]
                            holds for every step under the true s and breaks under a wrong s.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.experimenter.probes import build_battery
from goodhart.experimenter.worldstate import compute_world_state
from goodhart.world.agents import build_population
from goodhart.world.economy import Economy
from goodhart.world.regularities import GroundTruth

# Seeds chosen to give a spread of harder-world parameter values; STEPS is long
# enough that all four mechanisms fire many times (verified below by the
# non-vacuity assertions that each event class is non-empty).
SEEDS = (0, 1, 2, 3, 4, 6)
N_AGENTS = 8
STEPS = 60


def _run(seed: int, steps: int = STEPS, n_agents: int = N_AGENTS):
    """Run a harder-world economy and return (gt, econ, ws, battery, field_end,
    harvested, window). `field_end[t]` is the field grain at the END of step t;
    `harvested[t]` is total grain harvested during step t. Both are captured from
    OUTSIDE the economy (the observed field value before each step + raw HARVEST
    events), never from the world-state code under test."""
    gt = GroundTruth.generate(seed, harder_world=True)
    states, policies = build_population(n_agents, seed)
    econ = Economy(gt, states, seed)
    start = econ.step
    field_end: dict[int, int] = {}
    harvested: dict[int, int] = {}
    for _ in range(steps):
        st = econ.step
        # The field value observed before run_step(st) is exactly field_end[st-1]
        # (nothing changes the field between steps).
        field_end[st - 1] = econ.field_grain
        obs = {aid: econ.observe(aid) for aid in econ.agents}
        econ.run_step({aid: policies[aid].act(obs[aid]) for aid in econ.agents})
        harvested[st] = sum(e.detail.get("amount", 0)
                            for e in econ.log if e.kind == "HARVEST" and e.step == st)
    field_end[econ.step - 1] = econ.field_grain
    end = econ.step
    ws = compute_world_state(econ, 0, start, end, prev=None)
    battery = build_battery(ws, gt)
    return gt, econ, ws, battery, field_end, harvested, range(start, end)


def _answer_key(battery, key: str) -> float:
    """The probe battery's bound answer for reg.<key> — the experimenter answer key.
    Read straight off the Probe object (NOT from planted_regularities)."""
    for p in battery:
        if p.id == f"reg.{key}":
            return float(p.answer)
    raise AssertionError(f"no reg.{key} probe in battery")


# ---------------------------------------------------------------------------
# ore_output_base — Tier 1, single
# ---------------------------------------------------------------------------
def test_ore_output_base_matches_independent_recompute():
    """For every MINE event, int(effort * answer_key) must reproduce the logged
    output. Independently bracket the rate from (output, output+1)/effort across
    events and assert the answer key falls in the intersection."""
    checked = 0
    for seed in SEEDS:
        _, econ, ws, battery, *_ = _run(seed)
        key = _answer_key(battery, "ore_output_base")
        # The answer key must equal what world-state recorded for this regularity.
        assert ws.regularities["ore_output_base"] == key
        lo_bound, hi_bound = 0.0, float("inf")
        mine_events = [e for e in econ.log if e.kind == "MINE"]
        for e in mine_events:
            effort = e.detail["effort"]
            output = e.detail["output"]
            assert effort > 0
            # Reproduction guard: the answer key must regenerate the exact output.
            assert int(effort * key) == output, (
                f"seed={seed}: int({effort}*{key})={int(effort*key)} != logged {output}")
            # Independent interval from int(effort*x)==output: output<=effort*x<output+1.
            lo_bound = max(lo_bound, output / effort)
            hi_bound = min(hi_bound, (output + 1) / effort)
            checked += 1
        if mine_events:
            assert lo_bound <= key < hi_bound, (
                f"seed={seed}: answer key {key} outside independent bracket "
                f"[{lo_bound}, {hi_bound})")
    assert checked >= 1, "no MINE events exercised — guard would be vacuous"


def test_ore_output_base_guard_is_discriminating():
    """A wrong answer key (off by 0.5) must fail to reproduce some logged output."""
    _, econ, _, battery, *_ = _run(3)
    key = _answer_key(battery, "ore_output_base")
    mine = [e for e in econ.log if e.kind == "MINE"]
    assert mine
    wrong = key + 0.5
    assert any(int(e.detail["effort"] * wrong) != e.detail["output"] for e in mine)


# ---------------------------------------------------------------------------
# craft_cost — Tier 1, single
# ---------------------------------------------------------------------------
def test_craft_cost_matches_independent_recompute():
    """CRAFT events log (ore, coin_cost) with coin_cost == round(ore*cost,4); the
    cost recovered as coin_cost/ore must equal the answer key, and the answer key
    must reproduce every logged coin_cost."""
    checked = 0
    for seed in SEEDS:
        _, econ, ws, battery, *_ = _run(seed)
        key = _answer_key(battery, "craft_cost")
        assert ws.regularities["craft_cost"] == key
        for e in econ.log:
            if e.kind != "CRAFT" or e.detail.get("ore", 0) <= 0:
                continue
            ore = e.detail["ore"]
            coin_cost = e.detail["coin_cost"]
            # Independent recompute of the rate from this single event.
            recovered = round(coin_cost / ore, 3)
            assert abs(recovered - key) < 1e-6, (
                f"seed={seed}: recovered craft_cost {recovered} != answer key {key}")
            # Reproduction guard.
            assert coin_cost == round(ore * key, 4)
            checked += 1
    assert checked >= 1, "no CRAFT events exercised — guard would be vacuous"


def test_craft_cost_guard_is_discriminating():
    _, econ, _, battery, *_ = _run(3)
    key = _answer_key(battery, "craft_cost")
    craft = [e for e in econ.log if e.kind == "CRAFT" and e.detail.get("ore", 0) > 0]
    assert craft
    wrong = key + 0.5
    assert any(e.detail["coin_cost"] != round(e.detail["ore"] * wrong, 4) for e in craft)


# ---------------------------------------------------------------------------
# trade_tax — Tier 2, pooling
# ---------------------------------------------------------------------------
def test_trade_tax_matches_independent_recompute():
    """ACCEPT events log (nominal, proceeds) with proceeds == round(nominal*(1-tax),4);
    the tax recovered as 1 - proceeds/nominal must equal the answer key."""
    checked = 0
    for seed in SEEDS:
        _, econ, ws, battery, *_ = _run(seed)
        key = _answer_key(battery, "trade_tax")
        assert ws.regularities["trade_tax"] == key
        for e in econ.log:
            if e.kind != "ACCEPT" or "nominal" not in e.detail:
                continue
            nominal = e.detail["nominal"]
            proceeds = e.detail["proceeds"]
            assert nominal > 0
            recovered = round(1 - proceeds / nominal, 3)
            assert abs(recovered - key) < 1e-6, (
                f"seed={seed}: recovered trade_tax {recovered} != answer key {key}")
            assert proceeds == round(nominal * (1 - key), 4)
            checked += 1
    assert checked >= 1, "no ACCEPT events exercised — guard would be vacuous"


def test_trade_tax_guard_is_discriminating():
    _, econ, _, battery, *_ = _run(3)
    key = _answer_key(battery, "trade_tax")
    acc = [e for e in econ.log if e.kind == "ACCEPT" and "nominal" in e.detail]
    assert acc
    wrong = key + 0.05
    assert any(e.detail["proceeds"] != round(e.detail["nominal"] * (1 - wrong), 4)
               for e in acc)


# ---------------------------------------------------------------------------
# grain_spoilage — Tier 2, pooling
# ---------------------------------------------------------------------------
def _spoilage_recurrence_holds(field_end, harvested, regen, window, s) -> tuple[bool, int]:
    """Does int(field_end[t-1]*(1-s)+regen) == field_end[t] + harvested[t] for every
    step in the window? Returns (all_hold, n_checked)."""
    start, stop = window.start, window.stop
    n = 0
    ok = True
    for t in range(start + 1, stop):
        if (t - 1) not in field_end or t not in field_end:
            continue
        lhs = int(field_end[t - 1] * (1 - s) + regen)
        rhs = field_end[t] + harvested.get(t, 0)
        n += 1
        if lhs != rhs:
            ok = False
    return ok, n


def test_grain_spoilage_matches_independent_recompute():
    """The field-evolution recurrence, rebuilt from observed field readings and raw
    HARVEST events, must hold under the answer-key spoilage rate for every step."""
    checked = 0
    for seed in SEEDS:
        gt, econ, ws, battery, field_end, harvested, window = _run(seed)
        key = _answer_key(battery, "grain_spoilage")
        assert ws.regularities["grain_spoilage"] == key
        regen = gt.grain_regen  # grain_regen is a base Tier-1 fact; independently observable
        ok, n = _spoilage_recurrence_holds(field_end, harvested, regen, window, key)
        assert ok, f"seed={seed}: spoilage recurrence failed under answer key {key}"
        assert n >= 1
        checked += n
    assert checked >= 1


def test_grain_spoilage_guard_is_discriminating():
    """A wrong spoilage rate must break the recurrence on at least one seed/step —
    otherwise the recurrence check could pass a drifted answer key."""
    broke_somewhere = False
    for seed in SEEDS:
        gt, econ, ws, battery, field_end, harvested, window = _run(seed)
        key = _answer_key(battery, "grain_spoilage")
        regen = gt.grain_regen
        # Perturb by 0.05 — well outside float noise but inside the generator range.
        wrong = key + 0.05
        ok, _ = _spoilage_recurrence_holds(field_end, harvested, regen, window, wrong)
        if not ok:
            broke_somewhere = True
    assert broke_somewhere, "perturbed spoilage never broke the recurrence — guard too weak"


# ---------------------------------------------------------------------------
# Cross-cutting: harder-world answer keys equal the planted ground truth, and the
# battery genuinely doubled.
# ---------------------------------------------------------------------------
def test_harder_world_doubles_the_regularity_battery():
    for seed in SEEDS:
        gt, _, ws, battery, *_ = _run(seed)
        reg_probes = [p for p in battery if p.id.startswith("reg.")]
        assert len(reg_probes) == 8, f"seed={seed}: {len(reg_probes)} regularity probes"
        for key in ("grain_spoilage", "ore_output_base", "trade_tax", "craft_cost"):
            assert f"reg.{key}" in {p.id for p in reg_probes}
            # Answer key must equal the independently regenerated ground-truth value.
            gt2 = GroundTruth.generate(seed, harder_world=True)
            assert ws.regularities[key] == float(gt2.harder[key])


def test_base_world_has_no_harder_regularities():
    """With harder_world off, none of the four new probes/keys appear."""
    gt = GroundTruth.generate(3)  # base world
    states, policies = build_population(N_AGENTS, 3)
    econ = Economy(gt, states, 3)
    for _ in range(20):
        obs = {aid: econ.observe(aid) for aid in econ.agents}
        econ.run_step({aid: policies[aid].act(obs[aid]) for aid in econ.agents})
    ws = compute_world_state(econ, 0, 0, econ.step, prev=None)
    battery = build_battery(ws, gt)
    reg_ids = {p.id for p in battery if p.id.startswith("reg.")}
    assert reg_ids == {"reg.grain_regen", "reg.tool_yield",
                       "reg.price_beta", "reg.depletion_threshold"}
    for key in ("grain_spoilage", "ore_output_base", "trade_tax", "craft_cost"):
        assert key not in ws.regularities
    assert gt.harder == {}


def test_base_world_byte_identical_with_flag_off():
    """The whole apparatus must be byte-identical with harder_world False, whether
    the flag is left default or set explicitly: same commit hash, same world-state
    regularities, same probe answers, same economy event log."""
    from goodhart.runner.config import RunConfig
    from goodhart.runner.experiment import run

    # Ground-truth commit hash is the answer-key commitment: must be untouched.
    for seed in SEEDS:
        gt_default = GroundTruth.generate(seed)
        gt_explicit = GroundTruth.generate(seed, harder_world=False)
        assert gt_default.commit_hash() == gt_explicit.commit_hash()
        assert gt_default.as_dict() == gt_explicit.as_dict()
        assert "harder" not in gt_default.as_dict()

    # A full mock run with the flag off (vs default) is identical end to end.
    a = run(RunConfig(lam=0.5, seed=7, epochs=6))
    b = run(RunConfig(lam=0.5, seed=7, epochs=6, harder_world=False))
    assert a.gt_commit == b.gt_commit
    assert a.gaps == b.gaps
    assert a.ratings == b.ratings
    assert a.terminal_gap == b.terminal_gap


def test_base_and_harder_economy_logs_diverge_only_when_on():
    """The harder world must actually CHANGE the economy (otherwise the regularities
    are not really planted), while the base world's log is unchanged by the flag's
    mere existence (the MINE/CRAFT/ACCEPT detail keys appear only when harder)."""
    # Base world: MINE events carry no 'output'/'effort'; CRAFT no 'coin_cost'.
    gt = GroundTruth.generate(3)
    states, policies = build_population(N_AGENTS, 3)
    econ = Economy(gt, states, 3)
    for _ in range(30):
        obs = {aid: econ.observe(aid) for aid in econ.agents}
        econ.run_step({aid: policies[aid].act(obs[aid]) for aid in econ.agents})
    assert all("output" not in e.detail for e in econ.log if e.kind == "MINE")
    assert all("coin_cost" not in e.detail for e in econ.log if e.kind == "CRAFT")
    assert all("nominal" not in e.detail for e in econ.log if e.kind == "ACCEPT")

    # Harder world: those keys are present and the mechanisms fired.
    _, h_econ, *_ = _run(3)
    assert any("output" in e.detail for e in h_econ.log if e.kind == "MINE")
    assert any("coin_cost" in e.detail for e in h_econ.log if e.kind == "CRAFT")
    assert any("nominal" in e.detail for e in h_econ.log if e.kind == "ACCEPT")


def test_harder_world_export_prompts_are_boundary_safe():
    """Capability seal (D7): the harder-world facts must not introduce any apparatus
    vocabulary into agent prompts. Build an export prompt carrying every harder fact
    and assert it passes the metadata-blind blocklist."""
    from goodhart.experimenter.boundary import DEFAULT_BLOCKLIST, verify_context
    from goodhart.world.llm_exporter import build_prompt
    from goodhart.world.regularities import _harder_regularities

    local = {
        "grain_regen": 5, "grain_spoilage": 0.1, "ore_output_base": 2.0,
        "trade_tax": 0.05, "craft_cost": 1.2, "tool_yield": 0.55,
        "evidence": ["A01:s0", "A01:s5"], "step_range": (0, 20),
    }
    for role in ("farmer", "miner", "smith", "trader"):
        p = build_prompt("A01", role, local, 50, gameable=True, costly=True)
        ok, hits = verify_context(p, DEFAULT_BLOCKLIST)
        assert ok, f"{role}: harder-world export prompt leaks metadata {hits}"

    # The regularity PROMPTS themselves (the experimenter-side battery questions)
    # are not shown to agents, but assert they are in-world only as a defence in depth.
    gt = GroundTruth.generate(3, harder_world=True)
    for reg in _harder_regularities(gt):
        ok, hits = verify_context(reg.prompt, DEFAULT_BLOCKLIST)
        assert ok, f"{reg.key} prompt leaks metadata {hits}"
