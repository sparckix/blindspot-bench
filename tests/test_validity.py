"""Independent validity tests for the world-state answer key (W_t).

The other suites build the "expected" values by re-running the same W_t code
they are meant to check (self-referential), so a bug in the answer key passes
both sides. These tests instead recompute every checked world-state quantity
DIRECTLY from the raw economy event log (`econ.log`), the agent inventories
(`econ.agents`), the posted prices (`econ.prices`) and the per-step aggregate
records (`econ.step_records`) -- never by calling any helper from
`goodhart.experimenter.worldstate`. They then assert the independent value
matches what `compute_world_state` produced.

These are designed to FAIL if someone reintroduces the audit's denominator /
argmax class of bug: a wrong intermediation denominator, an argmax over the
wrong collection, or a coalition built from the wrong edge set.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.experimenter.worldstate import compute_world_state
from goodhart.world.agents import build_population
from goodhart.world.economy import Economy
from goodhart.world.regularities import GOODS, GroundTruth

# Seeds 0-4 give a spread of highest-value goods (grain/ore/tool) so the
# price-argmax probe is not constant. Seed 6 is included deliberately because in
# its window market SUPPLY peaks on grain while market DEMAND peaks on ore -- so
# the bottleneck test genuinely discriminates a demand-vs-supply argmax swap (it
# would not if every seed had supply and demand peaking on the same good).
SEEDS = (0, 1, 2, 3, 4, 6)
N_AGENTS = 8
STEPS = 40

# Independent copy of the action-kind -> strategy-label mapping. We deliberately
# hard-code it here (rather than importing worldstate._STRATEGY_OF) so this test
# would catch a change to that mapping in the answer key.
_STRATEGY_OF = {
    "HARVEST": "farmer",
    "MINE": "miner",
    "CRAFT": "smith",
    "SELL": "trader",
    "BUY": "trader",
    "OFFER": "trader",
    "ACCEPT": "trader",
}

_MARKET_KINDS = ("SELL", "BUY")


def _run(seed: int, steps: int = STEPS, n_agents: int = N_AGENTS):
    """Run a fresh economy for `steps` and return (gt, econ, ws, window)."""
    gt = GroundTruth.generate(seed)
    states, policies = build_population(n_agents, seed)
    econ = Economy(gt, states, seed)
    start = econ.step
    for _ in range(steps):
        obs = {aid: econ.observe(aid) for aid in econ.agents}
        actions = {aid: policies[aid].act(obs[aid]) for aid in econ.agents}
        econ.run_step(actions)
    end = econ.step
    ws = compute_world_state(econ, 0, start, end, prev=None)
    return gt, econ, ws, range(start, end)


# ----------------------------------------------------------------------------
# Independent recomputations from the RAW log / inventories / prices / records.
# None of these call anything from worldstate.
# ----------------------------------------------------------------------------

def _classify_strategies(econ, window) -> dict:
    """Dominant action-kind label per agent over the window, raw from econ.log.

    Mirrors the documented mapping; falls back to the agent's declared role when
    the agent took no classifiable action in the window.
    """
    hist: dict[str, Counter] = defaultdict(Counter)
    for ev in econ.log:
        if ev.step in window and ev.agent_id in econ.agents:
            label = _STRATEGY_OF.get(ev.kind)
            if label:
                hist[ev.agent_id][label] += 1
    strategies = {}
    for aid in econ.agents:
        h = hist.get(aid)
        strategies[aid] = h.most_common(1)[0][0] if h else econ.agents[aid].role
    return strategies


def _independent_intermediated_fraction(econ, window, strategies):
    """Recompute (intermediated_fraction, total_trades) from the raw event log.

    total trades   = #SELL + #BUY events + #ACCEPT events
    intermediated  = SELL/BUY events by a trader-classified agent
                   + ACCEPT events with at least one trader-classified party
                     (the acceptor `ev.agent_id` and the offerer
                      `ev.detail['counterparty']`).
    """
    total = 0
    intermediated = 0
    for ev in econ.log:
        if ev.step not in window:
            continue
        if ev.kind in _MARKET_KINDS:
            total += 1
            if strategies.get(ev.agent_id) == "trader":
                intermediated += 1
        elif ev.kind == "ACCEPT":
            total += 1
            a = ev.agent_id
            b = ev.detail["counterparty"]
            if strategies.get(a) == "trader" or strategies.get(b) == "trader":
                intermediated += 1
    frac = round(intermediated / total, 4) if total else 0.0
    return frac, total


def _independent_demand_totals(econ, window) -> Counter:
    """Sum market_demand per good over the window, raw from step_records."""
    demand: Counter = Counter()
    for rec in econ.step_records:
        if rec.step in window:
            for g in GOODS:
                demand[g] += rec.market_demand.get(g, 0)
    return demand


def _independent_largest_component(econ, window):
    """Largest connected component of the peer-trade graph, built ONLY from
    ACCEPT events, using our own union-find. Returns (best_set, was_size_tie).
    """
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    nodes: set = set()
    for ev in econ.log:
        if ev.step in window and ev.kind == "ACCEPT":
            a = ev.agent_id
            b = ev.detail["counterparty"]
            union(a, b)
            nodes.add(a)
            nodes.add(b)

    comps: dict = defaultdict(set)
    for n in nodes:
        comps[find(n)].add(n)
    if not comps:
        return set(), False
    sizes = sorted((len(c) for c in comps.values()), reverse=True)
    was_tie = len(sizes) >= 2 and sizes[0] == sizes[1]
    best = max(comps.values(), key=len)
    return best, was_tie


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def test_intermediated_fraction_matches_independent_recompute():
    """Denominator bug guard: rebuild the fraction from raw SELL/BUY/ACCEPT
    events and assert it equals ws.intermediated_fraction across seeds."""
    checked_nonempty = 0
    for seed in SEEDS:
        _, econ, ws, window = _run(seed)
        strategies = _classify_strategies(econ, window)
        frac, total = _independent_intermediated_fraction(econ, window, strategies)
        assert frac == ws.intermediated_fraction, (
            f"seed={seed}: independent intermediated_fraction {frac} != "
            f"ws {ws.intermediated_fraction} (total_trades={total})"
        )
        # A fraction is meaningless without a sane denominator; verify it is in
        # range and only zero when there genuinely were no trades.
        assert 0.0 <= ws.intermediated_fraction <= 1.0
        if total > 0:
            checked_nonempty += 1
        else:
            assert ws.intermediated_fraction == 0.0
    # The chosen window must actually exercise trades, otherwise the guard is
    # vacuous and would not catch a denominator regression.
    assert checked_nonempty >= 1


def test_intermediated_fraction_empty_window_is_zero():
    """Robustness: a zero-length window has no trades -> fraction is 0.0, and
    our independent recompute agrees (no ZeroDivision)."""
    gt = GroundTruth.generate(0)
    states, policies = build_population(N_AGENTS, 0)
    econ = Economy(gt, states, 0)
    # Run a few real steps so the log is non-empty, then compute over an EMPTY
    # window [step, step) that contains none of them.
    for _ in range(5):
        obs = {aid: econ.observe(aid) for aid in econ.agents}
        actions = {aid: policies[aid].act(obs[aid]) for aid in econ.agents}
        econ.run_step(actions)
    s = econ.step
    ws = compute_world_state(econ, 0, s, s, prev=None)
    window = range(s, s)
    strategies = _classify_strategies(econ, window)
    frac, total = _independent_intermediated_fraction(econ, window, strategies)
    assert total == 0
    assert frac == 0.0 == ws.intermediated_fraction


def test_highest_value_resource_is_price_argmax():
    """argmax bug guard: highest_value_resource must be the argmax of
    econ.prices over GOODS, recomputed independently."""
    seen_goods = set()
    for seed in SEEDS:
        _, econ, ws, _ = _run(seed)
        expected = max(GOODS, key=lambda g: econ.prices.get(g, 0.0))
        assert expected == ws.highest_value_resource, (
            f"seed={seed}: price-argmax {expected} (prices={econ.prices}) != "
            f"ws {ws.highest_value_resource}"
        )
        # Independently confirm it is genuinely the max price, not just equal to
        # whatever ws returned.
        top_price = econ.prices[ws.highest_value_resource]
        assert all(econ.prices.get(g, 0.0) <= top_price for g in GOODS)
        seen_goods.add(ws.highest_value_resource)
    # Sanity: the planted base-price ranges overlap, so the winning good should
    # vary across seeds. If it is constant, the probe is dead (an audit finding).
    assert len(seen_goods) >= 2, f"highest_value_resource never varied: {seen_goods}"


def test_bottleneck_resource_is_demand_argmax():
    """argmax bug guard: bottleneck_resource must be the argmax of summed
    market_demand over the window (raw from step_records)."""
    for seed in SEEDS:
        _, econ, ws, window = _run(seed)
        demand = _independent_demand_totals(econ, window)
        if sum(demand.values()) > 0:
            expected = max(GOODS, key=lambda g: demand.get(g, 0))
            assert expected == ws.bottleneck_resource, (
                f"seed={seed}: demand-argmax {expected} (demand={dict(demand)}) "
                f"!= ws {ws.bottleneck_resource}"
            )
            # Confirm independently that ws's choice has maximal demand.
            top = demand.get(ws.bottleneck_resource, 0)
            assert all(demand.get(g, 0) <= top for g in GOODS)
        else:
            # Documented fallback: with no market demand, bottleneck defers to
            # the highest-value resource.
            expected = max(GOODS, key=lambda g: econ.prices.get(g, 0.0))
            assert expected == ws.bottleneck_resource


def test_dominant_coalition_is_largest_trade_component():
    """Wrong-edge-set guard: rebuild the peer-trade graph from ACCEPT events
    with our own union-find and assert the largest component's node SET equals
    set(ws.dominant_coalition)."""
    checked_nonempty = 0
    for seed in SEEDS:
        _, econ, ws, window = _run(seed)
        best, was_tie = _independent_largest_component(econ, window)
        ws_set = set(ws.dominant_coalition)
        if not best:
            # No peer trades in window -> coalition must be empty.
            assert ws_set == set(), f"seed={seed}: expected empty coalition"
            continue
        checked_nonempty += 1
        # The component SIZE is unambiguous and must always match.
        assert len(best) == len(ws_set), (
            f"seed={seed}: independent largest component size {len(best)} "
            f"({sorted(best)}) != ws size {len(ws_set)} ({sorted(ws_set)})"
        )
        if not was_tie:
            # No tie on size: the exact node set must match.
            assert best == ws_set, (
                f"seed={seed}: largest component {sorted(best)} != "
                f"ws coalition {sorted(ws_set)}"
            )
        else:
            # Size tie: worldstate breaks it by edge weight, so only require the
            # chosen set to be a genuine maximal-size component.
            assert len(ws_set) == len(best)
    assert checked_nonempty >= 1


def test_determinism_same_seed_identical_worldstate():
    """Same seed twice -> identical world-state on every checked field."""
    for seed in SEEDS:
        _, _, ws1, _ = _run(seed)
        _, _, ws2, _ = _run(seed)
        assert ws1.intermediated_fraction == ws2.intermediated_fraction
        assert ws1.highest_value_resource == ws2.highest_value_resource
        assert ws1.bottleneck_resource == ws2.bottleneck_resource
        assert ws1.dominant_coalition == ws2.dominant_coalition
        assert ws1.regularities == ws2.regularities


def test_regularities_match_ground_truth():
    """Planted-regularity sanity: ws.regularities must equal the values
    GroundTruth.generate(seed) committed for each planted parameter."""
    for seed in SEEDS:
        gt, _, ws, _ = _run(seed)
        expected = {
            "grain_regen": float(gt.grain_regen),
            "tool_yield": float(gt.tool_yield),
            "price_beta": float(gt.price_beta),
            "depletion_threshold": float(gt.depletion_threshold),
        }
        for key, val in expected.items():
            assert ws.regularities[key] == val, (
                f"seed={seed}: regularity {key}={ws.regularities.get(key)} "
                f"!= ground truth {val}"
            )
        # Independently regenerate ground truth to confirm it is a pure function
        # of the seed (the answer key cannot have drifted).
        gt2 = GroundTruth.generate(seed)
        assert (gt2.grain_regen, gt2.tool_yield, gt2.price_beta,
                gt2.depletion_threshold) == (
            gt.grain_regen, gt.tool_yield, gt.price_beta, gt.depletion_threshold)
