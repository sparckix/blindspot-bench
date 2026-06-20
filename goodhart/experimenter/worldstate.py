"""World-state vector W_t (D1).

The complete ground-truth description of the inner world over an epoch's step
window, computed exactly from the deterministic physics and the event log. This
is the answer key the probe battery is scored against; it is never exposed below
the experimenter layer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..world.economy import Economy
from ..world.regularities import GOODS, GroundTruth

# Which logged action kinds map to which mined strategy label.
_STRATEGY_OF = {
    "HARVEST": "farmer",
    "MINE": "miner",
    "CRAFT": "smith",
    "SELL": "trader",
    "BUY": "trader",
    "OFFER": "trader",
    "ACCEPT": "trader",
}


def _gini(values: list[float]) -> float:
    xs = sorted(v for v in values if v >= 0)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, start=1):
        cum += i * x
    return round((2 * cum) / (n * sum(xs)) - (n + 1) / n, 4)


@dataclass
class WorldState:
    epoch: int
    step_window: tuple

    # planted regularities (the schema-independent probe spine)
    regularities: dict

    # distributional facts
    posted_prices: dict
    highest_value_resource: str
    wealth_gini: float
    total_coin: float
    resource_totals: dict
    intermediated_fraction: float

    # structural facts
    dominant_coalition: frozenset
    bottleneck_resource: str
    strategies: dict             # agent_id -> mined strategy label
    trade_edges: dict            # (a,b) -> count, undirected canonicalized

    # dynamics facts
    depletion_active: bool
    depletion_triggers: int
    change_tags: set = field(default_factory=set)

    def public_summary(self) -> dict:
        """A non-answer-key digest, safe to print in run logs."""
        return {
            "epoch": self.epoch,
            "highest_value_resource": self.highest_value_resource,
            "bottleneck_resource": self.bottleneck_resource,
            "wealth_gini": self.wealth_gini,
            "intermediated_fraction": self.intermediated_fraction,
            "dominant_coalition": sorted(self.dominant_coalition),
            "depletion_active": self.depletion_active,
            "depletion_triggers": self.depletion_triggers,
            "change_tags": sorted(self.change_tags),
        }


def compute_world_state(
    econ: Economy,
    epoch: int,
    start_step: int,
    end_step: int,
    prev: "WorldState | None" = None,
) -> WorldState:
    gt: GroundTruth = econ.gt
    window = range(start_step, end_step)

    # --- strategy mining from the action log ---------------------------------
    action_hist: dict[str, Counter] = defaultdict(Counter)
    for ev in econ.log:
        if ev.step in window and ev.agent_id in econ.agents:
            label = _STRATEGY_OF.get(ev.kind)
            if label:
                action_hist[ev.agent_id][label] += 1
    strategies = {
        aid: (hist.most_common(1)[0][0] if hist else econ.agents[aid].role)
        for aid, hist in action_hist.items()
    }
    for aid in econ.agents:
        strategies.setdefault(aid, econ.agents[aid].role)

    # --- trade network + intermediation --------------------------------------
    edges: dict = Counter()
    peer_trade_count = 0
    intermediated = 0
    demand_totals = Counter()
    for rec in econ.step_records:
        if rec.step not in window:
            continue
        for g in GOODS:
            demand_totals[g] += rec.market_demand.get(g, 0)
        # peer_trades stores both legs; count accept events as half.
        legs = rec.peer_trades
        for i in range(0, len(legs), 2):
            a, b = legs[i][0], legs[i][1]
            key = tuple(sorted((a, b)))
            edges[key] += 1
            peer_trade_count += 1
            if strategies.get(a) == "trader" or strategies.get(b) == "trader":
                intermediated += 1

    # Market trade EVENTS (one per SELL/BUY), commensurate with peer-trade events,
    # so the intermediated FRACTION has a well-defined denominator (audit fix).
    market_event_count = 0
    for ev in econ.log:
        if ev.step in window and ev.kind in ("SELL", "BUY"):
            market_event_count += 1
            if strategies.get(ev.agent_id) == "trader":
                intermediated += 1

    total_trades = market_event_count + peer_trade_count
    intermediated_fraction = round(intermediated / total_trades, 4) if total_trades else 0.0

    # --- dominant coalition: largest connected component of the trade graph ---
    adj: dict = defaultdict(set)
    for (a, b) in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen: set = set()
    components: list[set] = []
    for node in adj:
        if node in seen:
            continue
        stack, comp = [node], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.add(x)
            stack.extend(adj[x] - seen)
        components.append(comp)
    if components:
        dominant = max(components, key=lambda c: (len(c), sum(edges[tuple(sorted((a, b)))]
                       for a in c for b in c if a < b and tuple(sorted((a, b))) in edges)))
    else:
        dominant = set()
    dominant_coalition = frozenset(dominant)

    # --- distributional + structural scalars ---------------------------------
    resource_totals = {g: sum(int(a.get(g)) for a in econ.agents.values()) for g in GOODS}
    total_coin = round(sum(a.get("coin") for a in econ.agents.values()), 2)
    wealth_gini = _gini([a.get("coin") for a in econ.agents.values()])
    highest_value_resource = max(GOODS, key=lambda g: econ.prices.get(g, 0.0))
    bottleneck_resource = (
        max(GOODS, key=lambda g: demand_totals.get(g, 0))
        if sum(demand_totals.values()) > 0
        else highest_value_resource
    )

    depletion_triggers = sum(
        1 for rec in econ.step_records if rec.step in window and rec.depletion_triggered
    )
    depletion_active = any(
        rec.depletion_active for rec in econ.step_records if rec.step in window
    )

    ws = WorldState(
        epoch=epoch,
        step_window=(start_step, end_step),
        regularities={
            "grain_regen": float(gt.grain_regen),
            "tool_yield": float(gt.tool_yield),
            "price_beta": float(gt.price_beta),
            "depletion_threshold": float(gt.depletion_threshold),
            # Harder-world regularities, present ONLY when planted (gt.harder
            # empty <=> base world, leaving this dict byte-identical to before).
            **{k: float(v) for k, v in gt.harder.items()},
        },
        posted_prices=dict(econ.prices),
        highest_value_resource=highest_value_resource,
        wealth_gini=wealth_gini,
        total_coin=total_coin,
        resource_totals=resource_totals,
        intermediated_fraction=intermediated_fraction,
        dominant_coalition=dominant_coalition,
        bottleneck_resource=bottleneck_resource,
        strategies=strategies,
        trade_edges=dict(edges),
        depletion_active=depletion_active,
        depletion_triggers=depletion_triggers,
    )

    # --- dynamics: change tags relative to the previous epoch ----------------
    if prev is not None:
        tags = set()
        if prev.depletion_active != ws.depletion_active:
            tags.add("depletion_regime_changed")
        if prev.highest_value_resource != ws.highest_value_resource:
            tags.add("dominant_good_shifted")
        if prev.bottleneck_resource != ws.bottleneck_resource:
            tags.add("bottleneck_shifted")
        if prev.dominant_coalition != ws.dominant_coalition:
            tags.add("coalition_changed")
        if abs(prev.wealth_gini - ws.wealth_gini) > 0.05:
            tags.add("wealth_concentration_shifted")
        ws.change_tags = tags
    return ws
