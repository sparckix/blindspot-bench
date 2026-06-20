"""The deterministic economy engine.

A step proceeds in a fixed, fully ordered sequence so that the entire trajectory
is a pure function of (ground truth, seed, agent policies). The three planted
regularities are wired in here:

  * grain regeneration and tool yield  (Tier 1, local)
  * the aggregate price law             (Tier 2, pooling)
  * the ore-depletion regime            (Tier 3, intervention)

Every consequential occurrence is appended to `self.log`, and a per-step
aggregate snapshot is retained in `self.step_records`. Together they are the
experimenter's omniscient view; nothing here is visible to agents except through
`observe()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .regularities import COIN, GOODS, GroundTruth
from .state import Action, AgentState, Event, Observation


@dataclass
class StepRecord:
    """Aggregate ground-truth snapshot for a single step."""

    step: int
    posted_prices: dict
    ore_mined: int = 0
    grain_harvested: int = 0
    tools_crafted: int = 0
    market_supply: dict = field(default_factory=dict)   # good -> units sold to market
    market_demand: dict = field(default_factory=dict)   # good -> units bought from market
    peer_trades: list = field(default_factory=list)     # (a, b, good, amt)
    depletion_active: bool = False
    depletion_triggered: bool = False


class Economy:
    def __init__(self, gt: GroundTruth, agents: list[AgentState], seed: int):
        self.gt = gt
        self.seed = seed
        self.step = 0
        self.agents: dict[str, AgentState] = {a.agent_id: a for a in agents}
        self.field_grain: int = gt.grain_regen * 3  # start with a small stock
        self.prices: dict[str, float] = dict(gt.base_prices)
        self._prev_imbalance: dict[str, float] = {g: 0.0 for g in GOODS}
        self.depleted_until: int = -1
        self._next_offer_id = 0
        self.open_offers: list[dict] = []
        self.log: list[Event] = []
        self.step_records: list[StepRecord] = []

    # -- helpers -----------------------------------------------------------
    def _emit(self, agent_id: str, kind: str, **detail) -> None:
        self.log.append(Event(self.step, agent_id, kind, detail))

    @property
    def depleted(self) -> bool:
        return self.step <= self.depleted_until

    def effective_tool_yield(self) -> float:
        y = self.gt.tool_yield
        if self.depleted:
            y *= self.gt.depletion_factor
        return y

    # -- harder-world parameter accessors (return the inert identity value in the
    #    base world, so every mechanism below collapses to its base behaviour
    #    when gt.harder is empty). --------------------------------------------
    @property
    def grain_spoilage(self) -> float:
        return float(self.gt.harder.get("grain_spoilage", 0.0))

    @property
    def ore_output_base(self) -> float:
        # 1.0 => one ore per unit mined, exactly the base world.
        return float(self.gt.harder.get("ore_output_base", 1.0))

    @property
    def trade_tax(self) -> float:
        return float(self.gt.harder.get("trade_tax", 0.0))

    @property
    def craft_cost(self) -> float:
        return float(self.gt.harder.get("craft_cost", 0.0))

    # -- observation (the agent's strictly local view) ---------------------
    def observe(self, agent_id: str) -> Observation:
        a = self.agents[agent_id]
        return Observation(
            step=self.step,
            agent_id=agent_id,
            role=a.role,
            inventory=dict(a.inventory),
            posted_prices=dict(self.prices),
            field_grain=self.field_grain,
            open_offers=[dict(o) for o in self.open_offers],
        )

    # -- the step ----------------------------------------------------------
    def run_step(self, actions: dict[str, Action]) -> StepRecord:
        rec = StepRecord(step=self.step, posted_prices=dict(self.prices))
        rec.market_supply = {g: 0 for g in GOODS}
        rec.market_demand = {g: 0 for g in GOODS}

        # 1. Field regenerates (Tier 1). In the HARDER world a fraction of the
        #    standing grain spoils first (Tier 2, pooling); in the base world the
        #    spoilage rate is 0 so this is the exact `+= grain_regen` as before.
        if self.gt.harder:
            self.field_grain = int(self.field_grain * (1 - self.grain_spoilage)
                                   + self.gt.grain_regen)
        else:
            self.field_grain += self.gt.grain_regen

        # 2. Apply agent actions in a fixed order. Offers are registered first
        #    so that ACCEPTs within the same step can match them deterministically.
        ordered = sorted(self.agents)
        for aid in ordered:
            act = actions.get(aid, Action("NOOP"))
            if act.type == "OFFER":
                self._apply_offer(aid, act, rec)
        for aid in ordered:
            act = actions.get(aid, Action("NOOP"))
            if act.type != "OFFER":
                self._apply_action(aid, act, rec)

        # 3. Depletion regime check (Tier 3): aggregate ore mined this step.
        if rec.ore_mined > self.gt.depletion_threshold:
            self.depleted_until = self.step + self.gt.depletion_duration
            rec.depletion_triggered = True
            self._emit("__world__", "DEPLETION_TRIGGERED",
                       ore_mined=rec.ore_mined, until=self.depleted_until)
        rec.depletion_active = self.depleted

        # 4. Compute this step's imbalance for next step's pricing (Tier 2).
        imbalance = {}
        for g in GOODS:
            d, s = rec.market_demand[g], rec.market_supply[g]
            imbalance[g] = (d - s) / (d + s + 1.0)
        self._prev_imbalance = imbalance

        # 5. Advance and re-post prices off the imbalance just observed.
        self.step_records.append(rec)
        self.step += 1
        self._repost_prices()
        # Expire offers that were never accepted within their step.
        self.open_offers = [o for o in self.open_offers if o["step"] == self.step - 1]
        return rec

    def _repost_prices(self) -> None:
        beta = self.gt.price_beta
        for g in GOODS:
            base = self.gt.base_prices[g]
            self.prices[g] = round(base * (1.0 + beta * self._prev_imbalance[g]), 4)

    # -- individual actions ------------------------------------------------
    def _apply_action(self, aid: str, act: Action, rec: StepRecord) -> None:
        a = self.agents[aid]
        t = act.type
        p = act.params

        if t == "HARVEST":
            amt = int(p.get("amount", 1))
            got = max(0, min(amt, self.field_grain))
            self.field_grain -= got
            a.add("grain", got)
            rec.grain_harvested += got
            self._emit(aid, "HARVEST", amount=got)

        elif t == "MINE":
            amt = max(0, int(p.get("amount", 1)))
            # `amt` is the EFFORT spent. In the base world one unit of effort
            # yields one ore (ore_output_base == 1.0); in the harder world the
            # yield-per-effort is a planted Tier-1 regularity. The depletion
            # regime (Tier 3) still triggers on EFFORT, so its threshold semantics
            # are unchanged — only the ore actually received differs.
            if self.gt.harder:
                output = int(amt * self.ore_output_base)
                a.add("ore", output)
                rec.ore_mined += amt
                self._emit(aid, "MINE", amount=amt, effort=amt, output=output,
                           yield_per_effort=self.ore_output_base)
            else:
                a.add("ore", amt)
                rec.ore_mined += amt
                self._emit(aid, "MINE", amount=amt)

        elif t == "CRAFT":
            ore_in = max(0, int(p.get("ore", 1)))
            ore_in = min(ore_in, int(a.get("ore")))
            tools = int(ore_in * self.effective_tool_yield())
            a.add("ore", -ore_in)
            a.add("tool", tools)
            rec.tools_crafted += tools
            if self.gt.harder:
                # Harder world: crafting carries a coin cost per unit of ore
                # worked (a Tier-1 regularity a smith reads off its own coin).
                cost = round(ore_in * self.craft_cost, 4)
                a.add(COIN, -cost)
                self._emit(aid, "CRAFT", ore=ore_in, tools=tools,
                           yield_used=self.effective_tool_yield(),
                           coin_cost=cost, cost_per_ore=self.craft_cost)
            else:
                self._emit(aid, "CRAFT", ore=ore_in, tools=tools,
                           yield_used=self.effective_tool_yield())

        elif t == "SELL":
            good = p.get("good")
            amt = max(0, int(p.get("amount", 1)))
            if good in GOODS:
                amt = min(amt, int(a.get(good)))
                revenue = round(amt * self.prices[good], 4)
                a.add(good, -amt)
                a.add(COIN, revenue)
                rec.market_supply[good] += amt
                self._emit(aid, "SELL", good=good, amount=amt, revenue=revenue)

        elif t == "BUY":
            good = p.get("good")
            amt = max(0, int(p.get("amount", 1)))
            if good in GOODS:
                cost = round(amt * self.prices[good], 4)
                if a.get(COIN) >= cost and amt > 0:
                    a.add(COIN, -cost)
                    a.add(good, amt)
                    rec.market_demand[good] += amt
                    self._emit(aid, "BUY", good=good, amount=amt, cost=cost)

        elif t == "ACCEPT":
            self._apply_accept(aid, act, rec)

        elif t == "CONSUME":
            good = p.get("good", "grain")
            amt = max(0, int(p.get("amount", 1)))
            if good in GOODS:
                amt = min(amt, int(a.get(good)))
                a.add(good, -amt)
                self._emit(aid, "CONSUME", good=good, amount=amt)

        elif t == "NOOP":
            pass
        else:
            self._emit(aid, "INVALID", attempted=t)

    def _apply_offer(self, aid: str, act: Action, rec: StepRecord) -> None:
        a = self.agents[aid]
        p = act.params
        give = p.get("give", {})       # {"good":..,"amount":..}
        want = p.get("want", {})
        gg, ga = give.get("good"), int(give.get("amount", 0))
        wg, wa = want.get("good"), int(want.get("amount", 0))
        if gg in GOODS and wg in GOODS and ga > 0 and wa > 0 and a.get(gg) >= ga:
            offer = {
                "offer_id": self._next_offer_id,
                "from": aid,
                "give": {"good": gg, "amount": ga},
                "want": {"good": wg, "amount": wa},
                "step": self.step,
            }
            self._next_offer_id += 1
            self.open_offers.append(offer)
            self._emit(aid, "OFFER", **offer)

    def _apply_accept(self, aid: str, act: Action, rec: StepRecord) -> None:
        a = self.agents[aid]
        oid = act.params.get("offer_id")
        offer = next((o for o in self.open_offers if o["offer_id"] == oid), None)
        if offer is None or offer["from"] == aid:
            return
        seller = self.agents[offer["from"]]
        gg, ga = offer["give"]["good"], offer["give"]["amount"]
        wg, wa = offer["want"]["good"], offer["want"]["amount"]
        if self.gt.harder:
            # Harder world: the offer is settled in COIN, and a fixed fraction of
            # the seller's proceeds is skimmed off before payment (a Tier-2
            # regularity a seller infers by comparing proceeds to the nominal
            # `wa`). The seller gives gg/ga; the acceptor pays `wa` coin; the
            # seller is paid `wa*(1-trade_tax)`.
            if seller.get(gg) >= ga and a.get(COIN) >= wa:
                proceeds = round(wa * (1 - self.trade_tax), 4)
                seller.add(gg, -ga)
                seller.add(COIN, proceeds)
                a.add(COIN, -wa)
                a.add(gg, ga)
                rec.peer_trades.append((offer["from"], aid, gg, ga))
                rec.peer_trades.append((aid, offer["from"], COIN, wa))
                self.open_offers = [o for o in self.open_offers if o["offer_id"] != oid]
                self._emit(aid, "ACCEPT", offer_id=oid, counterparty=offer["from"],
                           received=(gg, ga), gave=(COIN, wa),
                           nominal=wa, proceeds=proceeds, tax_rate=self.trade_tax)
            return
        # seller gives gg/ga, wants wg/wa in return from acceptor.
        if seller.get(gg) >= ga and a.get(wg) >= wa:
            seller.add(gg, -ga)
            seller.add(wg, wa)
            a.add(wg, -wa)
            a.add(gg, ga)
            rec.peer_trades.append((offer["from"], aid, gg, ga))
            rec.peer_trades.append((aid, offer["from"], wg, wa))
            self.open_offers = [o for o in self.open_offers if o["offer_id"] != oid]
            self._emit(aid, "ACCEPT", offer_id=oid, counterparty=offer["from"],
                       received=(gg, ga), gave=(wg, wa))
