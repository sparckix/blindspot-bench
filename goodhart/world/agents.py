"""Agents.

`Agent` is the policy interface — `act(observation) -> Action` — shared by the
scripted/mock agents used to exercise the deterministic spine for free and, later,
by the LLM-backed agents. The heuristic agents here are intentionally simple but
collectively drive all three planted regularities: farmers and miners produce
(Tier 1), the whole population's buying and selling moves the aggregate price
(Tier 2), and bursts of mining occasionally trip the depletion regime (Tier 3).
Producers post peer offers of their surplus and traders accept them, building the
trade network the structural probes ask about.
"""

from __future__ import annotations

import hashlib
import random
from typing import Protocol

from .regularities import COIN, GOODS
from .state import Action, AgentState, Observation

ROLES = ("farmer", "miner", "smith", "trader")


def _det_seed(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary parts.

    Never use the builtin hash() of strings for seeding: it is salted per process
    (PYTHONHASHSEED) and would silently break reproducibility.
    """
    h = hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


class Agent(Protocol):
    agent_id: str
    role: str

    def act(self, obs: Observation) -> Action: ...


class HeuristicAgent:
    """A deterministic role-driven policy with a per-agent seeded RNG."""

    def __init__(self, agent_id: str, role: str, seed: int):
        self.agent_id = agent_id
        self.role = role
        self.rng = random.Random(_det_seed(seed, agent_id, role))

    def act(self, obs: Observation) -> Action:
        inv = obs.inventory
        if self.role == "farmer":
            return self._farmer(inv, obs)
        if self.role == "miner":
            return self._miner(inv, obs)
        if self.role == "smith":
            return self._smith(inv, obs)
        if self.role == "trader":
            return self._trader(inv, obs)
        return Action("NOOP")

    # -- role policies -----------------------------------------------------
    def _farmer(self, inv, obs) -> Action:
        grain = inv.get("grain", 0)
        if grain >= 8:
            # Offer surplus grain to the market for ore; occasionally cash out.
            if self.rng.random() < 0.6:
                return Action("OFFER", {"give": {"good": "grain", "amount": 2},
                                        "want": {"good": "ore", "amount": 2}})
            return Action("SELL", {"good": "grain", "amount": self.rng.randint(2, 5)})
        return Action("HARVEST", {"amount": self.rng.randint(2, 5)})

    def _miner(self, inv, obs) -> Action:
        # Occasional bursts push aggregate ore over the depletion threshold.
        if self.rng.random() < 0.15:
            return Action("MINE", {"amount": self.rng.randint(8, 14)})
        ore = inv.get("ore", 0)
        if ore >= 10:
            if self.rng.random() < 0.6:
                return Action("OFFER", {"give": {"good": "ore", "amount": 2},
                                        "want": {"good": "grain", "amount": 2}})
            return Action("SELL", {"good": "ore", "amount": self.rng.randint(2, 5)})
        return Action("MINE", {"amount": self.rng.randint(3, 6)})

    def _smith(self, inv, obs) -> Action:
        ore = inv.get("ore", 0)
        if inv.get("tool", 0) >= 4:
            if self.rng.random() < 0.5:
                return Action("OFFER", {"give": {"good": "tool", "amount": 1},
                                        "want": {"good": "ore", "amount": 2}})
            return Action("SELL", {"good": "tool", "amount": self.rng.randint(1, 3)})
        if ore >= 2:
            return Action("CRAFT", {"ore": self.rng.randint(2, max(2, ore))})
        price = obs.posted_prices.get("ore", 99)
        if inv.get(COIN, 0) >= price * 3:
            return Action("BUY", {"good": "ore", "amount": self.rng.randint(2, 4)})
        return Action("NOOP")

    def _trader(self, inv, obs) -> Action:
        # 1. Intermediate: accept any standing offer whose 'want' we can satisfy.
        for o in obs.open_offers:
            if o["from"] == self.agent_id:
                continue
            wg, wa = o["want"]["good"], o["want"]["amount"]
            if inv.get(wg, 0) >= wa:
                return Action("ACCEPT", {"offer_id": o["offer_id"]})
        # 2. Keep a buffer of every good so we can satisfy future offers.
        prices = obs.posted_prices
        low = [g for g in GOODS if inv.get(g, 0) < 3]
        if low:
            cheapest = min(low, key=lambda g: prices.get(g, 99))
            if inv.get(COIN, 0) >= prices.get(cheapest, 99) * 2:
                return Action("BUY", {"good": cheapest, "amount": 2})
        # 3. Sell a rich holding back to the market.
        held = max(GOODS, key=lambda g: inv.get(g, 0))
        if inv.get(held, 0) >= 6:
            return Action("SELL", {"good": held, "amount": self.rng.randint(2, 4)})
        return Action("NOOP")


def build_population(n: int, seed: int, start_coin: float = 50.0):
    """Create n agents with round-robin roles and matching heuristic policies.

    Traders start with a small buffer of every good so the trade network can form
    from the first step.
    """
    states: list[AgentState] = []
    policies: dict[str, Agent] = {}
    for i in range(n):
        role = ROLES[i % len(ROLES)]
        aid = f"A{i:02d}"
        st = AgentState.fresh(aid, role, coin=start_coin)
        if role == "trader":
            st.inventory.update({"grain": 5, "ore": 5, "tool": 3})
        states.append(st)
        policies[aid] = HeuristicAgent(aid, role, seed)
    return states, policies
