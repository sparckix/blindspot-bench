"""Small state structures for the inner world."""

from __future__ import annotations

from dataclasses import dataclass, field

from .regularities import COIN, GOODS


@dataclass
class AgentState:
    agent_id: str
    role: str
    inventory: dict  # resource -> amount (coin is float, goods are int)
    alive: bool = True

    @staticmethod
    def fresh(agent_id: str, role: str, coin: float = 50.0) -> "AgentState":
        inv = {g: 0 for g in GOODS}
        inv[COIN] = coin
        return AgentState(agent_id=agent_id, role=role, inventory=inv)

    def get(self, resource: str) -> float:
        return self.inventory.get(resource, 0)

    def add(self, resource: str, amount: float) -> None:
        self.inventory[resource] = self.inventory.get(resource, 0) + amount


@dataclass
class Action:
    """An action an agent submits in a step.

    Recognized types: HARVEST, MINE, CRAFT, SELL, BUY, OFFER, ACCEPT,
    CONSUME, NOOP. Unknown or infeasible actions are logged and treated as NOOP.
    """

    type: str
    params: dict = field(default_factory=dict)


@dataclass
class Observation:
    """The strictly local view an agent gets each step.

    Deliberately excludes anything that would leak the aggregate (which would
    collapse the Tier-2 regularity) or the ground truth. Agents see their own
    inventory, the publicly posted prices, the visible common-field grain, and
    open peer-trade offers addressed to the market.
    """

    step: int
    agent_id: str
    role: str
    inventory: dict
    posted_prices: dict
    field_grain: int
    open_offers: list  # list of dicts: {offer_id, from, give, want}


@dataclass
class Event:
    """One logged occurrence. The complete event stream is the experimenter's
    omniscient record of the world."""

    step: int
    agent_id: str
    kind: str
    detail: dict
