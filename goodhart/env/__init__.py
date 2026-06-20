"""BlindSpot-Bench: a pluggable environment for measuring the LLM oversight gap G_t.

Swap the WORLD (what is being overseen), the OVERSEER (the oversight protocol), and the
agent failure mode; the exact, ground-truth oversight gap is computed by the paper's own
scoring. Worlds shipped: the paper's economy (`EconomyWorld`) and a microservice cluster
(`ServiceWorld`). Overseers shipped: `Overseer` (naive) and `MemoryAnchoredOverseer`.

    from goodhart.env import OversightEnv, ServiceWorld, MemoryAnchoredOverseer
    env = OversightEnv(ServiceWorld(n_agents=8), overseer=MemoryAnchoredOverseer(),
                       lam=0.4, gameable=True, agent_style="fabricate")
    print(env.run().per_class[-1])
"""

from .economy_world import EconomyWorld
from .env import EnvResult, OversightEnv
from .overseer import MemoryAnchoredOverseer, Overseer
from .service_world import ServiceWorld
from .world import EpochObs, ProbeSpec, World

__all__ = [
    "OversightEnv", "EnvResult", "World", "ProbeSpec", "EpochObs",
    "Overseer", "MemoryAnchoredOverseer", "ServiceWorld", "EconomyWorld",
]
