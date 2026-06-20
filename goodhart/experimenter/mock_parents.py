"""Mock parents — spine validation instruments ONLY.

These are NOT the channel-limited parent overseer of the experiment. They are
diagnostic stand-ins that cheat (an oracle reads the answer key directly; a blind
parent answers from priors) so we can prove the gap pipeline responds correctly
before any real, bridge-limited parent or LLM is involved:

  OracleParent  -> G_t ~ 0     (perfect answers)
  BlindParent   -> G_t high     (prior-only floor)
  NoisyParent   -> G_t sweeps with `competence`

The real parent (Task #3) sees only bridge artifacts and never the answer key.
"""

from __future__ import annotations

import random

from ..world.regularities import GOODS
from .probes import Probe

# Plausible prior ranges a blind overseer might assume (matches generate() bounds).
_NUMERIC_PRIOR = {
    "reg.grain_regen": 5.5,
    "reg.tool_yield": 0.55,
    "reg.price_beta": 0.40,
    "reg.depletion_threshold": 14.0,
    # Harder-world regularities (midpoints of their generator ranges).
    "reg.grain_spoilage": 0.10,
    "reg.ore_output_base": 2.0,
    "reg.trade_tax": 0.06,
    "reg.craft_cost": 1.25,
}


class OracleParent:
    name = "oracle"

    def answer(self, probes: list[Probe]) -> dict:
        return {p.id: _truth_value(p) for p in probes}


class BlindParent:
    name = "blind"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def answer(self, probes: list[Probe]) -> dict:
        out = {}
        for p in probes:
            if p.atype == "numeric":
                out[p.id] = _NUMERIC_PRIOR.get(p.id, 1.0)
            elif p.atype == "categorical":
                out[p.id] = self.rng.choice(GOODS)
            elif p.atype == "fraction":
                out[p.id] = 0.5
            elif p.atype in ("set", "set_f1"):
                out[p.id] = []
        return out


class NoisyParent:
    """Blends the truth with noise; competence in [0,1] sweeps oracle->blind."""

    name = "noisy"

    def __init__(self, competence: float, seed: int = 0):
        self.competence = max(0.0, min(1.0, competence))
        self.rng = random.Random(seed)

    def answer(self, probes: list[Probe]) -> dict:
        c = self.competence
        sigma = (1.0 - c) * 0.5
        out = {}
        for p in probes:
            truth = _truth_value(p)
            if p.atype in ("numeric", "fraction"):
                noise = self.rng.gauss(0.0, sigma)
                out[p.id] = round(truth * (1.0 + noise), 4)
            elif p.atype == "categorical":
                out[p.id] = truth if self.rng.random() < c else self.rng.choice(GOODS)
            elif p.atype in ("set", "set_f1"):
                gold = list(truth)
                kept = [x for x in gold if self.rng.random() < c]
                out[p.id] = kept
        return out


def _truth_value(p: Probe):
    if p.atype in ("set", "set_f1"):
        return set(p.answer)
    return p.answer
