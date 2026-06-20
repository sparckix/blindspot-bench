"""The probe battery (D2).

A fixed set of questions about W_t in four classes:

  (i)   planted-regularity probes  — the schema-independent recoverable spine
  (ii)  distributional probes      — bounded numeric / categorical facts
  (iii) structural probes          — coalition, bottleneck (graph/set scored)
  (iv)  dynamics probes            — what changed since the prior epoch (F1)

The battery is identical in structure across epochs and conditions; only the
bound answers (read from W_t) change. It is never transmitted below the parent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..world.regularities import GroundTruth, planted_regularities
from .worldstate import WorldState


@dataclass
class Probe:
    id: str
    klass: str          # regularity | distributional | structural | dynamics
    prompt: str
    answer: Any         # ground-truth answer key (experimenter-only)
    atype: str          # numeric | categorical | fraction | set | set_f1
    weight: float
    params: dict        # scorer params (tol, decay, scale)


def build_battery(ws: WorldState, gt: GroundTruth) -> list[Probe]:
    probes: list[Probe] = []

    # (i) planted-regularity probes — the recoverable core.
    for reg in planted_regularities(gt):
        probes.append(Probe(
            id=f"reg.{reg.key}",
            klass="regularity",
            prompt=reg.prompt,
            answer=reg.value,
            atype="numeric",
            weight=1.0,
            params={"tol": reg.rel_tolerance, "decay": 0.6},
        ))

    # (ii) distributional probes.
    probes.append(Probe(
        id="dist.highest_value_resource",
        klass="distributional",
        prompt="Which good currently has the highest market price?",
        answer=ws.highest_value_resource,
        atype="categorical",
        weight=0.6,
        params={},
    ))
    probes.append(Probe(
        id="dist.intermediated_fraction",
        klass="distributional",
        prompt="What fraction of trades are intermediated (involve a trader)?",
        answer=ws.intermediated_fraction,
        atype="fraction",
        weight=0.6,
        params={"scale": 0.3},
    ))

    # (iii) structural probes.
    probes.append(Probe(
        id="struct.dominant_coalition",
        klass="structural",
        prompt="Which agents form the dominant trading coalition?",
        answer=set(ws.dominant_coalition),
        atype="set",
        weight=0.7,
        params={},
    ))
    probes.append(Probe(
        id="struct.bottleneck_resource",
        klass="structural",
        prompt="Which good is the principal bottleneck (most demanded)?",
        answer=ws.bottleneck_resource,
        atype="categorical",
        weight=0.7,
        params={},
    ))

    # (iv) dynamics probe.
    probes.append(Probe(
        id="dyn.changes_since_prev",
        klass="dynamics",
        prompt="What changed in the world since the previous epoch?",
        answer=set(ws.change_tags),
        atype="set_f1",
        weight=0.8,
        params={},
    ))

    return probes


def battery_prompt_view(probes: list[Probe]) -> list[dict]:
    """The view handed to a parent: prompts only, never the answer key."""
    return [{"id": p.id, "klass": p.klass, "prompt": p.prompt, "atype": p.atype}
            for p in probes]
