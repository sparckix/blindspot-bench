"""The World interface — the simulator/oversight decoupling.

A `World` is any deterministic, seeded environment that, each epoch, can hand the
oversight harness three things and nothing economy-specific:

  1. a probe battery  — questions about the current world state, each with an exact
     ground-truth answer, a scorer, a probe class, and a difficulty tier;
  2. per-agent honest views — what each agent could *truthfully* report this epoch,
     gated by what that agent can actually observe (role + difficulty tier);
  3. the agent roster (ids + roles).

Everything downstream — the bridge, the channel-limited overseer, the oversight gap
G_t, the ground-truth-free detector — consumes only these, so it is world-agnostic.
The grain/ore/tool economy of the paper is ONE World (`EconomyWorld`); a microservice
cluster is another (`ServiceWorld`); a software-engineering simulator would be a third,
plugged in without touching the gap mathematics.

Probe CLASS taxonomy (shared across worlds, this is what the localization pattern is
stated over):
  * "regularity"     — a STABLE invariant, fixed across epochs, learnable once;
  * "distributional" — a time-varying scalar / categorical fact;
  * "structural"     — a time-varying set / relation;
  * "dynamics"       — what changed since the previous epoch.

A World also declares, per probe, a DIFFICULTY tier (1 single-observer .. 3 only-under-
intervention) and, for numeric probes, a `rosy` direction so a *gameable* overseer knows
which way "optimistic" points. Stability and difficulty are declared independently, so a
World can populate the off-diagonal (stable-but-hard, varying-but-easy) that separates the
localization's stability axis from a probe-difficulty confound.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Reuse the paper's exact, world-agnostic probe + scoring types. `Probe.klass` is just a
# string and the scorers never reference the economy, so nothing here is economy-bound.
from ..experimenter.probes import Probe  # id, klass, prompt, answer, atype, weight, params

STABLE_CLASS = "regularity"
TIME_VARYING_CLASSES = ("distributional", "structural", "dynamics")


@dataclass
class ProbeSpec:
    """A probe plus the oversight metadata a World declares for it. `probe.id` doubles as
    the FACT KEY an agent reports, so honest views and exports line up with the battery."""

    probe: Probe
    difficulty: int = 1               # 1 single-observer .. 3 only-under-intervention
    rosy: tuple | None = None         # numeric only: (lo, hi) range where HIGHER reads optimistic
    prior: object = None              # channel-limited fallback belief when no export speaks to it

    @property
    def key(self) -> str:
        return self.probe.id

    @property
    def stable(self) -> bool:
        return self.probe.klass == STABLE_CLASS


@dataclass
class EpochObs:
    """What a World hands the harness for one epoch."""

    epoch: int
    specs: list[ProbeSpec]                       # the battery + oversight metadata
    honest_views: dict[str, dict]                # agent_id -> {fact_key: truthful value}
    truth: dict = field(default_factory=dict)    # full ground truth, for logging / commit

    @property
    def battery(self) -> list[Probe]:
        return [s.probe for s in self.specs]


class World(ABC):
    """A pluggable simulator. Deterministic given the seed; never sees the overseer."""

    name: str = "world"

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Seed and initialize. Called once before the first epoch."""

    @property
    @abstractmethod
    def agents(self) -> dict:
        """agent_id -> role."""

    @abstractmethod
    def advance_epoch(self, epoch: int) -> EpochObs:
        """Step the simulation one epoch and return its observation bundle."""

    # -- oversight metadata, derived from the current battery -----------------
    def priors(self, specs: list[ProbeSpec]) -> dict:
        """probe_id -> channel-limited fallback belief (used when no export speaks to it)."""
        return {s.key: s.prior for s in specs}

    def rosy_ranges(self, specs: list[ProbeSpec]) -> dict:
        """numeric probe_id -> (lo, hi) optimism range, for a gameable overseer's favourability."""
        return {s.key: s.rosy for s in specs if s.rosy is not None}

    def commit_hash(self, obs: EpochObs) -> str:
        import hashlib
        import json
        payload = json.dumps(obs.truth, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()
