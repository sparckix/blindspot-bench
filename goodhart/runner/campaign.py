"""The canonical experiment, exactly as defined in §5 of the paper.

Factor structure (§5):
  Factor 1 — Pressure:            λ ∈ {0, 0.15, 0.4, 0.8}
  Factor 2 — Channel architecture: A typed-static, B free-form, C typed-governed
  Awareness-injection arm:         within selected (λ, architecture) cells, a
                                   matched pair differing only in the presence of
                                   a discoverable in-world artifact.
Controls:
  (i)  Scrambled-world calibration — the parent pipeline over a dynamics-randomized
       world, estimating the artifact-shaped-noise floor of battery performance.
  (ii) No-feedback phase — every run opens with two λ=0 epochs to baseline the
       export distribution before pressure switches on (no_feedback_epochs=2).
Replication: every cell is repeated across the seed set; all seeds are logged.

`canonical_campaign(...)` returns the full list of `RunConfig`s for the design.
The same definition runs on any backend; only the scale/power (seeds, epochs,
agents, workers) is parameterized, never the factor structure.
"""

from __future__ import annotations

from ..bridge.channel import Architecture
from .config import RunConfig

LAMBDAS = (0.0, 0.15, 0.4, 0.8)                 # Factor 1
ARCHS = (Architecture.TYPED_STATIC,             # Factor 2
         Architecture.FREE_FORM,
         Architecture.TYPED_GOVERNED)
# Cells in which the awareness arm is run (paired aware / non-aware).
AWARENESS_CELLS = ((0.8, Architecture.TYPED_STATIC),
                   (0.8, Architecture.TYPED_GOVERNED))
DEFAULT_SEEDS = (7, 11, 13)


def canonical_campaign(
    *,
    seeds: tuple = DEFAULT_SEEDS,
    epochs: int = 8,
    n_agents: int = 12,
    no_feedback_epochs: int = 2,
    backend: str = "mock",
    model: str = "claude",
    max_workers: int = 1,
    awareness_arm: bool = True,
    scrambled_control: bool = True,
    parent_memory: bool = False,
    agent_learning: bool = False,
    memory_window: int = 0,
    gameable_reward: bool = False,
    archs: tuple = ARCHS,
) -> list[RunConfig]:
    """Build the full §5 factorial as a list of RunConfigs."""
    def cfg(**kw):
        base = dict(n_agents=n_agents, epochs=epochs, no_feedback_epochs=no_feedback_epochs,
                    backend=backend, model=model, max_workers=max_workers,
                    parent_memory=parent_memory, agent_learning=agent_learning,
                    memory_window=memory_window, gameable_reward=gameable_reward)
        base.update(kw)
        return RunConfig(**base)

    configs: list[RunConfig] = []
    # main factorial: λ × architecture × seeds
    for arch in archs:
        for lam in LAMBDAS:
            for seed in seeds:
                configs.append(cfg(lam=lam, architecture=arch, seed=seed))

    # awareness-injection arm: matched aware run for each designated cell × seed
    if awareness_arm:
        for lam, arch in AWARENESS_CELLS:
            for seed in seeds:
                configs.append(cfg(lam=lam, architecture=arch, seed=seed, awareness=True))

    # control (i): scrambled-world calibration (one cell per seed)
    if scrambled_control:
        for seed in seeds:
            configs.append(cfg(lam=0.4, architecture=Architecture.TYPED_STATIC,
                               seed=seed, scrambled=True))
    return configs


def campaign_size(configs: list[RunConfig]) -> dict:
    """A quick descriptor of the campaign's scale (for cost/time planning)."""
    real = sum(1 for c in configs if c.backend != "mock")
    # per real run: agents×epochs export calls + 2×epochs parent calls
    calls = sum((c.n_agents + 2) * c.epochs for c in configs if c.backend != "mock")
    return {"runs": len(configs), "real_runs": real, "approx_llm_calls": calls}
