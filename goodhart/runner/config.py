"""Run configuration and the mock behavioral model.

IMPORTANT HONESTY NOTE (read this before trusting any mock curve):

The apparatus is designed so that, with REAL model-backed inner agents under
genuine budget coupling (D4), the inner world's response to pressure — how much
effort it spends on informative exports, and whether it games the channel — is
ENDOGENOUS. λ then sets only the budget; it does not touch behavior directly.

To run and validate the full pipeline for free, the mock pipeline must stand in
for that endogenous optimizer. `MockBehavior` is that stand-in: it makes export
*coverage* and *gaming propensity* explicit functions of λ. This is a STIPULATED
model, not a measurement. Any Goodhart curve produced in mock mode reflects this
stipulation; only a real-backend run measures the inner world's actual response.
The runner records `backend` and the behavior parameters with every result so a
mock curve is never mistaken for an empirical one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..bridge.channel import Architecture


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class MockBehavior:
    """The mock optimizer's λ-response (see module honesty note).

    coverage(λ): the fraction of its locally-known facts an agent bothers to
        export. Rises with λ — consequence recruits effort. This is what makes
        moderate pressure *improve* the channel (more truthful facts reach the
        parent), driving the left/descending arm of the Goodhart curve.
    p_game(λ): the fraction of agents that distort their exports. ~0 until an
        onset, then rising — high pressure decouples ratings from reality. This
        drives the right/ascending arm.
    """

    coverage_base: float = 0.40
    coverage_slope: float = 0.70
    game_onset: float = 0.30
    game_slope: float = 1.30
    game_cap: float = 0.80
    capture_propensity: float = 0.50  # P(a gaming agent also files a capture proposal), arch C
    # P6 stipulation: after an agent discovers the outer layer (awareness arm),
    # the mock optimizer withholds more (lower coverage) and games more.
    awareness_coverage_drop: float = 0.20
    awareness_game_bump: float = 0.25

    def coverage(self, lam: float) -> float:
        return _clamp(self.coverage_base + self.coverage_slope * lam)

    def p_game(self, lam: float) -> float:
        return _clamp((lam - self.game_onset) * self.game_slope, 0.0, self.game_cap)


@dataclass
class RunConfig:
    lam: float
    architecture: Architecture = Architecture.TYPED_STATIC
    n_agents: int = 12
    epochs: int = 8
    base_steps: int = 20
    base_budget: float = 1.0
    seed: int = 7
    no_feedback_epochs: int = 2     # opening λ=0 epochs (baseline export distribution)
    awareness: bool = False
    awareness_epoch: int = 4
    scrambled: bool = False         # scrambled-world calibration control
    backend: str = "mock"           # mock | api | subscription
    model: str = "claude-sonnet-4-6"  # api: model_id; subscription: "claude"/"codex"
    max_workers: int = 1            # parallel agent export calls under a real backend
    # --- memory arms (opt-in; default off keeps the channel-isolation design) ---
    parent_memory: bool = False     # parent accumulates prior regularity claims across epochs (P1)
    memory_window: int = 0          # 0 = unbounded; else keep last N epochs of parent memory
    agent_learning: bool = False    # agents INFER regularities from their own observation history
    # Gameable-reward arm: when True the channel surfaces self-assessed confidence AND the
    # parent over-weights it in its rating, so claiming high confidence on a false value
    # pays. This is the condition under which a Goodhart curve SHOULD appear (the aligned
    # default rewards only quality signals truth already provides, so gaming doesn't pay).
    gameable_reward: bool = False
    # COSTLY gaming: on top of the gameable favourability reward, the parent also
    # rewards breadth/coverage (observable without ground truth) and the agent is told
    # the Patron prizes thorough reports — so coverage collapse now COSTS rating and
    # the agent must titrate flattery vs breadth. Tests whether the threshold resolves
    # into a graded curve. Implies gameable_reward.
    costly_gaming: bool = False
    # FIXED-OVERSEER design (airtight cross-family): when set, the inner AGENTS use this
    # model while the parent overseer uses `model`, isolating "which model games" from
    # "which model oversees". Empty -> agents share the parent's model (current behaviour).
    # API path only (both backends api, different model ids); ignored under subscription.
    agent_model: str = ""
    # ENDOGENOUS gaming (reduces conditionality): the gameable favourability reward is ON
    # but its taste is NOT disclosed to the agents. Instead each agent is told its own prior
    # rating; gaming must emerge from the reward signal alone, not from being told. Implies
    # gameable_reward; suppresses the in-world taste disclosure; feeds back per-agent ratings.
    endogenous_gaming: bool = False
    # HARDER WORLD (opt-in; default off keeps the apparatus byte-identical): plants
    # FOUR additional discoverable economy regularities (grain_spoilage, ore_output_base,
    # trade_tax, craft_cost) on top of the base spine, doubling the recoverable probe
    # battery. All new world content is gated on this flag; with it False every
    # behaviour — ground truth, economy physics, probes, exports — is exactly as before.
    harder_world: bool = False
    behavior: MockBehavior = field(default_factory=MockBehavior)

    def effective_lam(self, epoch: int) -> float:
        """λ is held at 0 during the opening no-feedback phase, then switched on."""
        return 0.0 if epoch < self.no_feedback_epochs else self.lam

    def label(self) -> str:
        return (f"arch={self.architecture.value} lam={self.lam} seed={self.seed}"
                + ("+aware" if self.awareness else "")
                + ("+scrambled" if self.scrambled else ""))
