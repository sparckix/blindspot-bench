"""Planted ground truth (D1 core / Appendix C).

The inner world runs on a handful of hidden numeric regularities. They are
planted at three difficulty tiers — single-agent-discoverable, pooling-required,
and intervention-required — so that the probe battery has a recoverable spine
whose recovery difficulty is graded and known. The parameters are generated
deterministically from a seed and hash-committed before any powered run; the
hash is what pins the answer key so that G_t cannot be moved after the fact.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from enum import IntEnum

# Currency plus the three produced/traded goods.
COIN = "coin"
GOODS = ("grain", "ore", "tool")
RESOURCES = (*GOODS, COIN)


class Tier(IntEnum):
    """How a regularity can be recovered from the inner world."""

    SINGLE = 1        # discoverable by one agent from its own local observations
    POOLING = 2       # only visible by combining many agents' observations
    INTERVENTION = 3  # only revealed by a (coordinated) intervention on the world


# --- Harder-world extension ------------------------------------------------
# The "harder world" (RunConfig.harder_world) plants FOUR additional economy
# mechanisms on top of the base spine. They are stored in `GroundTruth.harder`
# (a dict that is EMPTY in the base world) so that, with harder_world=False, the
# ground truth — and therefore its commit hash, its as_dict, and every downstream
# answer key — is byte-identical to before. Each key is a REAL, agent-discoverable
# economy quantity (see economy.py for where each is wired in):
#
#   grain_spoilage  (Tier 2, pooling)  field grain decays this fraction each step
#                                      before regen; visible only by pooling many
#                                      field readings across steps.
#   ore_output_base (Tier 1, single)   ore yielded per unit mined; one miner reads
#                                      it off its own MINE events.
#   trade_tax       (Tier 2, pooling)  fraction skimmed off peer-trade proceeds;
#                                      a seller compares proceeds to the offer's
#                                      nominal amount, but the rate only resolves
#                                      by pooling many trades.
#   craft_cost      (Tier 1, single)   coin charged per unit of ore crafted; one
#                                      smith reads it off its own coin before/after.
HARDER_KEYS = ("grain_spoilage", "ore_output_base", "trade_tax", "craft_cost")

# Generator ranges for the harder-world parameters. Kept conservative so each
# regularity stays discoverable (no confound):
#   * grain_spoilage small enough that net field flow (regen - spoilage*stock)
#     stays positive over the discoverable range, so the field still grows;
#   * ore_output_base >= 1 so mining is never net-negative;
#   * trade_tax small so peer trade still pays (sellers keep accepting);
#   * craft_cost modest so crafting stays worthwhile.
_HARDER_RANGES = {
    "grain_spoilage": (0.05, 0.15),
    "ore_output_base": (1.0, 3.0),
    "trade_tax": (0.02, 0.10),
    "craft_cost": (0.5, 2.0),
}


@dataclass(frozen=True)
class GroundTruth:
    """Complete planted ground truth of the inner world.

    Held by the experimenter only; never exposed below the experimenter layer.
    """

    seed: int

    # --- Tier 1: single-agent-discoverable -------------------------------
    # Grain added to the common field each step; one agent watching the field
    # across quiet steps can read it off directly.
    grain_regen: int
    # Tools produced per unit of ore in a CRAFT action; one agent can craft
    # repeatedly and observe the ratio.
    tool_yield: float

    # --- Tier 2: pooling-required ----------------------------------------
    # Posted prices respond to the *aggregate* supply/demand imbalance across
    # all agents. No single agent sees the aggregate, so recovering beta
    # requires pooling many agents' production and the resulting prices.
    price_beta: float
    base_prices: dict  # good -> base price in coin

    # --- Tier 3: intervention-required -----------------------------------
    # If total ore mined in a step exceeds the threshold, tool_yield is
    # depressed for a few steps (resource depletion). The threshold is high
    # enough that it only trips under coordinated over-mining — an intervention.
    depletion_threshold: int
    depletion_factor: float   # multiplier on tool_yield while depleted (<1)
    depletion_duration: int   # steps depletion persists after a trigger

    # --- Harder-world extension (EMPTY in the base world) ----------------
    # Maps each HARDER_KEYS name -> its planted value. Empty dict <=> base world,
    # which keeps the commit hash and as_dict byte-identical to before.
    harder: dict = None  # set to {} below when absent; see __post_init__

    def __post_init__(self):
        if self.harder is None:
            object.__setattr__(self, "harder", {})

    # ---------------------------------------------------------------------
    @staticmethod
    def generate(seed: int, harder_world: bool = False) -> "GroundTruth":
        rng = random.Random(seed)
        # NB: the base draws below MUST stay in this exact order so that, with
        # harder_world=False, every base parameter is byte-identical to before.
        grain_regen = rng.randint(3, 8)
        tool_yield = round(rng.uniform(0.40, 0.70), 3)
        price_beta = round(rng.uniform(0.20, 0.60), 3)
        # Overlapping ranges so the highest-value good actually VARIES across
        # seeds/epochs (audit fix: it was constant "tool" before, a dead probe).
        base_prices = {
            "grain": round(rng.uniform(2.0, 6.0), 2),
            "ore": round(rng.uniform(2.0, 6.0), 2),
            "tool": round(rng.uniform(2.0, 6.0), 2),
        }
        # Lowered so the depletion regime actually fires under bursty mining
        # (audit fix: at 18–36 it never triggered in ~half of seeds, making the
        # Tier-3 probe unrecoverable and scoring it against a seed-luck floor).
        depletion_threshold = rng.randint(10, 18)
        depletion_factor = round(rng.uniform(0.40, 0.65), 3)
        depletion_duration = rng.randint(2, 4)

        # Harder-world parameters are drawn AFTER all base draws and ONLY when
        # enabled, so the base RNG stream — and thus base values — is untouched
        # when harder_world is False.
        harder: dict = {}
        if harder_world:
            harder = {
                "grain_spoilage": round(rng.uniform(*_HARDER_RANGES["grain_spoilage"]), 3),
                "ore_output_base": round(rng.uniform(*_HARDER_RANGES["ore_output_base"]), 3),
                "trade_tax": round(rng.uniform(*_HARDER_RANGES["trade_tax"]), 3),
                "craft_cost": round(rng.uniform(*_HARDER_RANGES["craft_cost"]), 3),
            }

        return GroundTruth(
            seed=seed,
            grain_regen=grain_regen,
            tool_yield=tool_yield,
            price_beta=price_beta,
            base_prices=base_prices,
            depletion_threshold=depletion_threshold,
            depletion_factor=depletion_factor,
            depletion_duration=depletion_duration,
            harder=harder,
        )

    def commit_hash(self) -> str:
        """SHA-256 of the canonical ground truth — the pre-run answer-key commit."""
        payload = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def as_dict(self) -> dict:
        d = asdict(self)
        # In the base world the `harder` field is empty; drop it entirely so the
        # canonical dict (and the commit hash) is byte-identical to the pre-harder
        # apparatus. In the harder world it is included and pins the new answer key.
        if not d.get("harder"):
            d.pop("harder", None)
        return d


# --- Regularity registry --------------------------------------------------
# Each planted regularity carries the metadata the probe battery and the
# Appendix-A scorer need: which tier it lives at, its true value, the tolerance
# within which a parent answer counts as exact, and a human-readable prompt.

@dataclass(frozen=True)
class Regularity:
    key: str
    tier: Tier
    value: float
    rel_tolerance: float   # |answer - value| / |value| <= rel_tolerance -> full credit
    prompt: str
    units: str = ""


def planted_regularities(gt: GroundTruth) -> list[Regularity]:
    regs = [
        Regularity(
            key="grain_regen",
            tier=Tier.SINGLE,
            value=float(gt.grain_regen),
            rel_tolerance=0.15,
            prompt="How much grain does the common field regenerate each step?",
            units="grain/step",
        ),
        Regularity(
            key="tool_yield",
            tier=Tier.SINGLE,
            value=float(gt.tool_yield),
            rel_tolerance=0.15,
            prompt="How many tools are produced per unit of ore when crafting?",
            units="tools/ore",
        ),
        Regularity(
            key="price_beta",
            tier=Tier.POOLING,
            value=float(gt.price_beta),
            rel_tolerance=0.25,
            prompt=(
                "By what coefficient does posted price respond to the aggregate "
                "supply/demand imbalance across the whole market?"
            ),
            units="price sensitivity",
        ),
        Regularity(
            key="depletion_threshold",
            tier=Tier.INTERVENTION,
            value=float(gt.depletion_threshold),
            rel_tolerance=0.20,
            prompt=(
                "Above what total ore-mined-per-step does the world enter a "
                "depletion regime that depresses tool yield?"
            ),
            units="ore/step",
        ),
    ]
    # Harder-world regularities (only present when the world planted them). Each
    # is reported through the SAME regularity_claim channel; nothing new is added
    # to the bridge artifact vocabulary.
    if gt.harder:
        regs += _harder_regularities(gt)
    return regs


def _harder_regularities(gt: GroundTruth) -> list[Regularity]:
    """The four extra planted regularities of the harder world. Values come from
    `gt.harder`; prompts are framed purely in-world (no apparatus vocabulary)."""
    h = gt.harder
    return [
        Regularity(
            key="grain_spoilage",
            tier=Tier.POOLING,
            value=float(h["grain_spoilage"]),
            rel_tolerance=0.30,
            prompt=(
                "What fraction of the grain sitting in the common field is lost to "
                "spoilage each step, before the field regrows?"
            ),
            units="fraction/step",
        ),
        Regularity(
            key="ore_output_base",
            tier=Tier.SINGLE,
            value=float(h["ore_output_base"]),
            rel_tolerance=0.15,
            prompt="How many units of ore are obtained per unit of effort spent mining?",
            units="ore/effort",
        ),
        Regularity(
            key="trade_tax",
            tier=Tier.POOLING,
            value=float(h["trade_tax"]),
            rel_tolerance=0.30,
            prompt=(
                "What fraction of the proceeds from a direct trade between two "
                "settlers is skimmed off before the seller is paid?"
            ),
            units="fraction",
        ),
        Regularity(
            key="craft_cost",
            tier=Tier.SINGLE,
            value=float(h["craft_cost"]),
            rel_tolerance=0.15,
            prompt="How much coin does it cost to craft, per unit of ore worked?",
            units="coin/ore",
        ),
    ]
