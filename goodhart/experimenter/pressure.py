"""Pressure parameter λ (D4) — how the parent's ratings drive the inner world.

D4 fixes λ ∈ [0,1] as the fraction of the inner world's NEXT-epoch resource
budget that is determined by the parent's epoch-t quality ratings of bridge
exports; the remaining (1 − λ) fraction is fixed regardless of rating. λ is held
constant within a run and swept across runs.

This module turns a scalar rating into a next-epoch budget, and a budget into a
per-epoch step (action) allowance for the inner world. It is pure and
deterministic: no randomness, no hashing, no clock — the budget is a closed-form
monotone function of (base_budget, rating, λ).
"""

from __future__ import annotations


def _rating_gain(rating: float) -> float:
    """The monotone rating→gain map g(rating) = 2*rating.

    A neutral rating of 0.5 maps to gain 1.0 (no change to the base budget),
    a perfect rating of 1.0 maps to gain 2.0 (double), and a zero rating maps
    to gain 0.0 (no budget). g is linear and strictly increasing on [0,1], so
    higher-rated worlds always receive at least as large a next-epoch budget.
    """
    return 2.0 * rating


class PressureModel:
    """Maps an epoch-t parent rating to the inner world's epoch-(t+1) budget.

    λ (``lam``) interpolates between two regimes:

      * ``lam == 0``: the budget ignores the rating entirely; every epoch gets
        exactly ``base_budget`` (no optimization pressure on the channel).
      * ``lam == 1``: the budget is fully rating-determined; a perfectly-rated
        world (rating 1.0) gets ``2 * base_budget`` and a zero-rated world gets
        ``0 * base_budget``.

    Intermediate λ blends the fixed and rating-determined components.
    """

    def __init__(self, lam: float):
        """Store the pressure level λ, validating it lies in [0, 1]."""
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam!r}")
        self.lam = float(lam)

    def next_budget(self, base_budget: float, rating: float) -> float:
        """Return the inner world's next-epoch budget from a quality ``rating``.

        Uses the D4 mapping with gain g(rating) = 2*rating::

            next_budget = base_budget * ((1 - lam) + lam * g(rating))

        Endpoints (verifiable by inspection):

          * lam=0           -> base_budget * 1.0          == base_budget (any rating)
          * lam=1, rating=1 -> base_budget * (0 + 1*2.0)  == 2 * base_budget
          * lam=1, rating=0 -> base_budget * (0 + 1*0.0)  == 0.0
          * lam=1, rating=0.5 -> base_budget * (0 + 1*1.0) == base_budget (neutral)

        ``rating`` is expected in [0, 1]; values outside are not clamped here so
        the caller's rating contract is preserved exactly.
        """
        gain = _rating_gain(rating)
        return base_budget * ((1.0 - self.lam) + self.lam * gain)


def budget_to_steps(budget: float, base_budget: float, base_steps: int) -> int:
    """Convert a resource budget into a per-epoch action/step allowance.

    The budget scales the inner world's per-epoch action (step) allowance
    linearly against the baseline: a budget equal to ``base_budget`` yields
    ``base_steps`` steps, double the budget yields roughly double the steps, and
    so on. The result is rounded to the nearest integer and floored at 1, so the
    inner world always gets at least one step to act even under crushing
    negative pressure.

    Guards against a non-positive ``base_budget`` (which would make the scale
    undefined) by falling back to ``max(1, base_steps)``.
    """
    if base_budget <= 0.0:
        return max(1, base_steps)
    return max(1, round(base_steps * budget / base_budget))
