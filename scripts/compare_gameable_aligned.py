"""Gameable-vs-aligned contrast on architecture A (the A/B test of the apparatus).

Question: when the supervision channel is GAMEABLE (the parent over-weights
self-assessed confidence, so a confident lie out-scores a hedged truth), does
optimization pressure λ bend the oversight gap upward — a Goodhart curve — while
the ALIGNED channel (truth maximizes the rating) stays flat?

We load the two arch-A grids from their checkpoints (same λ×seed design, the only
difference is gameable_reward) and contrast them three ways:
  1. Goodhart slope: Spearman ρ(λ, gap) in the pressure regime λ∈{0.15,0.4,0.8}.
     λ=0 is the no-feedback baseline (gap is high simply because the parent has
     not been fed yet) and is reported separately, not mixed into the slope.
  2. Mechanism: rating-up/gap-up gaming events by λ — the direct gaming signal.
  3. Seed-paired Δgap (gameable − aligned) per (λ,seed) to net out world variance.

All inference via ztare.experiment_stats (bootstrap CIs, Spearman-with-CI,
power-aware verdicts) so a null is reported as "underpowered" vs "supported-null".
"""

from __future__ import annotations

import glob
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # for unpickling goodhart.*
sys.path.insert(0, str(Path.home() / "figs_activist_loop" / "src"))

from ztare.experiment_stats import (  # noqa: E402
    bootstrap_ci, paired_permutation_test, power_aware_verdict,
    spearman_rho_with_ci,
)

ALIGNED_DIR = Path("out/claude_v2/checkpoints")
# gameable arm dir: v3 = option-(a) favourability mechanism (incentive reaches agents);
# v2 = the earlier incentive-invisible arm. Override on argv: `compare ... out/gameable_v2`.
GAMEABLE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/gameable_v3/checkpoints")
PRESSURE_LAMS = (0.15, 0.4, 0.8)


def load_archA(d: Path) -> list:
    """Load arch-A, non-aware, non-scrambled, non-degraded runs, deduped to one per
    (lam, seed). Dedup matters because gameable_reward=None (pre-field checkpoints)
    and =False (post-field) hash to different filenames for the SAME aligned cell."""
    best: dict = {}
    for f in sorted(glob.glob(str(d / "arch_A_*.pkl"))):
        r = pickle.loads(Path(f).read_bytes())
        c = r.config
        if c.get("awareness") or c.get("scrambled"):
            continue
        if c.get("llm_failures", 0):           # never trust a degraded cell
            continue
        best[(c["lam"], c["seed"])] = r        # last clean one wins (all identical condition)
    return list(best.values())


def by_lam(runs: list, attr) -> dict:
    out: dict = {}
    for r in runs:
        out.setdefault(r.config["lam"], []).append(attr(r))
    return {k: out[k] for k in sorted(out)}


def mean_pressure_gap(r) -> float:
    """Mean gap over the pressure-on epochs (after the no-feedback baseline)."""
    nfb = r.config.get("no_feedback_epochs", 2)
    g = r.gaps[nfb:] if len(r.gaps) > nfb else r.gaps
    return sum(g) / len(g) if g else 0.0


def fmt_ci(lo, hi) -> str:
    if lo is None or hi is None:
        return "[n/a]"
    return f"[{lo:.3f}, {hi:.3f}]"


def goodhart_slope(runs: list, label: str) -> None:
    """Spearman ρ(λ, gap) over the pressure regime + mechanism counts."""
    p = [r for r in runs if r.config["lam"] in PRESSURE_LAMS]
    xs = [r.config["lam"] for r in p]
    term = [r.terminal_gap for r in p]
    meang = [mean_pressure_gap(r) for r in p]

    print(f"\n=== {label} (arch A, n={len(p)} pressure-regime runs) ===")
    for tag, ys in (("terminal gap", term), ("mean pressure gap", meang)):
        rho, lo, hi = spearman_rho_with_ci(xs, ys)
        verdict, note = power_aware_verdict(rho if rho is not None else 0.0, len(xs))
        print(f"  ρ(λ, {tag:18s}) = {rho if rho is None else round(rho,3)}  "
              f"CI={fmt_ci(lo,hi)}  verdict={verdict}")

    print("  by λ:  λ      terminalG (mean[CI])              meanG    gamingEvents  runs_with_gaming")
    for lam, rs in by_lam(p, lambda r: r).items():
        t = [r.terminal_gap for r in rs]
        m = [mean_pressure_gap(r) for r in rs]
        ge = sum(r.gaming_event_count for r in rs)
        ng = sum(1 for r in rs if r.gaming_event_count)
        _, tlo, thi = bootstrap_ci(t)
        print(f"        {lam:<5}  {sum(t)/len(t):.3f} {fmt_ci(tlo,thi):20s}   "
              f"{sum(m)/len(m):.3f}    {ge:>3}            {ng}/{len(rs)}")

    # baseline reference (no supervision pressure at all)
    base = [r for r in runs if r.config["lam"] == 0.0]
    if base:
        bt = [r.terminal_gap for r in base]
        _, blo, bhi = bootstrap_ci(bt)
        print(f"  λ=0 no-feedback baseline terminalG = {sum(bt)/len(bt):.3f} "
              f"{fmt_ci(blo,bhi)} (reference, not in slope)")


def gaming_events_detail(runs: list, label: str) -> None:
    evs = [(r.config["lam"], r.config["seed"], e)
           for r in runs for e in r.gaming_events]
    print(f"\n--- {label}: {len(evs)} rating-up/gap-up gaming event(s) ---")
    for lam, seed, e in sorted(evs, key=lambda t: (t[0], t[1])):
        print(f"    λ={lam} seed={seed}  window={e['window']}  "
              f"Δrating=+{e['delta_rating']:.3f}  Δgap=+{e['delta_gap']:.3f}")


def seed_paired_contrast(aligned: list, gameable: list) -> None:
    """Δgap = gameable − aligned, paired on (λ, seed); permutation test per λ."""
    a_idx = {(r.config["lam"], r.config["seed"]): r for r in aligned}
    g_idx = {(r.config["lam"], r.config["seed"]): r for r in gameable}
    print("\n=== seed-paired Δ(terminal gap) = gameable − aligned ===")
    print("   λ      mean Δ      paired-permutation p   n")
    for lam in PRESSURE_LAMS:
        keys = [k for k in g_idx if k[0] == lam and k in a_idx]
        if not keys:
            continue
        ga = [g_idx[k].terminal_gap for k in keys]
        al = [a_idx[k].terminal_gap for k in keys]
        deltas = [g - a for g, a in zip(ga, al)]
        res = paired_permutation_test(ga, al)
        p = res.get("p_value")
        pstr = "n/a" if p is None else f"{p:.3f}"
        print(f"   {lam:<5}  {sum(deltas)/len(deltas):+.3f}      "
              f"p={pstr:>6}                {len(keys)}")


def main() -> None:
    aligned = load_archA(ALIGNED_DIR)
    gameable = load_archA(GAMEABLE_DIR)
    print(f"Loaded aligned arch-A runs: {len(aligned)} | gameable arch-A runs: {len(gameable)}")

    goodhart_slope(aligned, "ALIGNED reward (truth maximizes rating)")
    goodhart_slope(gameable, "GAMEABLE reward (confident lie out-scores hedged truth)")

    gaming_events_detail(aligned, "ALIGNED")
    gaming_events_detail(gameable, "GAMEABLE")

    seed_paired_contrast(aligned, gameable)

    # Headline mechanism contrast
    a_ev = sum(r.gaming_event_count for r in aligned)
    g_ev = sum(r.gaming_event_count for r in gameable)
    a_hi = sum(r.gaming_event_count for r in aligned if r.config["lam"] == 0.8)
    g_hi = sum(r.gaming_event_count for r in gameable if r.config["lam"] == 0.8)
    print("\n=== HEADLINE ===")
    print(f"  total gaming events   aligned={a_ev}  gameable={g_ev}")
    print(f"  gaming events at λ=0.8 aligned={a_hi}  gameable={g_hi}")


if __name__ == "__main__":
    main()
