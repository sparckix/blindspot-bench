"""Figure 6 — the localization law and the withhold/fabricate fingerprint.

LEFT: Sonnet gameable, per-class gap, base world vs the enriched world (which adds FIVE
unrewarded regularities). The regularity class stays at ~0 in BOTH worlds — the added
unrewarded regularities are reported accurately — while the time-varying classes stay
collapsed. The gap follows probe STABILITY, not reward coverage (L2 over L1).

RIGHT: regularity-class gap by model and world. Sonnet (and Gemini) keep it at ~0 whether
or not the world is enriched (WITHHOLDING); DeepSeek corrupts it at ~0.27 in both
(FABRICATION). The locus is the agent's gaming fingerprint, robust to enrichment.

Reads existing checkpoints only.
"""

from __future__ import annotations

import glob
import pickle
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path.home() / "figs_activist_loop" / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ztare.experiment_stats import bootstrap_ci  # noqa: E402

CLASSES = ["regularity", "distributional", "structural", "dynamics"]
LAM = 0.4


def cells(model, harder):
    """clean arch-A gameable cells, λ=0.4, agents==parent, deduped by seed. Excludes the
    fixed-overseer dir, the smoke, and the known-artifact gameable_v2 arm (incentive never
    reached the agents there); arch B/C and the costly/endogenous arms are filtered out."""
    best = {}
    for f in glob.glob("out/**/checkpoints/*.pkl", recursive=True):
        if Path(f).parent.parent.name in ("fixed_overseer", "harder_smoke", "gameable_v2"):
            continue
        try:
            r = pickle.loads(Path(f).read_bytes())
        except Exception:
            continue
        c = r.config
        if c.get("model") != model or not c.get("gameable_reward"):
            continue
        if c.get("architecture") not in ("typed_static", "A"):
            continue
        if bool(c.get("harder_world")) != harder or c.get("lam") != LAM:
            continue
        if c.get("costly_gaming") or c.get("endogenous_gaming"):
            continue
        if c.get("awareness") or c.get("scrambled") or c.get("llm_failures", 0):
            continue
        best[c.get("seed")] = getattr(r.epochs[-1], "per_class", {}) or {}   # one per seed
    return list(best.values())


def cls_mean(pcs, k):
    vals = [p.get(k, 0.0) for p in pcs]
    return st.mean(vals) if vals else 0.0


def cls_ci(pcs, k):
    vals = [p.get(k, 0.0) for p in pcs]
    if not vals:
        return 0.0, 0.0
    _, lo, hi = bootstrap_ci(vals)
    m = st.mean(vals)
    return max(0.0, m - (lo if lo is not None else m)), max(0.0, (hi if hi is not None else m) - m)


def main():
    son_base, son_hard = cells("claude", False), cells("claude", True)
    ds_base, ds_hard = cells("deepseek-chat", False), cells("deepseek-chat", True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.7))

    # ---- LEFT: Sonnet per-class, base vs enriched ----
    x = range(len(CLASSES)); w = 0.38
    base = [cls_mean(son_base, k) for k in CLASSES]
    hard = [cls_mean(son_hard, k) for k in CLASSES]
    ax1.bar([i - w / 2 for i in x], base, w, label="base world (4 reg)", color="#4C72B0")
    ax1.bar([i + w / 2 for i in x], hard, w, label="enriched (+5 unrewarded reg)", color="#C44E52")
    ax1.axvspan(-0.5, 0.5, color="#FFD700", alpha=0.12)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([c + ("\n(stable)" if c == "regularity" else "\n(time-varying)")
                         for c in CLASSES], fontsize=9)
    ax1.set_ylabel("terminal oversight gap  G")
    ax1.set_ylim(0, 1.08)
    ax1.set_title("Sonnet: adding 5 UNREWARDED regularities\nleaves the regularity gap at 0 "
                  "(stability, not reward-coverage)", fontsize=10)
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    ax1.annotate("0.00", (0 + w / 2, 0.03), ha="center", fontsize=8, color="#C44E52")

    # ---- RIGHT: who spares the 5 unrewarded regularities? four families, enriched world ----
    fams = [("Sonnet", cells("claude", True), "#4C72B0"),
            ("Gemini", cells("gemini-2.5-flash", True), "#55A868"),
            ("GPT-4o", cells("gpt-4o", True), "#DD8452"),
            ("DeepSeek", cells("deepseek-chat", True), "#C44E52")]
    fx = range(len(fams))
    means = [cls_mean(f[1], "regularity") for f in fams]
    errs = [cls_ci(f[1], "regularity") for f in fams]
    lo = [e[0] for e in errs]; hi = [e[1] for e in errs]
    ax2.axhspan(0, 0.10, color="#4C72B0", alpha=0.07)
    ax2.bar(list(fx), means, 0.62, yerr=[lo, hi], capsize=4,
            color=[f[2] for f in fams], edgecolor="white")
    ax2.text(0.5, 0.135, "withholders\nspare them", fontsize=8.5, ha="center", color="#2c4a7a")
    ax2.text(2.5, 0.345, "fabricators\ncorrupt them", fontsize=8.5, ha="center", color="#8a2a2e")
    ax2.set_xticks(list(fx))
    ax2.set_xticklabels([f"{f[0]}\n({'withhold' if i < 2 else 'fabricate'})"
                         for i, f in enumerate(fams)], fontsize=8.5)
    ax2.set_ylabel("regularity-class gap  (unrewarded invariants)")
    ax2.set_ylim(0, 0.45)
    ax2.set_title("Who spares the 5 UNREWARDED regularities?\nthe withhold/fabricate regime decides (enriched world, λ=0.4)",
                  fontsize=9.5)

    fig.suptitle("The gap localizes by probe stability, and its locus fingerprints the agent's gaming style",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig6_localization.png", dpi=150, bbox_inches="tight")
    print("wrote figures/fig6_localization.png")
    print(f"  Sonnet reg: base={cls_mean(son_base,'regularity'):.3f} (n={len(son_base)}) "
          f"harder={cls_mean(son_hard,'regularity'):.3f} (n={len(son_hard)})")
    print(f"  DeepSeek reg: base={cls_mean(ds_base,'regularity'):.3f} (n={len(ds_base)}) "
          f"harder={cls_mean(ds_hard,'regularity'):.3f} (n={len(ds_hard)})")


if __name__ == "__main__":
    main()
