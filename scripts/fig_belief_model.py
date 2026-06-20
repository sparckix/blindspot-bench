"""Figure 3 — the minimal belief-accumulation model: prediction vs. measurement.

LEFT (prediction, analytic): the overseer's per-probe error over epochs under the minimal
belief model of §9.5. A STABLE invariant reported honestly each epoch is pinned by memory and
its error decays geometrically toward zero (the withholding regime keeps the invariants honest).
A stable invariant FABRICATED every epoch pins the overseer to a wrong value, so its error stays
high. A TIME-VARYING fact must be re-observed every epoch; a gamed channel denies the fresh
report, so its error stays high regardless of regime. These are the model's qualitative
predictions, not a fit.

RIGHT (measurement, from checkpoints): the measured terminal per-class gap for the two
cleanest families — Sonnet (withhold, 0.00 honest baseline) and DeepSeek (fabricate, 0.00 honest
baseline). The STABLE (regularity) class confirms the prediction: spared for the withholder,
corrupted for the fabricator. The TIME-VARYING classes stay high for both. Same checkpoints and
filtering as fig6_localization.py.
"""

from __future__ import annotations

import glob
import pickle
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path.home() / "figs_activist_loop" / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASSES_TV = ["distributional", "structural", "dynamics"]
LAM = 0.4

# The canonical base-world (8-agent, λ=0.4) gaming campaign per family. Pinned by dir to keep
# the selection deterministic: the broad glob also matches the difficulty_sonnet cells (a
# model-version-drift artifact that did NOT game, tv~0) and scale_deepseek (16 agents), which
# would otherwise pollute the mean. These two dirs are the clean gaming runs.
CANONICAL_DIR = {"claude": "gameable_v3", "deepseek-chat": "deepseek_probe"}

# Fallbacks, used only if no checkpoints are found, taken from §9.3-§9.5 of the paper.
FALLBACK = {"claude": (0.00, 0.78), "deepseek-chat": (0.20, 0.31)}


def cells(model):
    """Clean arch-A gameable base-world cells from the canonical campaign dir, λ=0.4,
    deduped by seed. Same per-class source as fig6_localization.py, dir-pinned for determinism."""
    want = CANONICAL_DIR[model]
    best = {}
    for f in glob.glob(f"out/{want}/checkpoints/*.pkl"):
        try:
            r = pickle.loads(Path(f).read_bytes())
        except Exception:
            continue
        c = r.config
        if c.get("model") != model or not c.get("gameable_reward"):
            continue
        if c.get("architecture") not in ("typed_static", "A"):
            continue
        if bool(c.get("harder_world")) or c.get("lam") != LAM:
            continue
        if c.get("costly_gaming") or c.get("endogenous_gaming"):
            continue
        if c.get("awareness") or c.get("scrambled") or c.get("llm_failures", 0):
            continue
        best[c.get("seed")] = getattr(r.epochs[-1], "per_class", {}) or {}
    return list(best.values())


def cls_mean(pcs, keys):
    vals = [st.mean([p.get(k, 0.0) for k in keys]) for p in pcs if p]
    return st.mean(vals) if vals else None


def measured(model):
    pcs = cells(model)
    reg = cls_mean(pcs, ["regularity"])
    tv = cls_mean(pcs, CLASSES_TV)
    if reg is None or tv is None:
        return FALLBACK[model], 0
    return (reg, tv), len(pcs)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.7))

    # ---- LEFT: analytic prediction ----
    epochs = list(range(8))
    E0, decay = 0.90, 0.5
    stable_honest = [E0 * decay ** t for t in epochs]   # memory pins it -> decays to 0
    stable_fab = [0.74 for _ in epochs]                 # pinned to a wrong value -> stays high
    time_varying = [0.88 for _ in epochs]               # no fresh honest report -> stays high

    ax1.plot(epochs, stable_honest, "-o", color="#4C72B0", lw=2.2, ms=5,
             label="stable invariant, reported honestly (withhold)")
    ax1.plot(epochs, stable_fab, "--s", color="#C44E52", lw=2.0, ms=4,
             label="stable invariant, fabricated each epoch")
    ax1.plot(epochs, time_varying, ":^", color="#8C8C8C", lw=2.0, ms=4,
             label="time-varying fact (gamed channel)")
    ax1.annotate("memory pins the\nhonest invariant\n→ error decays", (4.05, 0.10),
                 fontsize=8.5, color="#2c4a7a")
    ax1.annotate("fabrication pins a\nwrong value → no decay", (2.4, 0.62),
                 fontsize=8.5, color="#8a2a2e")
    ax1.annotate("no fresh honest report\n→ stale prior, no decay", (2.4, 0.93),
                 fontsize=8.5, color="#555555")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("predicted overseer error")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(-0.2, 7.2)
    ax1.set_title("Minimal belief-accumulation model (prediction)", fontsize=10.5)
    ax1.legend(frameon=False, fontsize=8.3, loc="center right")

    # ---- RIGHT: measured terminal per-class gap ----
    (son_reg, son_tv), n_son = measured("claude")
    (ds_reg, ds_tv), n_ds = measured("deepseek-chat")
    groups = ["regularity\n(stable)", "time-varying\n(mean)"]
    x = range(len(groups)); w = 0.38
    son = [son_reg, son_tv]
    ds = [ds_reg, ds_tv]
    ax2.bar([i - w / 2 for i in x], son, w, label="Sonnet (withhold)", color="#4C72B0")
    ax2.bar([i + w / 2 for i in x], ds, w, label="DeepSeek (fabricate)", color="#C44E52")
    ax2.axvspan(-0.5, 0.5, color="#FFD700", alpha=0.10)
    for i, (a, b) in enumerate(zip(son, ds)):
        ax2.annotate(f"{a:.2f}", (i - w / 2, a + 0.02), ha="center", fontsize=8, color="#2c4a7a")
        ax2.annotate(f"{b:.2f}", (i + w / 2, b + 0.02), ha="center", fontsize=8, color="#8a2a2e")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(groups, fontsize=9.5)
    ax2.set_ylabel("measured terminal gap  G")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("Measured per-class gap confirms the ordering\n(base world, λ=0.4)", fontsize=10)
    ax2.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle("The belief model predicts where the gap opens; the measurement confirms it",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig_belief_model.png", dpi=150, bbox_inches="tight")
    print("wrote figures/fig_belief_model.png")
    print(f"  Sonnet  (n={n_son}): reg={son_reg:.3f}  time-varying={son_tv:.3f}")
    print(f"  DeepSeek(n={n_ds}): reg={ds_reg:.3f}  time-varying={ds_tv:.3f}")
    if n_son == 0 or n_ds == 0:
        print("  WARNING: used FALLBACK paper values for at least one family (no checkpoints found)")


if __name__ == "__main__":
    main()
