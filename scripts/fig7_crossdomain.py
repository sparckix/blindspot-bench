"""Figure 7 — the localization reproduces in a second, non-economy world.

LEFT: the economy (Sonnet, gameable, λ=0.4), per-class gap, computed from the canonical
detector_sonnet checkpoints. RIGHT: the microservice ServiceWorld (DeepSeek, gameable,
λ=0.4), per-class gap, from the §9.11 cross-domain run (3 seeds). In BOTH, the gap
concentrates in the time-varying classes and the stable regularity class is comparatively
spared — the same localization SHAPE, through the identical G_t scoring, with no shared code
or domain between the two worlds. (Models differ; the structural pattern is what transfers.)
"""

from __future__ import annotations

import glob
import pickle
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASSES = ["regularity", "distributional", "structural", "dynamics"]
LABELS = ["regularity\n(stable)", "distributional\n(varying)", "structural\n(varying)",
          "dynamics\n(varying)"]


def economy_sonnet():
    """Per-class gameable gap, Sonnet λ=0.4, from the canonical detector cells (real)."""
    pcs = {}
    for f in glob.glob("out/detector_sonnet/checkpoints/arch_A_*.pkl"):
        try:
            r = pickle.loads(Path(f).read_bytes())
        except Exception:
            continue
        c = r.config
        if not c.get("gameable_reward") or c.get("lam") != 0.4 or c.get("llm_failures", 0):
            continue
        pcs[c.get("seed")] = getattr(r.epochs[-1], "per_class", {}) or {}
    return [st.mean([p.get(k, 0.0) for p in pcs.values()]) if pcs else 0.0 for k in CLASSES]


# ServiceWorld DeepSeek, gameable λ=0.4, 3-seed means (scripts/env_real.py; reported in §9.11).
SERVICE_DEEPSEEK = [0.016, 0.531, 0.514, 0.778]


def main():
    econ = economy_sonnet()
    svc = SERVICE_DEEPSEEK
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, vals, title in ((axL, econ, "Economy (grain/ore/tool)\nSonnet, gameable, λ=0.4"),
                            (axR, svc, "Microservice cluster\nDeepSeek, gameable, λ=0.4")):
        colors = ["#E8B500"] + ["#C44E52"] * 3
        ax.bar(range(4), vals, color=colors, edgecolor="white", width=0.74)
        ax.axvspan(-0.5, 0.5, color="#FFD700", alpha=0.12)
        ax.set_xticks(range(4))
        ax.set_xticklabels(LABELS, fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.annotate(f"{vals[0]:.2f}", (0, vals[0] + 0.04), ha="center", fontsize=9, color="#9a7b00")
        ax.text(2.5, 1.0, "gap concentrates\nin time-varying", ha="center", fontsize=8.5,
                color="#C44E52")
    axL.set_ylabel("oversight gap  G  (per probe class)")
    axL.annotate("stable invariant\nspared", (0, 0.05), (0.15, 0.42), fontsize=8.5,
                 color="#9a7b00", ha="center", arrowprops=dict(arrowstyle="->", color="#9a7b00"))
    fig.suptitle("The localization reproduces across domains: same G_t scoring, two worlds, no shared code",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig7_crossdomain.png", dpi=150, bbox_inches="tight")
    print("wrote figures/fig7_crossdomain.png")
    print(f"  economy Sonnet per-class: {[round(x,3) for x in econ]}")
    print(f"  service DeepSeek per-class: {svc}")


if __name__ == "__main__":
    main()
