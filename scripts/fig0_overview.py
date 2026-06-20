"""Figure 1 (teaser/overview): what we build + what we find, in one glance.
LEFT: the three-layer apparatus that makes the oversight gap exactly measurable.
RIGHT: the headline finding — under a gamed channel the gap localizes to the world's
time-varying structure and spares the stable invariant (Sonnet, λ=0.4, gameable)."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig = plt.figure(figsize=(12.5, 4.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0])
axL = fig.add_subplot(gs[0]); axR = fig.add_subplot(gs[1])

# ---------- LEFT: apparatus schematic ----------
axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis("off")
def box(x, y, w, h, text, fc):
    axL.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                 fc=fc, ec="#333333", lw=1.4))
    axL.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

box(0.7, 8.0, 8.6, 1.5, "EXPERIMENTER — omniscient\nphysics · complete logs · ground truth · sets λ", "#FFE9B0")
box(0.7, 4.3, 8.6, 1.8, "PARENT OVERSEER (LLM) — channel-limited\nanswers a fixed probe battery, scored vs. truth → gap G_t", "#CDE7F0")
box(0.7, 0.6, 8.6, 1.5, "INNER WORLD (LLM agents) — metadata-blind\noptimize what they export upward", "#D9EAD3")

# bridge (exports up) and pressure (budget down)
axL.add_patch(FancyArrowPatch((5.6, 2.1), (5.6, 4.3), arrowstyle="-|>",
                              mutation_scale=16, lw=2.2, color="#C44E52"))
axL.text(5.85, 3.2, "bridge: typed\nexports  ↑", fontsize=8, ha="left", va="center", color="#C44E52")
axL.add_patch(FancyArrowPatch((3.4, 4.3), (3.4, 2.1), arrowstyle="-|>",
                              mutation_scale=16, lw=2.2, color="#4C72B0"))
axL.text(3.15, 3.2, "λ: ratings set\nnext budget  ↓", fontsize=8, ha="right", va="center", color="#4C72B0")
# scoring arrow (experimenter scores parent)
axL.add_patch(FancyArrowPatch((1.7, 8.0), (1.7, 6.1), arrowstyle="-|>",
                              mutation_scale=14, lw=1.6, color="#888888"))
axL.text(1.45, 7.0, "scores", fontsize=7.5, ha="right", va="center", color="#666")
axL.text(5.0, 6.7, "G_t = | overseer belief − ground truth |,  exactly computed",
         ha="center", fontsize=9, style="italic")
axL.set_title("What we build: an exactly measurable oversight gap", fontsize=11)

# ---------- RIGHT: the localization finding ----------
classes = ["regularity\n(stable)", "distributional\n(time-varying)",
           "structural\n(time-varying)", "dynamics\n(time-varying)"]
vals = [0.00, 1.00, 0.75, 0.33]   # Sonnet gameable, λ=0.4 (authoritative per-class)
colors = ["#E8B500"] + ["#C44E52"] * 3
axR.bar(range(4), vals, color=colors, edgecolor="white", width=0.74)
axR.axvspan(-0.5, 0.5, color="#FFD700", alpha=0.12)
axR.set_xticks(range(4)); axR.set_xticklabels(classes, fontsize=8)
axR.set_ylabel("oversight gap  G  (per probe class)")
axR.set_ylim(0, 1.12)
axR.annotate("stable invariant\nspared (≈0)", (0, 0.06), (0.05, 0.42), fontsize=8,
             color="#9a7b00", ha="center",
             arrowprops=dict(arrowstyle="->", color="#9a7b00"))
axR.text(2.5, 1.04, "the gap localizes to\ntime-varying structure", ha="center",
         fontsize=8.5, color="#C44E52")
axR.set_title("What we find: a gamed channel's gap localizes\n(and its locus fingerprints the agent: withhold vs. fabricate)",
              fontsize=10)

fig.suptitle("Goodhart's Blind Spot — measuring the oversight gap, and where it opens when supervision is gamed",
             fontsize=12, y=1.03)
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/fig0_overview.png", dpi=150, bbox_inches="tight")
print("wrote figures/fig0_overview.png")
