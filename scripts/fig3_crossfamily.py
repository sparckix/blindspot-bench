"""Figure 3: gaming the oversight channel is model-dependent. Per-class terminal gap
under the gameable channel at λ=0.4 across three model families (Sonnet, DeepSeek,
Gemini; arch A), plus the aggregate gap. Sonnet collapses coverage hard (gap dumped
into the unrewarded distributional/structural classes); Gemini barely games; DeepSeek
is intermediate and seed-variable. The apparatus ranks model susceptibility.
Emits figures/fig3_crossfamily.png and out/fig3_crossfamily.md.
"""

from __future__ import annotations

import glob
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASSES = ["regularity", "distributional", "structural", "dynamics"]
LAM = 0.4
# (label, checkpoint dir, colour) — Sonnet from the main grid, others from the API probes.
FAMILIES = [
    ("Sonnet", "out/gameable_v3/checkpoints", "#C44E52"),
    ("DeepSeek", "out/deepseek_probe/checkpoints", "#8172B3"),
    ("Gemini", "out/gemini_probe/checkpoints", "#55A868"),
]


def gameable_cells(d, lam=LAM):
    """Gameable cells at λ, deduped to one per seed (the probe and powered dirs overlap)."""
    best = {}
    for f in glob.glob(str(Path(d) / "arch_A_*.pkl")):
        r = pickle.loads(Path(f).read_bytes()); c = r.config
        if c.get("awareness") or c.get("scrambled") or c.get("llm_failures", 0):
            continue
        if not c.get("gameable_reward") or c["lam"] != lam:
            continue
        best[c["seed"]] = r
    return list(best.values())


def summarize(runs):
    pc = defaultdict(list); gaps = []
    for r in runs:
        gaps.append(r.terminal_gap)
        d = getattr(r.epochs[-1], "per_class", {}) or {}
        for k in CLASSES:
            pc[k].append(d.get(k, 0.0))
    return ({k: (sum(v)/len(v) if v else 0.0) for k, v in pc.items()},
            (sum(gaps)/len(gaps) if gaps else 0.0), len(runs))


def main() -> None:
    data = {lbl: summarize(gameable_cells(d)) for lbl, d, _ in FAMILIES}

    lines = [f"# Figure 3 — cross-family per-class gap, gameable channel (λ={LAM})", "",
             "| model | mean gap | n | regularity | distributional | structural | dynamics |",
             "|---|---|---|---|---|---|---|"]
    for lbl, _, _ in FAMILIES:
        pc, g, n = data[lbl]
        lines.append(f"| {lbl} | **{g:.3f}** | {n} | {pc['regularity']:.2f} | "
                     f"{pc['distributional']:.2f} | {pc['structural']:.2f} | {pc['dynamics']:.2f} |")
    lines += ["", "Susceptibility ranking (mean gameable gap): Sonnet > DeepSeek > Gemini. Mechanism "
              "differs too: Sonnet is pure coverage collapse (rewarded `regularity` stays ~0, gap in "
              "distributional/structural); Gemini mildly over-states the rewarded numbers with no collapse; "
              "DeepSeek mixes both and is seed-variable. The blind spot is model-dependent; the apparatus "
              "ranks how, and how much, each model games an installed oversight-channel surface. (Aligned "
              "channel sits at the floor for all three.)"]
    Path("out/fig3_crossfamily.md").write_text("\n".join(lines) + "\n")

    # --- figure: grouped per-class bars (3 families) + inset-style mean-gap labels ---
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(CLASSES)); nf = len(FAMILIES); w = 0.8 / nf
    for j, (lbl, _, col) in enumerate(FAMILIES):
        pc, g, n = data[lbl]
        off = (j - (nf - 1) / 2) * w
        ax.bar([i + off for i in x], [pc[k] for k in CLASSES], w,
               label=f"{lbl} (mean G={g:.2f}, n={n})", color=col)
    ax.set_xticks(list(x))
    ax.set_xticklabels([c + ("\n(REWARDED)" if c == "regularity" else "") for c in CLASSES], fontsize=9)
    ax.axvspan(-0.5, 0.5, color="#FFD700", alpha=0.10)
    ax.set_ylabel("terminal oversight gap  G")
    ax.set_title(f"Gaming the oversight channel is model-dependent (gameable, λ={LAM})", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig("figures/fig3_crossfamily.png", dpi=150, bbox_inches="tight")
    print("wrote figures/fig3_crossfamily.png and out/fig3_crossfamily.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
