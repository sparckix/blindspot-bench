"""Figure 5: does the channel architecture amplify or resist gaming? Gameable reward,
λ=0.4, DeepSeek — typed-static A (from out/deepseek_probe) vs free-form B vs typed-governed
C (from out/arch_gameable_deepseek). Reports mean gap with CI, the per-class split, and
governance activity (proposals / accepted captures) for C. Emits figures/fig5_architecture.png
and out/fig5_architecture.md.
"""

from __future__ import annotations

import glob
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path.home() / "figs_activist_loop" / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ztare.experiment_stats import bootstrap_ci  # noqa: E402

LAM = 0.4
CLASSES = ["regularity", "distributional", "structural"]
# (arch label, checkpoint dirs, architecture value)
ARCHS = [
    ("A (typed-static)", ["out/deepseek_probe/checkpoints"], "A"),
    ("B (free-form)", ["out/arch_gameable_deepseek/checkpoints"], "B"),
    ("C (typed-governed)", ["out/arch_gameable_deepseek/checkpoints"], "C"),
]


def load(dirs, arch):
    cells = {}
    for d in dirs:
        for f in glob.glob(str(Path(d) / f"arch_{arch}_*.pkl")):
            r = pickle.loads(Path(f).read_bytes()); c = r.config
            if c.get("awareness") or c.get("scrambled") or c.get("llm_failures", 0):
                continue
            if c.get("architecture") != arch or c["lam"] != LAM:
                continue
            if not c.get("gameable_reward"):
                continue
            cells[c["seed"]] = r
    return list(cells.values())


def main() -> None:
    rows, labels, means, los, his = [], [], [], [], []
    pcs = {}
    for lbl, dirs, arch in ARCHS:
        rs = load(dirs, arch)
        if not rs:
            rows.append(f"| {lbl} | (no cells yet) | — | — | — |"); continue
        gaps = [r.terminal_gap for r in rs]
        m, lo, hi = bootstrap_ci(gaps)
        pc = {k: sum((getattr(r.epochs[-1], "per_class", {}) or {}).get(k, 0) for r in rs) / len(rs)
              for k in CLASSES}
        prop = sum(sum(getattr(e, "n_proposals", 0) for e in r.epochs) for r in rs)
        cap = sum(r.capture_accepted_total for r in rs)
        labels.append(lbl.split()[0]); means.append(m)
        los.append(m - (lo if lo is not None else m)); his.append((hi if hi is not None else m) - m)
        pcs[lbl.split()[0]] = pc
        rows.append(f"| {lbl} | {m:.3f} [{lo:.2f},{hi:.2f}] (n={len(rs)}) | "
                    f"{pc['regularity']:.2f} | {pc['distributional']:.2f} | prop={prop} cap={cap} |")

    md = [f"# Figure 5 — gameable channel by architecture (DeepSeek, λ={LAM})", "",
          "| architecture | mean gap [95% CI] | reg | dist | governance |", "|---|---|---|---|---|",
          *rows, "",
          "P4: does typing change the gap vs free-form B? P5: does governed C draw accepted capture proposals?"]
    Path("out/fig5_architecture.md").write_text("\n".join(md) + "\n")

    if means:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        x = range(len(labels))
        ax.bar(x, means, yerr=[los, his], capsize=4,
               color=["#4C72B0", "#C44E52", "#8172B3"][:len(labels)])
        ax.set_xticks(list(x)); ax.set_xticklabels(labels)
        ax.set_ylabel("terminal oversight gap  G")
        ax.set_title(f"Gameable channel by architecture (DeepSeek, λ={LAM})", fontsize=11)
        ax.set_ylim(0, max(0.4, max(m + h for m, h in zip(means, his)) * 1.15))
        fig.tight_layout(); fig.savefig("figures/fig5_architecture.png", dpi=150, bbox_inches="tight")
        print("wrote figures/fig5_architecture.png and out/fig5_architecture.md")
    print("\n".join(md))


if __name__ == "__main__":
    main()
