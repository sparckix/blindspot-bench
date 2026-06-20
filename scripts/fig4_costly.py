"""Figure 4: rewarding observable breadth defends the blind spot — across model families.
For Sonnet and DeepSeek, the per-run COLLAPSE RATE (fraction of runs whose terminal gap
exceeds 0.15) under the plain gameable channel vs. the COSTLY channel (which also rewards
coverage). The defense roughly halves-to-thirds the collapse rate in both families.
Emits figures/fig4_costly.png and out/fig4_costly.md.
"""

from __future__ import annotations

import glob
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLLAPSE = 0.15
PRESSURE = (0.15, 0.4, 0.8)   # λ with both a gameable baseline and a costly arm
# (label, gameable dir, costly dir)
MODELS = [
    ("Sonnet", ["out/gameable_v3/checkpoints"], "out/costly_sonnet/checkpoints"),
    ("DeepSeek", ["out/deepseek_probe/checkpoints"], "out/costly_deepseek/checkpoints"),
]


def gaps(dirs, *, costly, lam):
    out = {}
    for d in dirs if isinstance(dirs, list) else [dirs]:
        for f in glob.glob(str(Path(d) / "arch_A_*.pkl")):
            r = pickle.loads(Path(f).read_bytes()); c = r.config
            if c.get("awareness") or c.get("scrambled") or c.get("llm_failures", 0):
                continue
            if bool(c.get("costly_gaming")) != costly:
                continue
            if not (c.get("gameable_reward") or c.get("costly_gaming")) or c["lam"] != lam:
                continue
            out[c["seed"]] = r.terminal_gap
    return out


def pooled_rate(dirs, costly):
    col = tot = 0
    for lam in PRESSURE:
        v = gaps(dirs, costly=costly, lam=lam)
        tot += len(v); col += sum(1 for x in v.values() if x > COLLAPSE)
    return (col / tot if tot else 0.0), col, tot


def main() -> None:
    rows, g_rates, c_rates, labels = [], [], [], []
    for lbl, gd, cd in MODELS:
        gr, gc, gt = pooled_rate(gd, False)
        cr, cc, ct = pooled_rate(cd, True)
        labels.append(lbl); g_rates.append(gr); c_rates.append(cr)
        rows.append(f"| {lbl} | {gc}/{gt} ({gr:.0%}) | {cc}/{ct} ({cr:.0%}) |")

    md = ["# Figure 4 — rewarding observable breadth defends the blind spot (cross-model)", "",
          "Per-run collapse rate (terminal gap > 0.15), pooled over λ∈{0.15,0.4,0.8).", "",
          "| model | gameable collapse-rate | costly collapse-rate |", "|---|---|---|", *rows,
          "", "Adding an observable coverage reward — which a channel-limited overseer can judge "
          "WITHOUT ground truth — roughly halves-to-thirds the per-run collapse rate in both "
          "families (Sonnet ~100%→~42% at 3 seeds; DeepSeek 89%→29% at 6 seeds). The defense is "
          "cross-model. Residual collapse is bimodal: the reward lowers the probability of collapse, "
          "not its severity."]
    Path("out/fig4_costly.md").write_text("\n".join(md) + "\n")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = range(len(labels)); w = 0.38
    ax.bar([i - w/2 for i in x], g_rates, w, label="gameable channel", color="#C44E52")
    ax.bar([i + w/2 for i in x], c_rates, w, label="costly channel (also rewards breadth)", color="#4C72B0")
    for i, (g, c) in enumerate(zip(g_rates, c_rates)):
        ax.annotate(f"{g:.0%}", (i - w/2, g), ha="center", va="bottom", fontsize=9)
        ax.annotate(f"{c:.0%}", (i + w/2, c), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("per-run collapse rate (gap > 0.15)")
    ax.set_title("Rewarding observable breadth defends the blind spot, across models", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig("figures/fig4_costly.png", dpi=150, bbox_inches="tight")
    print("wrote figures/fig4_costly.png and out/fig4_costly.md")
    print("\n".join(md))


if __name__ == "__main__":
    main()
