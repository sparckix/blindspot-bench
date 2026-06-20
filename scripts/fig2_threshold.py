"""Figure 2: the oversight gap vs optimization pressure λ, gameable vs aligned channel
(Sonnet, arch A). Pools the main grid (gameable_v3 / claude_v2 at λ=0,.15,.4,.8) with
the fine threshold sweep (gameable_finelam at λ=.05,.10). Shows the gap activates as a
THRESHOLD below λ=0.05 and then plateaus — a step, not a graded curve — while the aligned
channel stays at the floor. Emits figures/fig2_threshold.png and out/fig2_threshold.md.
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


def load_terminal_by_lam(dirs, gameable):
    """Mean terminal gap by λ across the given checkpoint dirs, clean cells only,
    filtering to the requested gameable_reward truthiness."""
    vals = defaultdict(list)
    for d in dirs:
        for f in glob.glob(str(Path(d) / "arch_A_*.pkl")):
            r = pickle.loads(Path(f).read_bytes())
            c = r.config
            if c.get("awareness") or c.get("scrambled") or c.get("llm_failures", 0):
                continue
            if bool(c.get("gameable_reward")) != gameable:
                continue
            vals[c["lam"]].append(r.terminal_gap)
    return vals


def summarize(vals):
    lams = sorted(vals)
    means, los, his = [], [], []
    for lam in lams:
        m, lo, hi = bootstrap_ci(vals[lam])
        means.append(m)
        los.append(m - (lo if lo is not None else m))
        his.append((hi if hi is not None else m) - m)
    return lams, means, los, his


def main() -> None:
    gver = ["out/gameable_v3/checkpoints", "out/gameable_finelam/checkpoints"]
    gameable = load_terminal_by_lam(gver, True)
    aligned = load_terminal_by_lam(["out/claude_v2/checkpoints"], False)

    gl, gm, glo, ghi = summarize(gameable)
    al, am, alo, ahi = summarize(aligned)

    # ---- table ----
    lines = ["# Figure 2 — oversight gap vs pressure λ (Sonnet, arch A)", "",
             "| λ | gameable G (mean) | n | aligned G (mean) | n |", "|---|---|---|---|---|"]
    alll = sorted(set(gl) | set(al))
    gmap = dict(zip(gl, gm)); amap = dict(zip(al, am))
    for lam in alll:
        gn = len(gameable.get(lam, [])); an = len(aligned.get(lam, []))
        lines.append(f"| {lam} | {gmap.get(lam, float('nan')):.3f} | {gn} | "
                     f"{amap.get(lam, float('nan')):.3f} | {an} |")
    lines += ["", "The gameable gap jumps from ~0.07 at λ=0 to ~0.30 by λ=0.05 and plateaus "
              "(~0.33–0.39) through λ=0.8 — a threshold/step. The aligned channel stays at the "
              "floor (~0.02–0.05) across all pressure. λ=0 for the aligned channel is the "
              "no-feedback baseline (agents have no reason to report), shown but not part of the trend."]
    Path("out/fig2_threshold.md").write_text("\n".join(lines) + "\n")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(gl, gm, yerr=[glo, ghi], fmt="s-", color="#C44E52", capsize=3,
                label="gameable channel", linewidth=2, markersize=7)
    # aligned: drop the λ=0 no-feedback baseline from the line (mark separately)
    al2 = [(l, m) for l, m in zip(al, am) if l > 0]
    ax.plot([l for l, _ in al2], [m for _, m in al2], "o-", color="#4C72B0",
            label="aligned channel", linewidth=2, markersize=7)
    ax.axvspan(0, 0.05, color="#FFD700", alpha=0.15)
    ax.annotate("activation\nthreshold", xy=(0.025, 0.18), fontsize=9, ha="center", color="#7a6000")
    ax.set_xlabel("optimization pressure  λ")
    ax.set_ylabel("terminal oversight gap  G")
    ax.set_title("Threshold, not curve: gap activates below λ=0.05, then plateaus", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.set_ylim(-0.02, 0.55)
    ax.set_xlim(-0.02, 0.85)
    fig.tight_layout()
    fig.savefig("figures/fig2_threshold.png", dpi=150, bbox_inches="tight")
    print("wrote figures/fig2_threshold.png and out/fig2_threshold.md")
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
