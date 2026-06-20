"""Mine the void: decompose the oversight gap by probe class and show that, under a
gameable channel, the gap opens as the exact COMPLEMENT of what the reward scores —
the rewarded `regularity` class stays accurate while the unrewarded structural and
distributional classes blow up, because export coverage collapses onto the rewarded
dimension. Emits out/blind_spot.png (2 panels) and out/blind_spot.md (table).

    .venv/bin/python scripts/blind_spot_decomposition.py
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
PRESSURE = (0.15, 0.4, 0.8)
ARMS = (("aligned", "out/claude_v2/checkpoints"),
        ("gameable", "out/gameable_v3/checkpoints"))


def load(d: str) -> list:
    out = []
    for f in glob.glob(str(Path(d) / "arch_A_*.pkl")):
        r = pickle.loads(Path(f).read_bytes())
        c = r.config
        if c.get("awareness") or c.get("scrambled") or c.get("llm_failures", 0):
            continue
        out.append(r)
    return out


def per_class_pooled(runs: list) -> dict:
    """Mean terminal-epoch per-class gap, pooled over the pressure regime."""
    acc = defaultdict(list)
    for r in runs:
        if r.config["lam"] not in PRESSURE:
            continue
        pc = getattr(r.epochs[-1], "per_class", {}) or {}
        for k in CLASSES:
            if k in pc:
                acc[k].append(pc[k])
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in acc.items()}


def coverage_by_lam(runs: list) -> dict:
    acc = defaultdict(list)
    for r in runs:
        for er in r.epochs:
            acc[r.config["lam"]].append(getattr(er, "n_exports", 0))
    return {lam: sum(v) / len(v) for lam, v in sorted(acc.items())}


def main() -> None:
    data = {name: load(d) for name, d in ARMS}
    pclass = {name: per_class_pooled(rs) for name, rs in data.items()}
    cover = {name: coverage_by_lam(rs) for name, rs in data.items()}

    # ---- markdown table ----
    lines = ["# Blind-spot decomposition (arch A, pressure regime λ∈{0.15,0.4,0.8})", "",
             "Terminal-epoch oversight gap by probe class. `regularity` is the ONLY class the",
             "gameable reward scores; the gap opens in every class it does *not*.", "",
             "| probe class | aligned G | gameable G | rewarded? |", "|---|---|---|---|"]
    for k in CLASSES:
        rewarded = "**yes**" if k == "regularity" else "no"
        lines.append(f"| {k} | {pclass['aligned'].get(k,0):.3f} | "
                     f"{pclass['gameable'].get(k,0):.3f} | {rewarded} |")
    lines += ["", "Export coverage (mean facts reported per epoch):", "",
              "| λ | aligned | gameable |", "|---|---|---|"]
    for lam in (0.0, *PRESSURE):
        lines.append(f"| {lam} | {cover['aligned'].get(lam,0):.1f} | "
                     f"{cover['gameable'].get(lam,0):.1f} |")
    lines += ["", "The gameable agents keep the rewarded `regularity` class accurate while "
              "collapsing coverage (~15→~5 facts/epoch) and abandoning the unrewarded classes, "
              "where the channel-limited overseer falls back on wrong priors. The oversight gap "
              "is the complement of the reward."]
    Path("out/blind_spot.md").write_text("\n".join(lines) + "\n")

    # ---- figure: 2 panels ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    x = range(len(CLASSES))
    w = 0.38
    a = [pclass["aligned"].get(k, 0) for k in CLASSES]
    g = [pclass["gameable"].get(k, 0) for k in CLASSES]
    ax1.bar([i - w / 2 for i in x], a, w, label="aligned channel", color="#4C72B0")
    ax1.bar([i + w / 2 for i in x], g, w, label="gameable channel", color="#C44E52")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([c + ("\n(stable)" if c == "regularity" else "\n(time-varying)")
                         for c in CLASSES], fontsize=9)
    ax1.set_ylabel("terminal oversight gap  G")
    ax1.set_title("Stable invariants stay accurate; the gap fills the time-varying classes",
                  fontsize=10)
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.axvspan(-0.5, 0.5, color="#FFD700", alpha=0.12)  # highlight rewarded class

    lams = [0.0, *PRESSURE]
    ax2.plot(lams, [cover["aligned"].get(l, 0) for l in lams], "o-",
             color="#4C72B0", label="aligned channel")
    ax2.plot(lams, [cover["gameable"].get(l, 0) for l in lams], "s-",
             color="#C44E52", label="gameable channel")
    ax2.set_xlabel("optimization pressure  λ")
    ax2.set_ylabel("facts reported per epoch")
    ax2.set_title("Coverage collapses under the gameable reward", fontsize=11)
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_ylim(0, max(max(cover["aligned"].values()), max(cover["gameable"].values())) * 1.15)

    fig.suptitle("Coverage collapse: the withholding regime (Sonnet)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig1_blind_spot.png", dpi=150, bbox_inches="tight")
    fig.savefig("out/blind_spot.png", dpi=150, bbox_inches="tight")
    print("wrote figures/fig1_blind_spot.png and out/blind_spot.md")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
