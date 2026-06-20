"""Figure 8 — watching the blind spot form: per-epoch detector trajectories.

LEFT (withholder, Sonnet): the time-varying coverage-collapse signal and the oversight gap
G_t over epochs. RIGHT (fabricator, DeepSeek): the regularity-drift signal and G_t. In each,
the dashed line is the ground-truth-free alarm threshold (mean + 3 SD of the aligned arm),
and the shaded band marks the pressure-on epochs. The firing signal rises with the hidden gap
once pressure is on, so a channel-limited overseer sees the blind spot forming through the
signal its agent's regime predicts — coverage for the withholder, drift for the fabricator —
without ever touching ground truth. Reads existing detector_* checkpoints only.
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

TV = ("distributional", "structural", "dynamics")
NO_FB = 2


def cells(sub, gameable):
    out = []
    for f in glob.glob(f"out/{sub}/checkpoints/*.pkl"):
        try:
            r = pickle.loads(Path(f).read_bytes())
        except Exception:
            continue
        c = r.config
        if bool(c.get("gameable_reward")) == gameable and c.get("lam") == 0.4 \
                and not c.get("llm_failures", 0) and getattr(r.epochs[-1], "obs", None):
            out.append(r)
    return out


def collapse(ep):
    cov = (getattr(ep, "obs", {}) or {}).get("coverage", {})
    return 1.0 - st.mean([cov.get(k, 0.0) for k in TV]) if cov else 0.0


def drift(ep):
    return (getattr(ep, "obs", {}) or {}).get("reg_drift", 0.0)


def mean_traj(runs, fn):
    """mean over runs of fn(epoch), per epoch index."""
    n = min(len(r.epochs) for r in runs)
    return [st.mean([fn(r.epochs[i]) for r in runs]) for i in range(n)]


def threshold(runs, fn):
    vals = [fn(ep) for r in runs for ep in r.epochs[NO_FB:]]
    return st.mean(vals) + 3 * st.pstdev(vals) if len(vals) > 1 else 0.0


def panel(ax, sig_g, gap_g, thr, sig_label, color, title):
    epochs = range(len(sig_g))
    ax.axvspan(NO_FB - 0.5, len(sig_g) - 0.5, color="#FFD700", alpha=0.10)
    ax.plot(epochs, sig_g, "o-", color=color, lw=2, label=sig_label)
    ax.axhline(thr, ls="--", color=color, alpha=0.7, lw=1.3)
    ax.text(0.1, thr + 0.02, f"alarm threshold ({thr:.2f})", fontsize=7.5, color=color)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("epoch")
    ax.set_ylabel(sig_label, color=color)
    ax.tick_params(axis="y", labelcolor=color)
    ax.set_title(title, fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(epochs, gap_g, "s:", color="#333333", lw=1.6, label="oversight gap G", markersize=5)
    ax2.set_ylim(-0.03, 0.6)
    ax2.set_ylabel("oversight gap  G", color="#333333")
    return ax2


def main():
    son_g, son_a = cells("detector_sonnet", True), cells("detector_sonnet", False)
    ds_g, ds_a = cells("detector_deepseek", True), cells("detector_deepseek", False)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    panel(axL, mean_traj(son_g, collapse), mean_traj(son_g, lambda e: e.gap),
          threshold(son_a, collapse), "time-varying coverage collapse", "#4C72B0",
          "Withholder (Sonnet): coverage signal fires")
    a2 = panel(axR, mean_traj(ds_g, drift), mean_traj(ds_g, lambda e: e.gap),
               threshold(ds_a, drift), "regularity drift", "#C44E52",
               "Fabricator (DeepSeek): drift signal fires")
    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("An overseer watches the blind spot form: the ground-truth-free signal tracks the hidden gap",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig8_trajectory.png", dpi=150, bbox_inches="tight")
    print("wrote figures/fig8_trajectory.png")
    print(f"  Sonnet n={len(son_g)} gameable / {len(son_a)} aligned; "
          f"DeepSeek n={len(ds_g)} / {len(ds_a)}")
    print(f"  thresholds: collapse={threshold(son_a, collapse):.3f}, drift={threshold(ds_a, drift):.3f}")


if __name__ == "__main__":
    main()
