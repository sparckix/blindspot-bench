"""Figure 5 — the observable detector and its defense boundary (four families).

LEFT: which ground-truth-free signal fires fingerprints the regime. For each model, the
coverage-collapse signal (mean 1-coverage over time-varying classes) and the regularity
drift signal, gameable vs the aligned baseline. Withholders (Sonnet, Gemini, GPT) light up
coverage collapse; the fabricator (DeepSeek) lights up drift. Every family is detected 6/6.

RIGHT: the defense boundary. Memory-anchoring's change in the regularity-class gap by model:
it reduces the gap for INCONSISTENT fabrication (Gemini) and fails for CONSISTENT fabrication
(DeepSeek, memory itself rosy); withholders have little regularity gap to act on. Detectable
does not imply defendable.

Reads detector_* checkpoints only.
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
MODELS = [("Sonnet\n(withhold)", "detector_sonnet"),
          ("Gemini\n(mild)", "detector_gemini"),
          ("GPT-4o\n(mild fab)", "detector_gpt4o"),
          ("DeepSeek\n(fabricate)", "detector_deepseek")]


def load(sub):
    g, a = [], []
    for f in sorted(glob.glob(f"out/{sub}/checkpoints/*.pkl")):
        try:
            r = pickle.loads(Path(f).read_bytes())
        except Exception:
            continue
        if r.config.get("llm_failures", 0) or not getattr(r.epochs[-1], "obs", None):
            continue
        (g if r.config.get("gameable_reward") else a).append(r)
    return g, a


def pe(r):
    nfb = r.config.get("no_feedback_epochs", 2)
    return r.epochs[nfb:]


def collapse(r):
    return [st.mean(1.0 - (e.obs.get("coverage", {}) or {}).get(k, 1.0) for k in TV) for e in pe(r)]


def drift(r):
    return [e.obs.get("reg_drift", 0.0) for e in pe(r)]


def reg_naive(r):
    return [e.per_class.get("regularity", 0.0) for e in pe(r)]


def reg_def(r):
    return [e.defended_per_class.get("regularity", e.per_class.get("regularity", 0.0)) for e in pe(r)]


def flat(rs, fn):
    return [v for r in rs for v in fn(r)]


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.7))
    x = range(len(MODELS)); w = 0.2
    col_g, col_a, dr_g, dr_a, dgap, regn = [], [], [], [], [], []
    for _, sub in MODELS:
        g, a = load(sub)
        col_g.append(st.mean(flat(g, collapse)) if g else 0)
        col_a.append(st.mean(flat(a, collapse)) if a else 0)
        dr_g.append(st.mean(flat(g, drift)) if g else 0)
        dr_a.append(st.mean(flat(a, drift)) if a else 0)
        nv = flat(g, reg_naive); df = flat(g, reg_def)
        dgap.append((st.mean(df) - st.mean(nv)) if nv else 0)
        regn.append(st.mean(nv) if nv else 0)

    # LEFT: observable signals, gameable (solid) vs aligned (hatched)
    ax1.bar([i - 1.5 * w for i in x], col_g, w, color="#C44E52", label="coverage-collapse (gameable)")
    ax1.bar([i - 0.5 * w for i in x], col_a, w, color="#C44E52", alpha=0.35, hatch="//",
            label="coverage-collapse (aligned)")
    ax1.bar([i + 0.5 * w for i in x], dr_g, w, color="#4C72B0", label="reg-drift (gameable)")
    ax1.bar([i + 1.5 * w for i in x], dr_a, w, color="#4C72B0", alpha=0.35, hatch="//",
            label="reg-drift (aligned)")
    ax1.set_xticks(list(x)); ax1.set_xticklabels([m[0] for m in MODELS], fontsize=8.5)
    ax1.set_ylabel("observable signal (ground-truth-free)")
    ax1.set_title("The signal that fires fingerprints the regime\n(all detected 6/6; collapse for "
                  "withholders, drift for fabricators)", fontsize=9.5)
    ax1.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center")
    ax1.set_ylim(0, 0.95)

    # RIGHT: defense boundary — colors/labels derived from the data (robust to N models)
    def color(dg):
        return "#55A868" if dg < -0.01 else ("#C44E52" if dg > 0.01 else "#888888")

    def label(dg, rn):
        if rn < 0.05:
            return "≈no reg-gap\nto fix"
        return "gap reduced" if dg < -0.01 else "no change\n(memory rosy)"

    bars = ax2.bar([m[0] for m in MODELS], dgap, color=[color(d) for d in dgap], width=0.6)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("Δ regularity-gap  (memory-anchored − naive)")
    ax2.set_title("Defense boundary: anchoring fixes INCONSISTENT fabrication,\nnot consistent "
                  "(detectable ≠ defendable)", fontsize=9.5)
    lo = min(-0.05, min(dgap) - 0.01); hi = max(0.03, max(dgap) + 0.012)
    ax2.set_ylim(lo, hi)
    for b, v, rn in zip(bars, dgap, regn):
        ax2.annotate(label(v, rn), (b.get_x() + b.get_width() / 2, v + (0.004 if v >= 0 else -0.004)),
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)

    fig.suptitle("An overseer can see the blind spot forming without ground truth — within a limit",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig_detector.png", dpi=150, bbox_inches="tight")
    print("wrote figures/fig_detector.png")
    for (name, _), cg, ca, dg, rn in zip(MODELS, col_g, col_a, dgap, regn):
        print(f"  {name.split(chr(10))[0]:11} collapse g/a={cg:.2f}/{ca:.2f}  reg_naive={rn:.3f}  defenseΔ={dg:+.3f}")


if __name__ == "__main__":
    main()
