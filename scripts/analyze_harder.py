"""Harder-world test: does adding unrewarded regularities make the coverage-collapse
blind spot SPREAD into the regularity class?

The registered prediction: under the gameable reward, agents keep the 3 rewarded
regularities accurate but drop/distort the now-5 UNrewarded ones, so the
regularity-class gap (≈0.00 in the base world for a model whose collapse was
coverage-confined) becomes NONZERO; aligned stays ≈0.

DECISIVE MODEL = the one whose BASE reg-class gap is ~0 (a clean floor to spread
FROM). For Sonnet that floor is 0.005. DeepSeek already distorts regularities in
the base world (~0.27), so it has no floor and cannot cleanly test the spread —
reported as a confounded cross-check, not confirmation.

All inference via ztare.experiment_stats (seed-paired permutation, bootstrap CI).
"""

from __future__ import annotations

import glob
import pickle
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path.home() / "figs_activist_loop" / "src"))

from ztare.experiment_stats import bootstrap_ci, paired_permutation_test  # noqa: E402

LAM = 0.4


def load(model, gameable, harder):
    """Canonical cells (agents==parent): exclude the fixed-overseer cross-model dir,
    costly/endogenous arms, awareness/scrambled, and any degraded cell."""
    seen = {}
    for f in glob.glob("out/**/checkpoints/*.pkl", recursive=True):
        if Path(f).parent.parent.name == "fixed_overseer":
            continue
        try:
            r = pickle.loads(Path(f).read_bytes())
        except Exception:
            continue
        c = r.config
        if c.get("model") != model or bool(c.get("gameable_reward")) != gameable:
            continue
        if bool(c.get("harder_world")) != harder or c.get("lam") != LAM:
            continue
        if c.get("costly_gaming") or c.get("endogenous_gaming"):
            continue
        if c.get("awareness") or c.get("scrambled") or c.get("llm_failures", 0):
            continue
        pc = getattr(r.epochs[-1], "per_class", {}) or {}
        seen[c["seed"]] = {"gap": r.terminal_gap, "reg": pc.get("regularity", 0.0),
                           "dist": pc.get("distributional", 0.0),
                           "struct": pc.get("structural", 0.0)}
    return seen


def line(tag, d, key):
    if not d:
        print(f"  {tag:24} (none)")
        return None
    vals = [v[key] for v in d.values()]
    _, lo, hi = bootstrap_ci(vals)
    ci = "[n/a]" if lo is None or hi is None else f"[{lo:.3f},{hi:.3f}]"
    print(f"  {tag:24} n={len(vals):2}  mean={st.mean(vals):.3f}  CI={ci}")
    return d


def model_block(model, floor_note):
    print(f"\n=== {model}  ({floor_note}) ===")
    bg = load(model, True, False)
    ba = load(model, False, False)
    hg = load(model, True, True)
    ha = load(model, False, True)
    print(" REGULARITY-class gap (the registered metric):")
    line("base   gameable", bg, "reg")
    line("base   aligned ", ba, "reg")
    line("HARDER gameable", hg, "reg")
    line("HARDER aligned ", ha, "reg")
    # spread test: harder-gameable reg vs base-gameable reg, seed-paired
    keys = sorted(set(bg) & set(hg))
    if keys:
        base_reg = [bg[k]["reg"] for k in keys]
        hard_reg = [hg[k]["reg"] for k in keys]
        res = paired_permutation_test(hard_reg, base_reg)
        d = st.mean(h - b for h, b in zip(hard_reg, base_reg))
        print(f"  SPREAD Δreg(harder−base, gameable) = {d:+.3f}  "
              f"paired-perm p={res.get('p_value')}  (n={len(keys)} seeds)")
    print(" overall gap:")
    line("base   gameable", bg, "gap")
    line("HARDER gameable", hg, "gap")
    print(" per-class (HARDER gameable):")
    if hg:
        for k in ("reg", "dist", "struct"):
            print(f"    {k:6} mean={st.mean(v[k] for v in hg.values()):.3f}")


def main():
    model_block("claude", "Sonnet — base reg-class ≈0.005, CLEAN FLOOR -> decisive test")
    model_block("deepseek-chat", "DeepSeek — base reg-class ≈0.27, NO floor -> confounded cross-check")


if __name__ == "__main__":
    main()
