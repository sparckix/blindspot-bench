"""Stability vs difficulty: does the localization follow probe STABILITY (the claim) or
probe DIFFICULTY (codex's confound)?

For each probe we read its per-probe gap (1 - score, averaged over the pressure-on epochs
and seeds) in the gameable and aligned arms. Each probe carries a STABILITY label (stable
invariant vs time-varying) and an ordinal DIFFICULTY (1 easy .. 3 hard). The decisive cells:

  * stable + HARD  (reg.price_beta T2, reg.depletion_threshold T3) — claim: spared (~0)
  * varying + EASY (dist.highest_value_resource, struct.bottleneck_resource) — claim: opens

If easy time-varying probes open while hard stable probes stay spared, the gap follows
stability, and the difficulty account is refuted (it predicts the opposite ordering).

    python scripts/analyze_difficulty.py difficulty_sonnet [difficulty_gemini ...]
"""

from __future__ import annotations

import glob
import pickle
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path.home() / "figs_activist_loop" / "src"))

# probe_id -> (stable?, difficulty 1..3, short label). Base-world battery.
PROBES = {
    "reg.grain_regen":              (True,  1, "grain_regen (T1)"),
    "reg.tool_yield":               (True,  1, "tool_yield (T1)"),
    "reg.price_beta":               (True,  2, "price_beta (T2)"),
    "reg.depletion_threshold":      (True,  3, "depletion_thr (T3)"),
    "dist.highest_value_resource":  (False, 1, "highest_value"),
    "struct.bottleneck_resource":   (False, 1, "bottleneck"),
    "dist.intermediated_fraction":  (False, 2, "intermed_frac"),
    "dyn.changes_since_prev":       (False, 2, "changes"),
    "struct.dominant_coalition":    (False, 3, "coalition"),
}
NO_FB = 2   # pressure-on epochs are epoch >= no_feedback_epochs


def cells(sub, gameable):
    out = {}
    for f in glob.glob(f"out/{sub}/checkpoints/*.pkl"):
        try:
            r = pickle.loads(Path(f).read_bytes())
        except Exception:
            continue
        c = r.config
        if bool(c.get("gameable_reward")) != gameable or c.get("lam") != 0.4:
            continue
        if c.get("llm_failures", 0):
            continue
        out[c.get("seed")] = r
    return list(out.values())


def probe_gap(runs, pid):
    """mean over seeds of (mean over pressure-on epochs of 1 - score) for one probe."""
    per_seed = []
    for r in runs:
        scores = [ep.per_probe.get(pid) for ep in r.epochs[NO_FB:]
                  if getattr(ep, "per_probe", None) and pid in ep.per_probe]
        if scores:
            per_seed.append(1.0 - st.mean(scores))
    return st.mean(per_seed) if per_seed else None, len(per_seed)


def report(sub):
    g, a = cells(sub, True), cells(sub, False)
    print(f"\n===== {sub} : gameable n={len(g)}  aligned n={len(a)} =====")
    print(f"{'probe':<26}{'stable':<8}{'diff':<6}{'aligned':<9}{'gameable':<9}{'lift':<8}")
    buckets = {("stable", "easy"): [], ("stable", "hard"): [],
               ("vary", "easy"): [], ("vary", "hard"): []}
    for pid, (stable, diff, lab) in PROBES.items():
        ga, na = probe_gap(g, pid)
        aa, _ = probe_gap(a, pid)
        if ga is None or aa is None:
            print(f"{lab:<26}{'--- no data ---'}")
            continue
        lift = ga - aa
        print(f"{lab:<26}{('yes' if stable else 'no'):<8}{diff:<6}{aa:<9.3f}{ga:<9.3f}{lift:<+8.3f}")
        sk = "stable" if stable else "vary"
        dk = "easy" if diff == 1 else "hard"
        buckets[(sk, dk)].append(ga)
    print("  -- gameable gap by bucket (mean) --")
    for k, v in buckets.items():
        if v:
            print(f"    {k[0]:<7}+{k[1]:<5}: {st.mean(v):.3f}  (n_probes={len(v)})")
    # the decisive contrast
    sh = buckets[("stable", "hard")]
    ve = buckets[("vary", "easy")]
    if sh and ve:
        print(f"  DECISIVE: stable+HARD gap {st.mean(sh):.3f}  vs  varying+EASY gap {st.mean(ve):.3f}")
        print(f"           difficulty account predicts the OPPOSITE ordering; "
              f"{'STABILITY wins' if st.mean(ve) > st.mean(sh) else 'difficulty wins'}.")


if __name__ == "__main__":
    subs = sys.argv[1:] or ["difficulty_sonnet"]
    for s in subs:
        report(s)
