"""BACK-TEST the localization law for confounders before building on it.

C1 scorer-harshness: is the dist/struct error a stricter-scorer artifact (categorical/set
   vs tolerant numeric) rather than real withholding?  -> if so the ALIGNED arm, scored by
   the SAME scorers at the same lambda, would also show high dist/struct. Check aligned is clean.
C2 parent-prior: is reg-class low independent of gaming (good numeric prior)?  -> compare
   reg-class in the no-feedback baseline epochs (lambda=0, epochs[:nfb]) vs pressure epochs.
C3 rosy==true / rewarded-only: is reg spared only because the rewarded regularities' rosy
   target overlaps truth, and is DeepSeek's reg-error ONLY rosy-pushing the 3 rewarded?
   -> back out the UNREWARDED-regularity error from base(4 regs,3 rewarded) vs harder(8,3).
C4 contamination/dedup: redo per-class on the pressure regime, deduped per
   (model,arm,gameable,lam,seed,epochs,agents), excluding mock and the 6-epoch smoke.

Zero-cost: existing checkpoints only.
"""

from __future__ import annotations

import glob
import pickle
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLASS_ORDER = ["regularity", "distributional", "structural", "dynamics"]
TIMEVARYING = {"distributional", "structural", "dynamics"}


def arm_label(c, d):
    if c.get("harder_world"):
        base = "harder"
    elif c.get("costly_gaming"):
        base = "costly"
    elif c.get("endogenous_gaming"):
        base = "endogenous"
    else:
        base = "base"
    arch = c.get("architecture", "A")
    if arch not in ("A", "typed_static", "TYPED_STATIC"):
        base += f"/{arch}"
    if Path(d).name == "fixed_overseer":
        base += "/fixedov"
    return base


def load_clean():
    """Return deduped runs: key -> run. Excludes mock, degraded, awareness/scrambled,
    and the 6-epoch harder smoke (keep only canonical 8-epoch harder)."""
    best = {}
    for f in glob.glob("out/**/checkpoints/*.pkl", recursive=True):
        d = str(Path(f).parent.parent)
        if Path(d).name == "harder_smoke":
            continue
        try:
            r = pickle.loads(Path(f).read_bytes())
        except Exception:
            continue
        c = r.config
        if c.get("backend") == "mock" or c.get("awareness") or c.get("scrambled"):
            continue
        if c.get("llm_failures", 0):
            continue
        model = c.get("model", "?")
        if Path(d).name == "fixed_overseer":
            am = c.get("agent_model", "")
            model = (am.split(":")[-1] if am else model)
        key = (model, arm_label(c, d), bool(c.get("gameable_reward")),
               c.get("lam"), c.get("seed"), c.get("epochs"), c.get("n_agents"))
        best[key] = (model, r)            # last clean wins (identical condition)
    return best


def pressure_class(r):
    nfb = r.config.get("no_feedback_epochs", 2)
    eps = r.epochs[nfb:] if len(r.epochs) > nfb else r.epochs
    acc = defaultdict(list)
    for ep in eps:
        for k, v in (getattr(ep, "per_class", {}) or {}).items():
            acc[k].append(v)
    return {k: st.mean(v) for k, v in acc.items()}


def baseline_class(r):
    nfb = r.config.get("no_feedback_epochs", 2)
    eps = r.epochs[:nfb]
    acc = defaultdict(list)
    for ep in eps:
        for k, v in (getattr(ep, "per_class", {}) or {}).items():
            acc[k].append(v)
    return {k: st.mean(v) for k, v in acc.items()}


def main():
    runs = load_clean()
    # group deduped runs by (model, arm, gameable)
    grp = defaultdict(list)
    for (model, arm, g, lam, seed, ep, na), (_, r) in runs.items():
        if lam == 0.0:
            continue                       # lambda=0 runs handled separately (C2)
        grp[(model, arm, g)].append(r)

    print("="*84)
    print("C1+C4 — clean pressure-regime per-class (deduped, lambda>0, no smoke). "
          "Localization real iff aligned dist/struct ~0 while gameable high.")
    for (model, arm, g) in sorted(grp, key=lambda k: (k[0], k[1], not k[2])):
        rs = grp[(model, arm, g)]
        pcs = [pressure_class(r) for r in rs]
        row = "  ".join(f"{k[:4]}={st.mean(p.get(k,0) for p in pcs):.2f}" for k in CLASS_ORDER)
        tag = "GAMEABLE" if g else "aligned "
        print(f"  {model:16} {arm:14} {tag} n={len(rs):2}  {row}")

    print("\n" + "="*84)
    print("C1 verdict — aligned dist/struct at pressure, per model (should be LOW if real):")
    for (model, arm, g), rs in sorted(grp.items()):
        if g or arm != "base":
            continue
        pcs = [pressure_class(r) for r in rs]
        ds = st.mean((p.get("distributional", 0) + p.get("structural", 0)) / 2 for p in pcs)
        print(f"  {model:16} aligned base dist/struct mean = {ds:.3f}")

    print("\n" + "="*84)
    print("C2 — reg-class in no-feedback BASELINE epochs vs PRESSURE epochs (gameable base):")
    print("  (if reg low already at baseline before pressure -> parent prior contributes)")
    for (model, arm, g), rs in sorted(grp.items()):
        if not g or arm != "base":
            continue
        base_reg = [baseline_class(r).get("regularity", 0) for r in rs]
        pres_reg = [pressure_class(r).get("regularity", 0) for r in rs]
        print(f"  {model:16} reg baseline={st.mean(base_reg):.3f}  pressure={st.mean(pres_reg):.3f}")

    print("\n" + "="*84)
    print("C3 — back out UNREWARDED-regularity error from base(4 regs,3 rewarded) vs harder(8,3).")
    print("  harder_reg*8 = base_reg*4(approx the 4 base) + 4*unrew_err  =>  unrew_err implied.")
    for model in ("claude", "deepseek-chat"):
        b = grp.get((model, "base", True))
        h = grp.get((model, "harder", True))
        if not b or not h:
            print(f"  {model}: base or harder gameable missing (b={bool(b)} h={bool(h)})")
            continue
        breg = st.mean(pressure_class(r).get("regularity", 0) for r in b)
        hreg = st.mean(pressure_class(r).get("regularity", 0) for r in h)
        # harder reg-class averages the same 4 base regs + 4 added unrewarded
        unrew = (hreg * 8 - breg * 4) / 4
        print(f"  {model:16} base_reg(4)={breg:.3f}  harder_reg(8)={hreg:.3f}  "
              f"=> implied 4 ADDED-unrewarded err ~ {unrew:.3f}  (n_h={len(h)})")
    print("  (Sonnet: implied ~0 => spares unrewarded too [L2]. DeepSeek: implied high "
          "=> distorts unrewarded too [broad fabrication, NOT rewarded-only rosy-push].)")


if __name__ == "__main__":
    main()
