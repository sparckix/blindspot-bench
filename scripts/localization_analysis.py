"""LOCALIZATION analysis (goal item 1): where does the gameable-channel gap concentrate,
and which candidate law predicts it across EVERY model and arm?

Candidate laws under test:
  L1 (reward-coverage): gap concentrates in classes the favourability reward does NOT
     score. Favourability scores 3 regularity-class point values -> L1 predicts the gap
     is LOW on regularity, HIGH on dist/struct/dynamics.
  L2 (stability/observability): gap concentrates in TIME-VARYING probes (dist/struct/
     dynamics, which must be re-observed and re-exported each epoch) and stays low on
     STABLE invariants (regularity, learnable once into parent memory). Distinguished
     from L1 by: does reg-class stay low even though the reward DOES touch it? and does
     the locus move with parent_memory rather than with reward coverage?

Reports per-class AND per-probe mean error (1-score) over the pressure epochs, gameable
vs aligned, per model and arm. Zero-cost: reads existing checkpoints only.
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


def arm_label(c) -> str:
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
    if Path(c.get("_dir", "")).name == "fixed_overseer":
        base += "/fixedov"
    return base


def pressure_epochs(r):
    nfb = r.config.get("no_feedback_epochs", 2)
    return r.epochs[nfb:] if len(r.epochs) > nfb else r.epochs


def collect():
    """model -> arm -> gameable(bool) -> {'class':{k:[errs]}, 'probe':{id:[errs]}, 'gap':[..]}"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: {"class": defaultdict(list), "probe": defaultdict(list), "gap": []})))
    for f in glob.glob("out/**/checkpoints/*.pkl", recursive=True):
        try:
            r = pickle.loads(Path(f).read_bytes())
        except Exception:
            continue
        c = dict(r.config)
        c["_dir"] = str(Path(f).parent.parent)
        if c.get("awareness") or c.get("scrambled") or c.get("llm_failures", 0):
            continue
        if c.get("backend") == "mock":
            continue
        model = c.get("model", "?")
        if c.get("_dir", "").endswith("fixed_overseer"):
            # fixed-overseer: label by the AGENT model (who games), keep separate arm
            am = c.get("agent_model", "")
            model = am.split(":")[-1] if am else model
        arm = arm_label(c)
        g = bool(c.get("gameable_reward"))
        slot = data[model][arm][g]
        # mean over pressure epochs of per-class error and per-probe error
        eps = pressure_epochs(r)
        for ep in eps:
            pc = getattr(ep, "per_class", {}) or {}
            for k, v in pc.items():
                slot["class"][k].append(v)
            pp = getattr(ep, "per_probe", {}) or {}
            for pid, score in pp.items():
                slot["probe"][pid].append(1.0 - score)   # store ERROR
        slot["gap"].append(r.terminal_gap)
    return data


def m(xs):
    return st.mean(xs) if xs else float("nan")


def main():
    data = collect()
    for model in sorted(data):
        print(f"\n{'='*78}\nMODEL: {model}")
        for arm in sorted(data[model]):
            for g in (True, False):
                slot = data[model][arm].get(g)
                if not slot or not slot["gap"]:
                    continue
                tag = "GAMEABLE" if g else "aligned "
                cls = slot["class"]
                row = "  ".join(f"{k[:4]}={m(cls[k]):.2f}" for k in CLASS_ORDER if cls.get(k))
                ncells = len(slot["gap"])
                print(f"  [{arm:16}] {tag}  gap={m(slot['gap']):.3f} (n={ncells})  per-class: {row}")
        # per-probe detail for the base gameable arm (the headline)
        base = data[model].get("base", {}).get(True)
        if base and base["probe"]:
            print(f"  -- {model} base GAMEABLE per-probe error (mean over pressure epochs) --")
            for pid in sorted(base["probe"], key=lambda p: -m(base["probe"][p])):
                klass = pid.split(".")[0]
                stable = "STABLE   " if pid.startswith("reg.") else "timevary "
                print(f"       {stable} {pid:32} err={m(base['probe'][pid]):.2f}")

    # ---- law scorecard: for each (model,arm,gameable) cell, is reg-class LOW vs timevarying HIGH? ----
    print(f"\n{'='*78}\nLAW SCORECARD — per gameable cell: regularity-class err vs mean time-varying err")
    print("  (L: gap localizes to time-varying classes; reg stays low. Violations flagged.)")
    for model in sorted(data):
        for arm in sorted(data[model]):
            slot = data[model][arm].get(True)
            if not slot:
                continue
            cls = slot["class"]
            reg = m(cls["regularity"]) if cls.get("regularity") else float("nan")
            tv = [m(cls[k]) for k in TIMEVARYING if cls.get(k)]
            tvm = m(tv) if tv else float("nan")
            verdict = "ok" if (reg == reg and tvm == tvm and reg < tvm - 0.05) else \
                      ("REG-ALSO-HIGH" if reg == reg and reg >= 0.15 else "flat/low")
            print(f"  {model:18} {arm:16} reg={reg:.2f}  timevarying={tvm:.2f}  -> {verdict}")


if __name__ == "__main__":
    main()
