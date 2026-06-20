"""01 — Quickstart (mock, no API key, < 5 seconds).

Measure the exact oversight gap G_t in the microservice world, aligned vs. gameable.
Run: python examples/01_quickstart.py
"""
import sys, pathlib, statistics as st
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from a clone, no install needed

from goodhart.env import OversightEnv, ServiceWorld


def avg(series, k, last=4):
    return st.mean([pc.get(k, 0.0) for pc in series[-last:]])


for gameable in (False, True):
    r = OversightEnv(ServiceWorld(n_agents=8), lam=0.4, gameable=gameable,
                     agent_style="withhold", epochs=8, seed=7).run()
    arm = "gameable" if gameable else "aligned "
    print(f"{arm}  G_t={r.terminal_gap:.3f}  | per class: "
          f"regularity={avg(r.per_class,'regularity'):.2f} (stable)  "
          f"distributional={avg(r.per_class,'distributional'):.2f}  "
          f"structural={avg(r.per_class,'structural'):.2f}  "
          f"dynamics={avg(r.per_class,'dynamics'):.2f}")

print("\nThe aligned channel holds the gap at the floor; the gameable one opens it,")
print("and the gap concentrates in the time-varying classes while sparing the stable invariant.")
