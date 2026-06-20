"""02 — Swap the oversight protocol (mock, no API key).

The core use case: hold the world and the agents fixed, swap the OVERSEER, and read
which protocol closes the gap. Here a naive overseer vs. a memory-anchored one, against a
fabricator (which corrupts stable invariants).
Run: python examples/02_swap_protocol.py
"""
import sys, pathlib, statistics as st
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from a clone, no install needed

from goodhart.env import OversightEnv, ServiceWorld, Overseer, MemoryAnchoredOverseer


def reg_gap(series, last=4):
    return st.mean([pc.get("regularity", 0.0) for pc in series[-last:]])


for overseer in (Overseer(), MemoryAnchoredOverseer()):
    r = OversightEnv(ServiceWorld(n_agents=8), overseer=overseer, lam=0.4,
                     gameable=True, agent_style="fabricate", epochs=8, seed=7).run()
    print(f"{overseer.name:<16} terminal G_t={r.terminal_gap:.3f}  "
          f"stable-invariant gap={reg_gap(r.per_class):.3f}")

print("\nThe memory-anchored protocol trusts accumulated belief over a drifting claim,")
print("reducing the fabricator's corruption of the stable invariants. Swap in your own")
print("Overseer subclass (debate, weak-to-strong, ...) to benchmark a protocol on exact G_t.")
