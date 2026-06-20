"""Oversight-Env demonstration: the paper's findings reproduce in a NON-economy world
through identical G_t scoring, and the overseer protocol is swappable.

Runs entirely in mock mode (free): the failure mode is stipulated so we can exercise the
whole harness without API calls. With a real model the agent would decide; the machinery
is identical.

Shows, on the microservice ServiceWorld:
  1. THRESHOLD     — aligned channel stays at the floor; any pressure on a gameable channel
                     opens the gap.
  2. LOCALIZATION  — under a withholder the gap concentrates in the time-varying classes and
                     spares the stable invariants, regardless of difficulty tier.
  3. FINGERPRINT   — a withholder spares the invariants; a fabricator corrupts them.
  4. PROTOCOL      — swapping in the memory-anchored overseer reduces the fabrication gap.
  5. DIFFICULTY    — per-probe: the hardest stable invariant (Tier-3 circuit breaker) is
                     spared while the easiest time-varying probe (Tier-1 latency) opens.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.env import (MemoryAnchoredOverseer, Overseer, OversightEnv,
                          ServiceWorld)


def avg_class(per_class_series, klass, last_k=4):
    vals = [pc.get(klass, 0.0) for pc in per_class_series[-last_k:]]
    return sum(vals) / len(vals) if vals else 0.0


def run(style, lam, gameable, overseer=None):
    env = OversightEnv(ServiceWorld(n_agents=8), overseer=overseer or Overseer(),
                       lam=lam, gameable=gameable, agent_style=style, epochs=8, seed=7)
    return env.run()


def main():
    print("=" * 74)
    print("BlindSpot-Bench — the paper's findings in a microservice cluster (mock)")
    print("=" * 74)

    # 1. THRESHOLD
    aligned = run("withhold", 0.4, gameable=False)
    gamed = run("withhold", 0.4, gameable=True)
    print("\n[1] THRESHOLD (terminal G_t)")
    print(f"    aligned channel : {aligned.terminal_gap:.3f}")
    print(f"    gameable channel: {gamed.terminal_gap:.3f}   "
          f"({gamed.terminal_gap / max(1e-6, aligned.terminal_gap):.1f}x)")

    # 2. LOCALIZATION (withholder)
    print("\n[2] LOCALIZATION — gameable withholder, gap by probe class (mean last 4 epochs)")
    for k in ("regularity", "distributional", "structural", "dynamics"):
        tag = "  <- STABLE invariant, spared" if k == "regularity" else ""
        print(f"    {k:<16}: {avg_class(gamed.per_class, k):.3f}{tag}")

    # 3. FINGERPRINT — withhold vs fabricate, regularity-class gap
    fab = run("fabricate", 0.4, gameable=True)
    print("\n[3] FINGERPRINT — stable-invariant (regularity) gap by failure mode")
    print(f"    withholder: {avg_class(gamed.per_class, 'regularity'):.3f}  (spares invariants)")
    print(f"    fabricator: {avg_class(fab.per_class, 'regularity'):.3f}  (corrupts invariants)")

    # 4. PROTOCOL SWAP — memory-anchored overseer vs naive, on the fabricator
    fab_def = run("fabricate", 0.4, gameable=True, overseer=MemoryAnchoredOverseer())
    print("\n[4] PROTOCOL — fabricator regularity gap, naive vs memory-anchored overseer")
    print(f"    naive          : {avg_class(fab.per_class, 'regularity'):.3f}")
    print(f"    memory-anchored: {avg_class(fab_def.per_class, 'regularity'):.3f}  (defended)")

    # 5. DIFFICULTY — per-probe, hardest-stable vs easiest-varying (withholder)
    pp = {}
    for per in gamed.per_probe[-4:]:
        for k, v in per.items():
            pp.setdefault(k, []).append(v)
    score = {k: sum(v) / len(v) for k, v in pp.items()}
    gap = {k: 1.0 - s for k, s in score.items()}
    print("\n[5] DIFFICULTY CONTROL — per-probe gap (withholder)")
    print(f"    circuit_breaker_threshold (STABLE, Tier-3 HARD): {gap.get('circuit_breaker_threshold', 0):.3f}  <- spared")
    print(f"    p99_latency_ms            (VARYING, Tier-1 EASY): {gap.get('p99_latency_ms', 0):.3f}  <- opens")
    print(f"    => gap follows STABILITY, not difficulty (difficulty predicts the opposite).")
    print("\n" + "=" * 74)
    print("Same G_t machinery as the economy. Zero economy code touched.")


if __name__ == "__main__":
    main()
