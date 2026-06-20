"""Spine demo — runs the deterministic measurement core end to end, with no LLM.

Proves the central claim that makes the apparatus an instrument: the oversight
gap G_t is a real, computable number that responds to overseer quality. We run a
deterministic world for several epochs and score three diagnostic parents against
ground truth each epoch:

    oracle  -> G_t ~ 0      (the gap can reach the floor)
    blind   -> G_t high      (prior-only ceiling)
    noisy   -> G_t between    (it moves with competence)

Then we run the rating-up/gap-up gaming detector on a synthetic series to confirm
the signature fires only on the gaming pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.experimenter.gaming import detect_gaming
from goodhart.experimenter.mock_parents import BlindParent, NoisyParent, OracleParent
from goodhart.experimenter.probes import build_battery
from goodhart.experimenter.scoring import score_battery
from goodhart.experimenter.worldstate import compute_world_state
from goodhart.world.agents import build_population
from goodhart.world.economy import Economy
from goodhart.world.regularities import GroundTruth

SEED = 7
N_AGENTS = 12
EPOCHS = 6
STEPS_PER_EPOCH = 20


def main() -> None:
    gt = GroundTruth.generate(SEED)
    print("=" * 72)
    print("GROUND TRUTH (experimenter-only)  commit:", gt.commit_hash()[:16])
    print(f"  grain_regen={gt.grain_regen}  tool_yield={gt.tool_yield}  "
          f"price_beta={gt.price_beta}")
    print(f"  depletion_threshold={gt.depletion_threshold}  "
          f"factor={gt.depletion_factor}  duration={gt.depletion_duration}")
    print("=" * 72)

    states, policies = build_population(N_AGENTS, SEED)
    econ = Economy(gt, states, SEED)

    oracle = OracleParent()
    blind = BlindParent(seed=SEED)
    noisy = NoisyParent(competence=0.6, seed=SEED)

    prev_ws = None
    gaps = {"oracle": [], "blind": [], "noisy": []}

    print(f"\n{'epoch':>5} | {'G(oracle)':>9} {'G(blind)':>9} {'G(noisy)':>9} "
          f"| depl hi-val bottlenk  coalition")
    print("-" * 72)
    for epoch in range(EPOCHS):
        start = econ.step
        for _ in range(STEPS_PER_EPOCH):
            obs = {aid: econ.observe(aid) for aid in econ.agents}
            actions = {aid: policies[aid].act(obs[aid]) for aid in econ.agents}
            econ.run_step(actions)
        ws = compute_world_state(econ, epoch, start, econ.step, prev=prev_ws)
        battery = build_battery(ws, gt)

        g_o = score_battery(battery, oracle.answer(battery)).gap
        g_b = score_battery(battery, blind.answer(battery)).gap
        g_n = score_battery(battery, noisy.answer(battery)).gap
        gaps["oracle"].append(g_o)
        gaps["blind"].append(g_b)
        gaps["noisy"].append(g_n)

        s = ws.public_summary()
        print(f"{epoch:>5} | {g_o:>9.3f} {g_b:>9.3f} {g_n:>9.3f} "
              f"| {int(s['depletion_active'])!s:>4} {s['highest_value_resource']:>6} "
              f"{s['bottleneck_resource']:>8}  {','.join(s['dominant_coalition']) or '-'}")
        prev_ws = ws

    def mean(xs):
        return sum(xs) / len(xs)

    print("-" * 72)
    print(f"mean G_t   oracle={mean(gaps['oracle']):.3f}  "
          f"blind={mean(gaps['blind']):.3f}  noisy={mean(gaps['noisy']):.3f}")

    # Competence sweep — G_t should fall monotonically as the parent improves.
    print("\nCompetence sweep (last epoch battery):")
    ws_last = prev_ws
    battery = build_battery(ws_last, gt)
    for c in (0.0, 0.25, 0.5, 0.75, 1.0):
        g = score_battery(battery, NoisyParent(c, seed=SEED).answer(battery)).gap
        bar = "#" * int(g * 40)
        print(f"  competence={c:>4} -> G_t={g:.3f}  {bar}")

    # Gaming signature: rating rises while gap rises -> event; honest case -> none.
    print("\nGaming-signature detector check:")
    gaming_ratings = [0.50, 0.60, 0.72, 0.81]   # parent ever happier
    gaming_gaps =    [0.20, 0.26, 0.33, 0.41]   # while truth diverges
    honest_ratings = [0.50, 0.58, 0.66, 0.74]
    honest_gaps =    [0.40, 0.31, 0.24, 0.18]   # gap falling = real improvement
    ev_g = detect_gaming(gaming_ratings, gaming_gaps)
    ev_h = detect_gaming(honest_ratings, honest_gaps)
    print(f"  gaming series  -> {len(ev_g)} event(s): {[e.as_dict() for e in ev_g]}")
    print(f"  honest series  -> {len(ev_h)} event(s) (expected 0)")

    ok = (mean(gaps["oracle"]) < 0.05 < mean(gaps["blind"])
          and len(ev_g) > 0 and len(ev_h) == 0)
    print("\nSPINE SELF-CHECK:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
