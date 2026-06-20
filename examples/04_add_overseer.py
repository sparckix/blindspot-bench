"""04 — Add your own oversight protocol (the core extension point, no API key).

The harness benchmarks PROTOCOLS, not just models. To add one, subclass `Overseer` and
implement `answer(specs, artifacts, priors) -> {probe_id: value}` (reconstruct the overseer's
beliefs from the exported claims alone, no ground truth). Optionally override `rate(...)`.
Drop it into `OversightEnv` and you have measured it against exact G_t.
Run: python examples/04_add_overseer.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from a clone, no install needed

from goodhart.env import OversightEnv, ServiceWorld, Overseer


class TrustfulOverseer(Overseer):
    """A baseline protocol: believe each claim at face value, fall back to priors. ~6 lines is
    all a new oversight protocol needs. Swap this body for debate, weak-to-strong, a learned
    monitor — anything that maps the exported claims to a belief about the world."""
    name = "trustful"

    def answer(self, specs, artifacts, priors):
        latest = {a["key"]: a["value"] for a in artifacts}
        return {s.key: latest.get(s.key, priors.get(s.key)) for s in specs}


r = OversightEnv(ServiceWorld(n_agents=8), overseer=TrustfulOverseer(),
                 lam=0.4, gameable=True, agent_style="fabricate", epochs=8, seed=7).run()
print(f"protocol={TrustfulOverseer().name}  terminal G_t={r.terminal_gap:.3f}")
print("Compare protocols by swapping `overseer=` and reading G_t — that is the harness's point.")
