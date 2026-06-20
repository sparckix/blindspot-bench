"""EconomyWorld — the paper's grain/ore/tool economy, exposed through the World interface.

This adapter proves the simulator/oversight decoupling is faithful and not just convenient
for the new world: the SAME economy that produced the paper's results plugs into the SAME
OversightEnv, so the economy is one World among others rather than the hard-wired centre.

It reuses the existing economy verbatim — `Economy`, `compute_world_state`, `build_battery`,
and the runner's tier-gated honest-observation helpers — and only translates between the
economy's bare fact names (`grain_regen`, `bottleneck_resource`, …) and the probe ids the
env keys on (`reg.grain_regen`, `struct.bottleneck_resource`, …). No economy behaviour is
changed; the paper's own harness (`runner.experiment.run`) remains the authoritative path.
"""

from __future__ import annotations

from ..experimenter.probes import build_battery
from ..experimenter.worldstate import compute_world_state
from ..overseer.parent import _FAVOR_RANGE, _NUMERIC_PRIOR
from ..runner.config import RunConfig
from ..runner.experiment import _compute_observed
from ..world.agents import build_population
from ..world.economy import Economy
from ..world.llm_agent import ROLE_REPORTS
from ..world.regularities import GroundTruth
from .world import EpochObs, ProbeSpec, World

# bare economy fact name -> probe id the battery/env key on
_FACT2PID = {
    "grain_regen": "reg.grain_regen", "tool_yield": "reg.tool_yield",
    "price_beta": "reg.price_beta", "depletion_threshold": "reg.depletion_threshold",
    "grain_spoilage": "reg.grain_spoilage", "ore_output_base": "reg.ore_output_base",
    "trade_tax": "reg.trade_tax", "craft_cost": "reg.craft_cost",
    "highest_value_resource": "dist.highest_value_resource",
    "intermediated_fraction": "dist.intermediated_fraction",
    "bottleneck_resource": "struct.bottleneck_resource",
    "coalition": "struct.dominant_coalition",
    "change_tags": "dyn.changes_since_prev",
}

# difficulty tiers, mirroring the planted-regularity tiers + the structural/dynamics probes
_DIFF = {
    "reg.grain_regen": 1, "reg.tool_yield": 1, "reg.price_beta": 2, "reg.depletion_threshold": 3,
    "reg.grain_spoilage": 2, "reg.ore_output_base": 1, "reg.trade_tax": 2, "reg.craft_cost": 1,
    "dist.highest_value_resource": 1, "dist.intermediated_fraction": 2,
    "struct.bottleneck_resource": 1, "struct.dominant_coalition": 3, "dyn.changes_since_prev": 2,
}
# numeric optimism ranges (only the production regularities; price_beta is rosy-ambiguous)
_ROSY = {"reg.grain_regen": _FAVOR_RANGE["grain_regen"],
         "reg.tool_yield": _FAVOR_RANGE["tool_yield"],
         "reg.depletion_threshold": _FAVOR_RANGE["depletion_threshold"]}


class EconomyWorld(World):
    name = "economy"

    def __init__(self, n_agents: int = 8, harder: bool = False, base_steps: int = 20):
        self.n_agents = n_agents
        self.harder = harder
        self.base_steps = base_steps

    def reset(self, seed: int) -> None:
        self.seed = seed
        self.gt = GroundTruth.generate(seed, harder_world=self.harder)
        states, self.policies = build_population(self.n_agents, seed)
        self.econ = Economy(self.gt, states, seed)
        self._prev_ws = None
        # config shim for the runner's tier-gated honest observation (uses only scrambled+seed)
        self._cfg = RunConfig(lam=0.0, seed=seed, n_agents=self.n_agents, harder_world=self.harder)

    @property
    def agents(self) -> dict:
        return {aid: self.econ.agents[aid].role for aid in self.econ.agents}

    def advance_epoch(self, epoch: int) -> EpochObs:
        start = self.econ.step
        for _ in range(self.base_steps):
            obs = {aid: self.econ.observe(aid) for aid in self.econ.agents}
            actions = {aid: self.policies[aid].act(obs[aid]) for aid in self.econ.agents}
            self.econ.run_step(actions)
        window = (start, self.econ.step)
        ws = compute_world_state(self.econ, epoch, start, self.econ.step, prev=self._prev_ws)
        battery = build_battery(ws, self.gt)
        observed = _compute_observed(self.gt, ws, self.econ, window, self._cfg)
        specs = self._specs(battery)
        views = self._honest_views(observed)
        self._prev_ws = ws
        truth = {p.id: (sorted(p.answer) if isinstance(p.answer, (set, frozenset)) else p.answer)
                 for p in battery}
        return EpochObs(epoch=epoch, specs=specs, honest_views=views, truth=truth)

    def _specs(self, battery) -> list[ProbeSpec]:
        out = []
        for p in battery:
            prior = _NUMERIC_PRIOR.get(p.id)
            if prior is None:
                prior = {"fraction": 0.5, "categorical": "grain"}.get(p.atype, [])
            out.append(ProbeSpec(p, difficulty=_DIFF.get(p.id, 1), rosy=_ROSY.get(p.id), prior=prior))
        return out

    def _honest_views(self, observed) -> dict:
        views: dict[str, dict] = {}
        for aid in self.econ.agents:
            role = self.econ.agents[aid].role
            allowed = ROLE_REPORTS.get(role, ())
            v = {}
            for k in allowed:
                if k in observed and observed[k] is not None:
                    pid = _FACT2PID.get(k)
                    if pid:
                        v[pid] = observed[k]
            views[aid] = v
        return views
