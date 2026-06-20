# BlindSpot-Bench

A pluggable environment for measuring the **LLM oversight gap** `G_t` — the divergence
between what a channel-limited overseer *believes* about a world and what is *true* — under
optimization pressure on a gameable supervision channel.

Most scalable-oversight evaluations bottleneck on a human or LLM **proxy judge**: the
"ground truth" is itself a noisy, gameable estimate, so you can never say exactly how wrong
the overseer is. BlindSpot-Bench removes that bottleneck by running on **deterministic,
seeded worlds with an exact answer key**. The overseer still sees only a channel; the
experimenter sees the truth; `G_t = |overseer belief − ground truth|` is an exactly computed
number, not an inference.

It is an **environment, not a dataset**: you plug in your model, your world, and your
oversight protocol, and call `run()`.

```python
from goodhart.env import OversightEnv, ServiceWorld, MemoryAnchoredOverseer

env = OversightEnv(
    world=ServiceWorld(n_agents=8),          # what is being overseen
    overseer=MemoryAnchoredOverseer(),       # the oversight PROTOCOL under test
    lam=0.4, gameable=True, agent_style="fabricate",
)
r = env.run()
r.terminal_gap        # exact oversight gap G_t
r.per_class[-1]       # gap by probe class — the localization read-out
r.per_probe[-1]       # gap by individual probe — stability vs difficulty
r.detector[-1]        # ground-truth-free signals (coverage collapse, drift)
```

## Why use this instead of building your own eval?

The hard part of an oversight eval is the **answer key**, and rolling your own reintroduces
the exact problem you are trying to study: if your gold signal is a human or a stronger model,
it is itself noisy and gameable, so you end up measuring your judge as much as your system.
BlindSpot-Bench gives you an exact, deterministic `G_t` instead, plus three things that are
tedious to rebuild correctly:

- a **metadata-blind boundary** (byte-verified) so agents cannot read the eval off their own context;
- a **mechanized gaming detector** (rating-up/gap-up, no human adjudication of intent);
- a **per-probe decomposition** that tells you *where* oversight fails, not just that it did.

You bring a model, a world, or an oversight protocol; you keep the exact measurement.

**Scope, stated plainly.** Two reference worlds ship today (a resource economy and a microservice
cluster). The finding that a gamed channel's gap localizes by probe *stability* reproduces across
both — initial evidence that the effect is structural rather than domain-specific, not yet an
established generality. This is a high-rigor, low-throughput instrument for researchers who need
exact oversight measurement, not a drop-in leaderboard.

## What you can swap

| Axis | What it is | Shipped |
|------|------------|---------|
| **World** | the simulated environment being overseen | `EconomyWorld` (the paper's economy), `ServiceWorld` (a microservice cluster) |
| **Overseer** | the oversight *protocol* reading the channel | `Overseer` (naive), `MemoryAnchoredOverseer` (memory-anchored defense) |
| **Agent / model** | who decides what to export, and whether to game | mock styles (`honest`/`withhold`/`fabricate`); any real model via `decide=` |
| **Pressure / mode** | `lam`, `gameable`, `agent_style` | continuous |

### Add a world
Implement `World` (`goodhart/env/world.py`): `reset(seed)`, `agents`, and
`advance_epoch(epoch) -> EpochObs`, where an `EpochObs` carries the probe battery (each probe
tagged with a **class** — stable `regularity` vs time-varying `distributional`/`structural`/
`dynamics` — and a **difficulty tier**) and the per-agent **honest views** (what each agent
could truthfully report, gated by what it can observe). The gap math, the bridge, the
detector, and the overseer never see your domain — only probes and claims.

### Add an oversight protocol
Subclass `Overseer` (`goodhart/env/overseer.py`) and implement `answer(...)` (reconstruct
beliefs from the exported claims) and optionally `rate(...)`. This is the seam for comparing
**debate**, **weak-to-strong**, **memory-anchored defense**, or your own protocol on identical
worlds and identical agents.

### Plug in a real model
Pass `decide=make_llm_decide(backend, framing)` (`goodhart/env/llm_agent.py`). The model is
shown only its in-world situation and the in-world incentive and decides for itself what to
file — honesty vs gaming is endogenous. Every prompt is asserted boundary-safe (no overseer,
battery, ground truth, or harness vocabulary reaches an agent).

## Two design principles (read before building on this)

1. **Benchmark the *protocol*, not the model.** Model benchmarks have a brutal half-life —
   they saturate as models improve. An *oversight-protocol* benchmark does not: hold the world
   and the agents fixed and ask which overseer (debate vs w2s vs memory-anchored vs …)
   minimizes `G_t` under pressure. That question stays live across model generations. The
   `Overseer` seam exists for exactly this.

2. **Ship the disclosed *and* undisclosed arm as a pair.** A single `G_t = 0` is ambiguous:
   it can mean the agent is aligned, or that it simply ignored the incentive prompt
   (instruction-following, not honesty). Reporting both the **disclosed** arm (the channel's
   taste is stated in-world) and the **undisclosed** arm (the same reward, taste not stated)
   disambiguates: games-disclosed-but-not-undisclosed = *follows legible incentives*; games-
   both = *spontaneous*; games-neither = *aligned or inert*. The pair is a more informative
   output than any single number.

## What this measures well — and what it does not

**Well:** the *exact* size and *structure* of an oversight gap; where a gamed channel's gap
concentrates (it localizes to time-varying structure and spares stable invariants — and this
reproduces across `EconomyWorld` and `ServiceWorld`, initial evidence it is structural rather than domain-specific);
which failure mode a model exhibits (withhold vs fabricate); whether a given overseer protocol
closes the gap, and where it provably cannot (consistent fabrication of a stable value is
detectable but undefendable without a ground-truth check).

**Honestly, not:** throughput. A run is a multi-agent society over many epochs with per-agent
and per-epoch model calls — this is a **high-rigor, low-throughput instrument**, not a cheap
MMLU-style eval. It is for the labs and researchers who need exact oversight measurement, not
a drop-in leaderboard. The worlds are deliberately simplified (the toy isolates the mechanism);
their value rests on the mechanism *transferring*, which the cross-world reproduction is the
evidence for. The gameable surface is *installed* and *disclosed*, so results are conditional
on a gameable channel — the matched aligned arm is the control, not a claim of spontaneous
subversion.

## Reproduce the headline findings (free, no API key)

```
python scripts/env_demo.py        # all five findings in ServiceWorld, mock mode
```

Shows, through the identical `G_t` scoring: the threshold (aligned floor vs gamed gap), the
localization (stable invariants spared), the withhold/fabricate fingerprint, a protocol swap
(memory-anchored defense reduces the fabrication gap), and the difficulty control (the hardest
stable invariant is spared while the easiest time-varying probe opens — the gap follows
stability, not difficulty).
