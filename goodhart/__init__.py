"""Goodhart on the Bridge — measurement apparatus.

Three layers with deliberately asymmetric epistemic access:

  experimenter  — omniscient; owns physics, logs, probe battery, gap, lambda, seeds
  parent        — channel-limited LLM overseer; sees the bridge only
  world         — metadata-blind society of agents in a deterministic economy

This package is built bottom-up. The deterministic *measurement spine*
(`goodhart.world` + `goodhart.experimenter`) depends on the stdlib only and is
always runnable without any API key. The bridge, parent, pressure, gaming, and
governance layers stack on top of it.
"""

__version__ = "0.2.0"
