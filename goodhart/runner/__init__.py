"""The runner — epoch loop, factor sweep, and run records.

Ties the world, the bridge, the parent, and the experimenter machinery into a
single run, then sweeps runs across the λ grid and channel architectures to trace
the Goodhart curve and evaluate the predictions P1–P6.
"""

from .config import MockBehavior, RunConfig
from .experiment import RunResult, EpochRecord, run, sweep

__all__ = ["RunConfig", "MockBehavior", "RunResult", "EpochRecord", "run", "sweep"]
