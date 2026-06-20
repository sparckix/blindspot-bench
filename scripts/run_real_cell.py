"""A small REAL cell on the Claude subscription path (no API billing).

Endogenous agents: each agent's exports are decided by a real Claude model from
its local view + the in-world incentive; the economy actions stay heuristic. Two
λ values (0.0 vs 0.8) on architecture A so we can see whether real agents keep the
gap low when the channel carries no consequence and let it rise (while ratings
stay high) when it does — the gaming signature, now endogenous rather than mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.analysis.report import write_report
from goodhart.bridge.channel import Architecture
from goodhart.runner.config import RunConfig
from goodhart.runner.experiment import run

CONFIGS = [
    RunConfig(lam=0.0, architecture=Architecture.TYPED_STATIC, n_agents=6, epochs=4,
              no_feedback_epochs=1, seed=7, backend="subscription", model="claude"),
    RunConfig(lam=0.8, architecture=Architecture.TYPED_STATIC, n_agents=6, epochs=4,
              no_feedback_epochs=1, seed=7, backend="subscription", model="claude"),
]


def main() -> None:
    results = []
    for cfg in CONFIGS:
        print(f"\n=== RUN {cfg.label()} (backend={cfg.backend}/{cfg.model}) ===", flush=True)
        r = run(cfg)
        results.append(r)
        print(f"  epoch | effLam nExp rating  gap   attr  labels", flush=True)
        for e in r.epochs:
            print(f"   ep{e.epoch}  | {e.eff_lam:<5} {e.n_exports:>3}  "
                  f"{e.rating:.3f} {e.gap:.3f} {str(e.gaming_attributed):>5}  {e.labels}",
                  flush=True)
        print(f"  terminalG={r.terminal_gap:.3f}  meanG={sum(r.gaps)/len(r.gaps):.3f}  "
              f"gamingEvents={r.gaming_event_count}  boundary="
              f"{'clear' if r.boundary_all_clear else 'LEAK'}", flush=True)

    print("\n=== REAL-CELL CONTRAST (Claude subscription, endogenous exports) ===", flush=True)
    for r in results:
        lam = r.config["lam"]
        post = r.gaps[r.config["no_feedback_epochs"]:]
        meanpost = sum(post) / len(post) if post else float("nan")
        print(f"  lam={lam}: terminalG={r.terminal_gap:.3f}  "
              f"meanG(pressured)={meanpost:.3f}  meanRating={sum(r.ratings)/len(r.ratings):.3f}",
              flush=True)

    path = write_report(results, "out/real_report.html",
                        title="Goodhart on the Bridge — REAL cell (Claude subscription)")
    print(f"\nReport: {path}", flush=True)


if __name__ == "__main__":
    main()
