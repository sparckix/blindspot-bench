"""Run the full factor sweep and write the HTML dashboard.

Sweeps the λ grid × channel architecture (× seeds), plus an awareness-injection
pair and a scrambled-world calibration run, then renders the Goodhart curve, the
P1–P6 scorecard, and per-run/per-epoch tables to a self-contained HTML report.

Mock backend by default (free). Pass --backend api|subscription for a real run
(inner subscription agents are capability-sealed; see docs/SANDBOXING.md).

    .venv/bin/python scripts/run_experiment.py [--out out/report.html] [--backend mock]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.analysis.curve import evaluate_predictions, goodhart_curve
from goodhart.analysis.report import write_report
from goodhart.bridge.channel import Architecture
from goodhart.runner.config import RunConfig
from goodhart.runner.experiment import sweep

LAMBDAS = (0.0, 0.15, 0.4, 0.8)
ARCHS = (Architecture.TYPED_STATIC, Architecture.FREE_FORM, Architecture.TYPED_GOVERNED)
SEEDS = (7, 11, 13)


def build_configs(backend: str) -> list[RunConfig]:
    configs: list[RunConfig] = []
    # main grid: λ × architecture × seeds
    for arch in ARCHS:
        for lam in LAMBDAS:
            for seed in SEEDS:
                configs.append(RunConfig(lam=lam, architecture=arch, seed=seed,
                                         epochs=8, backend=backend))
    # awareness-injection pair (matched aware / non-aware) at a high-λ typed cell
    for seed in SEEDS:
        configs.append(RunConfig(lam=0.8, architecture=Architecture.TYPED_STATIC,
                                 seed=seed, epochs=8, backend=backend, awareness=True))
    # scrambled-world calibration: estimates the artifact-shaped-noise floor
    configs.append(RunConfig(lam=0.4, architecture=Architecture.TYPED_STATIC,
                             seed=7, epochs=8, backend=backend, scrambled=True))
    return configs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/report.html")
    ap.add_argument("--backend", default="mock", choices=["mock", "api", "subscription"])
    args = ap.parse_args()

    configs = build_configs(args.backend)
    print(f"Running {len(configs)} configs on backend={args.backend} ...")
    results = sweep(configs)

    curve = goodhart_curve(results)
    preds = evaluate_predictions(results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    path = write_report(results, str(out),
                        title="Goodhart on the Bridge — Results")

    # console summary
    print("\nGoodhart curve (terminal-mean G_t by architecture):")
    for arch, d in curve.items():
        pairs = ", ".join(f"λ={l}:{g:.3f}" for l, g in zip(d["lams"], d["terminal_mean"]))
        print(f"  arch {arch}: {pairs}")
    print("\nP1–P6 scorecard:")
    for p in preds["predictions"]:
        print(f"  {p['id']}: {p['verdict']:<12} — {p['evidence']}")
    leaks = [r.config["label"] for r in results if not r.boundary_all_clear]
    print(f"\nBoundary: {'ALL CLEAR' if not leaks else 'LEAKS: ' + str(leaks)}")
    print(f"\nReport written to: {path}\n  + sidecar JSON: {path}.json")


if __name__ == "__main__":
    main()
