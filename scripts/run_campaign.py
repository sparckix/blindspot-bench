"""Run the canonical §5 experiment with per-run checkpointing + resume.

Each completed run is pickled to out/checkpoints/ immediately, and the HTML
report is rewritten after every run, so a stop or crash never loses a completed
run: re-invoking the SAME command resumes, skipping runs whose checkpoint exists.
Each fresh run also logs how many LLM calls failed (rate-limited / empty), so a
contaminated cell is visible rather than silently inflating the gap.

    .venv/bin/python scripts/run_campaign.py                       # full mock, free
    /opt/homebrew/bin/python3 scripts/run_campaign.py \
        --backend subscription --model claude --seeds 7,11 \
        --epochs 6 --agents 8 --workers 6 --no-awareness --no-scrambled \
        --out out/campaign_real.html

Resume = run the exact same command again. Force fresh = delete out/checkpoints/.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.analysis.curve import evaluate_predictions, goodhart_curve
from goodhart.analysis.report import write_report
from goodhart.llm.adapters import get_call_stats, reset_call_stats
from goodhart.runner.campaign import campaign_size, canonical_campaign
from goodhart.runner.experiment import run


def ckpt_key(cfg) -> str:
    """A checkpoint filename unique to the EXACT run-affecting config (not just
    arch/λ/seed) so re-running with different epochs/agents/backend never resumes
    a stale result."""
    sig = (cfg.architecture.value, cfg.lam, cfg.seed, cfg.epochs, cfg.n_agents,
           cfg.no_feedback_epochs, cfg.backend, cfg.model, cfg.awareness,
           cfg.awareness_epoch, cfg.scrambled,
           # memory/learning arms change the result -> must be in the key (audit fix:
           # omitting these let a memory-arm campaign resume baseline checkpoints).
           cfg.parent_memory, cfg.agent_learning, cfg.memory_window, cfg.gameable_reward,
           cfg.costly_gaming, cfg.agent_model, cfg.endogenous_gaming,
           # harder-world plants 4 extra regularities -> a different world; must be in
           # the key so harder runs never collide with / resume base checkpoints.
           cfg.harder_world)
    h = hashlib.sha256(repr(sig).encode()).hexdigest()[:8]
    base = re.sub(r"[^A-Za-z0-9]+", "_", cfg.label()).strip("_")
    return f"{base}_{h}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", choices=["mock", "api", "subscription"])
    ap.add_argument("--model", default="claude")
    ap.add_argument("--seeds", default="7,11,13")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--agents", type=int, default=12)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-awareness", action="store_true")
    ap.add_argument("--no-scrambled", action="store_true")
    ap.add_argument("--parent-memory", action="store_true",
                    help="parent accumulates prior regularity claims across epochs (P1)")
    ap.add_argument("--agent-learning", action="store_true",
                    help="agents infer regularities from their own observation history")
    ap.add_argument("--memory-window", type=int, default=0, help="0=unbounded")
    ap.add_argument("--gameable", action="store_true",
                    help="GAMEABLE reward: parent over-weights self-assessed confidence (lying pays)")
    ap.add_argument("--archs", default="A,B,C", help="restrict architectures, e.g. A")
    ap.add_argument("--out", default="out/campaign.html")
    args = ap.parse_args()

    from goodhart.bridge.channel import Architecture
    arch_map = {"A": Architecture.TYPED_STATIC, "B": Architecture.FREE_FORM,
                "C": Architecture.TYPED_GOVERNED}
    archs = tuple(arch_map[a.strip().upper()] for a in args.archs.split(",") if a.strip())
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    configs = canonical_campaign(
        seeds=seeds, epochs=args.epochs, n_agents=args.agents,
        backend=args.backend, model=args.model, max_workers=args.workers,
        awareness_arm=not args.no_awareness, scrambled_control=not args.no_scrambled,
        parent_memory=args.parent_memory, agent_learning=args.agent_learning,
        memory_window=args.memory_window, gameable_reward=args.gameable, archs=archs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out.parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    size = campaign_size(configs)
    extras = ([] if args.no_awareness else ["awareness arm"]) + \
             ([] if args.no_scrambled else ["scrambled control"])
    extra_str = (" + " + " + ".join(extras)) if extras else ""
    done = sum(1 for c in configs if (ckpt_dir / f"{ckpt_key(c)}.pkl").exists())
    print(f"Canonical §5 campaign: {size['runs']} runs "
          f"(λ×arch×{len(seeds)} seeds{extra_str}), backend={args.backend}. "
          f"{done} already checkpointed -> {size['runs'] - done} to run.", flush=True)
    if args.backend != "mock":
        print(f"  ~{size['approx_llm_calls']} LLM calls total, workers={args.workers}.", flush=True)

    results = []
    n = len(configs)
    for i, c in enumerate(configs, 1):
        cp = ckpt_dir / f"{ckpt_key(c)}.pkl"
        cached = None
        if cp.exists():
            try:
                cached = pickle.loads(cp.read_bytes())
            except Exception:
                cached = None   # torn/corrupt checkpoint -> redo this cell, don't crash
        # A checkpoint with any failed LLM calls is contaminated (an agent's
        # exports were silently suppressed) — treat it as NOT done so resume redoes it.
        if cached is not None and cached.config.get("llm_failures", 0) == 0:
            r = cached
            print(f"  [{i}/{n}] {c.label()}: RESUMED  terminalG={r.terminal_gap:.3f}", flush=True)
        else:
            if cached is not None:
                print(f"  [{i}/{n}] {c.label()}: REDO (prior had "
                      f"{cached.config.get('llm_failures', 0)} failed calls)", flush=True)
            reset_call_stats()
            try:
                r = run(c)
            except Exception as e:  # one bad run never sinks the campaign; retried on resume
                print(f"  [{i}/{n}] {c.label()}: ERROR {type(e).__name__}: {e} "
                      f"(no checkpoint; will retry on resume)", flush=True)
                continue
            stats = get_call_stats()
            r.config["llm_calls"] = stats["calls"]
            r.config["llm_failures"] = stats["failures"]
            tmp = cp.with_suffix(".pkl.tmp")   # atomic write: temp + replace
            tmp.write_bytes(pickle.dumps(r))
            tmp.replace(cp)
            fr = (f"  calls={stats['calls']} failures={stats['failures']}"
                  + ("  <-- DEGRADED" if stats["failures"] else "")) if stats["calls"] else ""
            print(f"  [{i}/{n}] {c.label()}: terminalG={r.terminal_gap:.3f} "
                  f"meanG={sum(r.gaps)/len(r.gaps):.3f} gaming={r.gaming_event_count} "
                  f"capture={r.capture_accepted_total} "
                  f"boundary={'clear' if r.boundary_all_clear else 'LEAK'}{fr}", flush=True)
        results.append(r)
        write_report(results, str(out), title="Goodhart on the Bridge — Canonical §5 campaign")

    if not results:
        print("No results.", flush=True)
        return

    curve = goodhart_curve(results)
    preds = evaluate_predictions(results)
    print("\nGoodhart curve (terminal-mean G_t):", flush=True)
    for arch, d in curve.items():
        print(f"  arch {arch}: " + ", ".join(f"λ={l}:{g:.3f}"
              for l, g in zip(d["lams"], d["terminal_mean"])), flush=True)
    print("\nP1–P6 scorecard:", flush=True)
    for p in preds["predictions"]:
        print(f"  {p['id']}: {p['verdict']:<12} — {p['evidence']}", flush=True)

    leaks = [r.config["label"] for r in results if not r.boundary_all_clear]
    tot_calls = sum(r.config.get("llm_calls", 0) for r in results)
    tot_fail = sum(r.config.get("llm_failures", 0) for r in results)
    print(f"\nBoundary: {'ALL CLEAR' if not leaks else 'LEAKS: ' + str(leaks)}", flush=True)
    if tot_calls:
        print(f"LLM calls: {tot_calls}  failures: {tot_fail} "
              f"({100*tot_fail/tot_calls:.1f}%)", flush=True)
    print(f"Report: {out}  (+ {out}.json)  | checkpoints: {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
