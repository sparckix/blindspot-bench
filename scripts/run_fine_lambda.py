"""Resolve the activation threshold of the gameable channel: sample λ between the
unpressured baseline (λ=0, gap≈0.07) and the first pressured cell (λ=0.15, gap≈0.33)
to test whether the gap's rise is a hard STEP or a graded curve. Gameable arm only
(the aligned arm is flat across λ). Same apparatus, same seeds, checkpointed/resumable.

    GOODHART_*  env + ZTARE_CLAUDE_AGENT_MODEL=claude-sonnet-4-6 \
    /opt/homebrew/bin/python3 scripts/run_fine_lambda.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.bridge.channel import Architecture
from goodhart.llm.adapters import get_call_stats, reset_call_stats
from goodhart.runner.config import RunConfig
from goodhart.runner.experiment import run
from scripts.run_campaign import ckpt_key

FINE_LAMS = (0.05, 0.10)            # bracket the 0 -> 0.15 transition
SEEDS = (7, 11, 13, 17, 19, 23)
OUT = Path("out/gameable_finelam")


def main() -> None:
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    configs = [RunConfig(architecture=Architecture.TYPED_STATIC, lam=lam, seed=s,
                         epochs=8, n_agents=8, no_feedback_epochs=2,
                         backend="subscription", model="claude", max_workers=6,
                         gameable_reward=True, parent_memory=True)
               for lam in FINE_LAMS for s in SEEDS]
    n = len(configs)
    print(f"fine-λ threshold sweep: {n} runs (gameable, λ∈{FINE_LAMS} × {len(SEEDS)} seeds)",
          flush=True)
    for i, c in enumerate(configs, 1):
        cp = ckpt_dir / f"{ckpt_key(c)}.pkl"
        if cp.exists():
            try:
                r = pickle.loads(cp.read_bytes())
                if r.config.get("llm_failures", 0) == 0:
                    print(f"  [{i}/{n}] {c.label()}: RESUMED terminalG={r.terminal_gap:.3f}",
                          flush=True)
                    continue
            except Exception:
                pass
        reset_call_stats()
        try:
            r = run(c)
        except Exception as e:
            print(f"  [{i}/{n}] {c.label()}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        st = get_call_stats()
        r.config["llm_calls"], r.config["llm_failures"] = st["calls"], st["failures"]
        tmp = cp.with_suffix(".pkl.tmp")
        tmp.write_bytes(pickle.dumps(r))
        tmp.replace(cp)
        deg = "  <-- DEGRADED" if st["failures"] else ""
        print(f"  [{i}/{n}] {c.label()}: terminalG={r.terminal_gap:.3f} "
              f"meanG={sum(r.gaps)/len(r.gaps):.3f} gaming={r.gaming_event_count} "
              f"calls={st['calls']} failures={st['failures']}{deg}", flush=True)
    print("fine-λ sweep done.", flush=True)


if __name__ == "__main__":
    main()
