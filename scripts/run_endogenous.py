"""Endogenous-gaming experiment (reduces conditionality). Same gameable favourability
reward, but its taste is NOT disclosed to the agents — each agent is fed only its own prior
rating and must read the reward to game. Compares against the disclosed-gameable baseline
(out/deepseek_probe) at matched λ to ask: does gaming still emerge without being told?
λ∈{0.15,0.4,0.8} × 6 seeds, API. Checkpointed/resumable, hang-guarded.

    python scripts/run_endogenous.py <backend> <model> <out_subdir>
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path.home() / "cognitive-firm" / "src"))

from goodhart.bridge.channel import Architecture
from goodhart.llm.adapters import get_call_stats, reset_call_stats
from goodhart.runner.config import RunConfig
from goodhart.runner.experiment import run
from scripts.run_campaign import ckpt_key

LAMS = tuple(float(x) for x in os.environ.get("GOODHART_LAMS", "0.15,0.4,0.8").split(","))
SEEDS = (7, 11, 13, 17, 19, 23)


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "api"
    model = sys.argv[2] if len(sys.argv) > 2 else "deepseek-chat"
    out_sub = sys.argv[3] if len(sys.argv) > 3 else "endogenous_deepseek"
    ckpt_dir = Path("out") / out_sub / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(os.environ.get("GOODHART_EPOCHS", "8"))
    configs = [RunConfig(architecture=Architecture.TYPED_STATIC, lam=lam, seed=s, epochs=epochs,
                         n_agents=8, no_feedback_epochs=2, backend=backend, model=model,
                         max_workers=6, endogenous_gaming=True, parent_memory=True)
               for lam in LAMS for s in SEEDS]
    n = len(configs)
    print(f"Endogenous-gaming [{backend}:{model}]: {n} cells (no-disclosure, λ{LAMS} × {len(SEEDS)} seeds)",
          flush=True)
    for i, c in enumerate(configs, 1):
        cp = ckpt_dir / f"{ckpt_key(c)}.pkl"
        if cp.exists():
            try:
                r = pickle.loads(cp.read_bytes())
                if r.config.get("llm_failures", 0) == 0:
                    print(f"  [{i}/{n}] λ={c.lam} seed={c.seed}: RESUMED G={r.terminal_gap:.3f}", flush=True)
                    continue
            except Exception:
                pass
        reset_call_stats()
        try:
            r = run(c)
        except Exception as e:
            print(f"  [{i}/{n}] λ={c.lam} seed={c.seed}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        st = get_call_stats()
        r.config["llm_calls"], r.config["llm_failures"] = st["calls"], st["failures"]
        tmp = cp.with_suffix(".pkl.tmp"); tmp.write_bytes(pickle.dumps(r)); tmp.replace(cp)
        pc = getattr(r.epochs[-1], "per_class", {}) or {}
        deg = "  <-- DEGRADED" if st["failures"] else ""
        print(f"  [{i}/{n}] λ={c.lam} seed={c.seed}: G={r.terminal_gap:.3f} "
              f"meanG={sum(r.gaps)/len(r.gaps):.3f} per_class={{reg:{pc.get('regularity',0):.2f} "
              f"dist:{pc.get('distributional',0):.2f} struct:{pc.get('structural',0):.2f}}} "
              f"calls={st['calls']} fail={st['failures']}{deg}", flush=True)
    print(f"Endogenous [{model}] done.", flush=True)


if __name__ == "__main__":
    main()
