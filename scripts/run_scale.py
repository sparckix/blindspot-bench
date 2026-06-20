"""Scale probe (roadmap item 4, cheap version): does coverage-collapse survive a bigger
world? Gameable typed-static A on DeepSeek at λ=0.4, but with the agent count and horizon
doubled (env GOODHART_AGENTS, GOODHART_EPOCHS). Compares against the base gameable-A DeepSeek
cell (8 agents / 8 epochs, gap 0.27). Checkpointed/resumable, hang-guarded.

    GOODHART_AGENTS=16 GOODHART_EPOCHS=16 python scripts/run_scale.py api deepseek-chat scale_deepseek
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

LAM = 0.4
SEEDS = tuple(int(x) for x in os.environ.get("GOODHART_SEEDS", "7,11,13,17,19,23").split(","))


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "api"
    model = sys.argv[2] if len(sys.argv) > 2 else "deepseek-chat"
    out_sub = sys.argv[3] if len(sys.argv) > 3 else "scale_deepseek"
    agents = int(os.environ.get("GOODHART_AGENTS", "16"))
    epochs = int(os.environ.get("GOODHART_EPOCHS", "16"))
    ckpt_dir = Path("out") / out_sub / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    configs = [RunConfig(architecture=Architecture.TYPED_STATIC, lam=LAM, seed=s, epochs=epochs,
                         n_agents=agents, no_feedback_epochs=2, backend=backend, model=model,
                         max_workers=int(os.environ.get("GOODHART_WORKERS", "6")),
                         gameable_reward=True, parent_memory=True)
               for s in SEEDS]
    n = len(configs)
    print(f"Scale probe [{backend}:{model}]: {n} cells (gameable A, {agents} agents × {epochs} epochs, "
          f"λ={LAM} × {len(SEEDS)} seeds)", flush=True)
    for i, c in enumerate(configs, 1):
        cp = ckpt_dir / f"{ckpt_key(c)}.pkl"
        if cp.exists():
            try:
                r = pickle.loads(cp.read_bytes())
                if r.config.get("llm_failures", 0) == 0:
                    print(f"  [{i}/{n}] seed={c.seed}: RESUMED G={r.terminal_gap:.3f}", flush=True)
                    continue
            except Exception:
                pass
        reset_call_stats()
        try:
            r = run(c)
        except Exception as e:
            print(f"  [{i}/{n}] seed={c.seed}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        st = get_call_stats()
        r.config["llm_calls"], r.config["llm_failures"] = st["calls"], st["failures"]
        tmp = cp.with_suffix(".pkl.tmp"); tmp.write_bytes(pickle.dumps(r)); tmp.replace(cp)
        pc = getattr(r.epochs[-1], "per_class", {}) or {}
        deg = "  <-- DEGRADED" if st["failures"] else ""
        print(f"  [{i}/{n}] seed={c.seed}: G={r.terminal_gap:.3f} "
              f"per_class={{reg:{pc.get('regularity',0):.2f} dist:{pc.get('distributional',0):.2f}}} "
              f"calls={st['calls']} fail={st['failures']}{deg}", flush=True)
    print(f"Scale probe [{model}] done.", flush=True)


if __name__ == "__main__":
    main()
