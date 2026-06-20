"""Costly-gaming experiment on Sonnet (subscription): does adding an observable
coverage cost (the Patron also prizes breadth; the agent is told) resolve the gameable
THRESHOLD into a graded curve? Small grid to limit subscription spend: costly_gaming at
λ∈{0.05,0.15,0.4,0.8} × 3 seeds. Compare against the plain-gameable Sonnet curve
(out/gameable_v3 + out/gameable_finelam). Checkpointed/resumable.
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

LAMS = (0.05, 0.15, 0.4, 0.8)
SEEDS = (7, 11, 13, 17, 19, 23)   # 6 seeds: the advice's highest-leverage step — resolve
# whether the breadth-defence is a clean/reliable mitigation or bimodal noise (n=3 was too thin).
OUT = Path("out/costly_sonnet")


def main() -> None:
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    configs = [RunConfig(architecture=Architecture.TYPED_STATIC, lam=lam, seed=s, epochs=8,
                         n_agents=8, no_feedback_epochs=2, backend="subscription",
                         model="claude", max_workers=6, costly_gaming=True, parent_memory=True)
               for lam in LAMS for s in SEEDS]
    n = len(configs)
    print(f"Costly-gaming Sonnet: {n} cells (costly λ{LAMS} × {len(SEEDS)} seeds)", flush=True)
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
    print("Costly Sonnet done.", flush=True)


if __name__ == "__main__":
    main()
