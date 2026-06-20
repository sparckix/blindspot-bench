"""Powered cross-family run on the raw-API path: tightens the susceptibility ranking
and adds per-family λ shape. For a given model: gameable arm at λ∈{0.15,0.4,0.8} × 6
seeds, plus aligned at λ=0.4 × 6 seeds (the floor control). Resumes the earlier 3-seed
probe checkpoints in the same out dir. Checkpointed, hang-guarded.

    python scripts/run_api_powered.py <model_id> <out_subdir>
"""

from __future__ import annotations

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

SEEDS = (7, 11, 13, 17, 19, 23)
GAMEABLE_LAMS = (0.15, 0.4, 0.8)
ALIGNED_LAMS = (0.4,)


def main() -> None:
    model = sys.argv[1]
    out_sub = sys.argv[2]
    ckpt_dir = Path("out") / out_sub / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for gm, lams in ((True, GAMEABLE_LAMS), (False, ALIGNED_LAMS)):
        for lam in lams:
            for s in SEEDS:
                configs.append(RunConfig(
                    architecture=Architecture.TYPED_STATIC, lam=lam, seed=s, epochs=8,
                    n_agents=8, no_feedback_epochs=2, backend="api", model=model,
                    max_workers=6, gameable_reward=gm, parent_memory=True))
    n = len(configs)
    print(f"Powered cross-family [{model}]: {n} cells "
          f"(gameable λ{GAMEABLE_LAMS} + aligned λ{ALIGNED_LAMS}, {len(SEEDS)} seeds)", flush=True)
    for i, c in enumerate(configs, 1):
        arm = "gameable" if c.gameable_reward else "aligned "
        cp = ckpt_dir / f"{ckpt_key(c)}.pkl"
        if cp.exists():
            try:
                r = pickle.loads(cp.read_bytes())
                if r.config.get("llm_failures", 0) == 0:
                    print(f"  [{i}/{n}] {arm} λ={c.lam} seed={c.seed}: RESUMED G={r.terminal_gap:.3f}",
                          flush=True)
                    continue
            except Exception:
                pass
        reset_call_stats()
        try:
            r = run(c)
        except Exception as e:
            print(f"  [{i}/{n}] {arm} λ={c.lam} seed={c.seed}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        st = get_call_stats()
        r.config["llm_calls"], r.config["llm_failures"] = st["calls"], st["failures"]
        tmp = cp.with_suffix(".pkl.tmp"); tmp.write_bytes(pickle.dumps(r)); tmp.replace(cp)
        deg = "  <-- DEGRADED" if st["failures"] else ""
        print(f"  [{i}/{n}] {arm} λ={c.lam} seed={c.seed}: G={r.terminal_gap:.3f} "
              f"calls={st['calls']} fail={st['failures']}{deg}", flush=True)
    print(f"Powered [{model}] done.", flush=True)


if __name__ == "__main__":
    main()
