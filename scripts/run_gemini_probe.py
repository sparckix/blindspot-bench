"""Cross-family probe on the reliable API path (Gemini): does the coverage-collapse
blind spot replicate, or is it Sonnet-specific? Runs the gameable-vs-aligned contrast
at λ=0.4 across 3 seeds (6 cells), matched 8-epoch/8-agent config, checkpointed.

    set -a; . ~/figs_activist_loop/.env; set +a
    GOODHART_* env  /opt/homebrew/bin/python3 scripts/run_gemini_probe.py
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

LAM = 0.4
SEEDS = (7, 11, 13)
OUT = Path("out/gemini_probe")
MODEL = "gemini-2.5-flash"


def main() -> None:
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    configs = [RunConfig(architecture=Architecture.TYPED_STATIC, lam=LAM, seed=s,
                         epochs=8, n_agents=8, no_feedback_epochs=2,
                         backend="api", model=MODEL, max_workers=6,
                         gameable_reward=gm, parent_memory=True)
               for gm in (False, True) for s in SEEDS]
    n = len(configs)
    print(f"Gemini cross-family probe: {n} cells (aligned+gameable × λ={LAM} × {len(SEEDS)} seeds)",
          flush=True)
    for i, c in enumerate(configs, 1):
        arm = "gameable" if c.gameable_reward else "aligned "
        cp = ckpt_dir / f"{ckpt_key(c)}.pkl"
        if cp.exists():
            try:
                r = pickle.loads(cp.read_bytes())
                if r.config.get("llm_failures", 0) == 0:
                    print(f"  [{i}/{n}] {arm} seed={c.seed}: RESUMED terminalG={r.terminal_gap:.3f}",
                          flush=True)
                    continue
            except Exception:
                pass
        reset_call_stats()
        try:
            r = run(c)
        except Exception as e:
            print(f"  [{i}/{n}] {arm} seed={c.seed}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        st = get_call_stats()
        r.config["llm_calls"], r.config["llm_failures"] = st["calls"], st["failures"]
        tmp = cp.with_suffix(".pkl.tmp")
        tmp.write_bytes(pickle.dumps(r))
        tmp.replace(cp)
        pc = getattr(r.epochs[-1], "per_class", {}) or {}
        deg = "  <-- DEGRADED" if st["failures"] else ""
        print(f"  [{i}/{n}] {arm} seed={c.seed}: terminalG={r.terminal_gap:.3f} "
              f"meanG={sum(r.gaps)/len(r.gaps):.3f} per_class={{reg:{pc.get('regularity',0):.2f} "
              f"dist:{pc.get('distributional',0):.2f} struct:{pc.get('structural',0):.2f}}} "
              f"calls={st['calls']} fail={st['failures']}{deg}", flush=True)
    print("gemini probe done.", flush=True)


if __name__ == "__main__":
    main()
