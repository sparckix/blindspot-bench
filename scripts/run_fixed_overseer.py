"""Fixed-overseer cross-family (the airtight design): hold the OVERSEER model constant
and vary only the inner-AGENT model, so the gap difference isolates "which model games"
from "which model oversees". Gameable channel, λ=0.4, 6 seeds, architecture A, API path.

Default: parent = Gemini (reliable free API), agents ∈ {Gemini, DeepSeek}. Compare each
cell against the same-model run (out/{gemini,deepseek}_probe) to see whether fixing the
overseer preserves the agent-gaming ranking. NOT launched automatically — costs API calls.

    set -a; . ~/figs_activist_loop/.env; set +a
    GOODHART_* env  python scripts/run_fixed_overseer.py
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

PARENT = "deepseek-chat"                          # overseer held fixed (reliable; Gemini-as-parent crawls)
AGENTS = ("deepseek-chat", "gemini-2.5-flash")    # inner-agent model varied
LAM = 0.4
SEEDS = (7, 11, 13, 17, 19, 23)
OUT = Path("out/fixed_overseer")


def main() -> None:
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # parent = PARENT (config.model); agents = agent_model. gameable on.
    configs = [RunConfig(architecture=Architecture.TYPED_STATIC, lam=LAM, seed=s, epochs=8,
                         n_agents=8, no_feedback_epochs=2, backend="api", model=PARENT,
                         agent_model=am, max_workers=6, gameable_reward=True, parent_memory=True)
               for am in AGENTS for s in SEEDS]
    n = len(configs)
    print(f"Fixed-overseer [parent={PARENT}]: {n} cells (agents {AGENTS} × λ={LAM} × {len(SEEDS)} seeds)",
          flush=True)
    for i, c in enumerate(configs, 1):
        cp = ckpt_dir / f"{ckpt_key(c)}.pkl"
        if cp.exists():
            try:
                r = pickle.loads(cp.read_bytes())
                if r.config.get("llm_failures", 0) == 0:
                    print(f"  [{i}/{n}] agent={c.agent_model} seed={c.seed}: RESUMED G={r.terminal_gap:.3f}",
                          flush=True)
                    continue
            except Exception:
                pass
        reset_call_stats()
        try:
            r = run(c)
        except Exception as e:
            print(f"  [{i}/{n}] agent={c.agent_model} seed={c.seed}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        st = get_call_stats()
        r.config["llm_calls"], r.config["llm_failures"] = st["calls"], st["failures"]
        tmp = cp.with_suffix(".pkl.tmp"); tmp.write_bytes(pickle.dumps(r)); tmp.replace(cp)
        pc = getattr(r.epochs[-1], "per_class", {}) or {}
        deg = "  <-- DEGRADED" if st["failures"] else ""
        print(f"  [{i}/{n}] agent={c.agent_model} seed={c.seed}: G={r.terminal_gap:.3f} "
              f"per_class={{reg:{pc.get('regularity',0):.2f} dist:{pc.get('distributional',0):.2f}}} "
              f"calls={st['calls']} fail={st['failures']}{deg}", flush=True)
    print("Fixed-overseer done.", flush=True)


if __name__ == "__main__":
    main()
