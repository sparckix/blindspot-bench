"""Gameable architecture comparison (P4/P5): run the gameable reward on free-form (B)
and typed-governed (C) channels, to compare against typed-static (A, already run). Tests
whether the channel architecture amplifies or resists coverage-collapse gaming, and whether
governed C draws capture-type proposals. λ=0.4 × 6 seeds, API. Checkpointed/resumable.

    python scripts/run_arch_gameable.py <backend> <model> <out_subdir>
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

ARCHS = (Architecture.FREE_FORM, Architecture.TYPED_GOVERNED)   # B, C (A already exists)
LAM = 0.4
SEEDS = (7, 11, 13, 17, 19, 23)


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "api"
    model = sys.argv[2] if len(sys.argv) > 2 else "deepseek-chat"
    out_sub = sys.argv[3] if len(sys.argv) > 3 else "arch_gameable_deepseek"
    ckpt_dir = Path("out") / out_sub / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    configs = [RunConfig(architecture=arch, lam=LAM, seed=s, epochs=8, n_agents=8,
                         no_feedback_epochs=2, backend=backend, model=model, max_workers=6,
                         gameable_reward=True, parent_memory=True)
               for arch in ARCHS for s in SEEDS]
    n = len(configs)
    print(f"Gameable arch comparison [{backend}:{model}]: {n} cells "
          f"(archs {[a.value for a in ARCHS]} × λ={LAM} × {len(SEEDS)} seeds)", flush=True)
    for i, c in enumerate(configs, 1):
        cp = ckpt_dir / f"{ckpt_key(c)}.pkl"
        if cp.exists():
            try:
                r = pickle.loads(cp.read_bytes())
                if r.config.get("llm_failures", 0) == 0:
                    print(f"  [{i}/{n}] {c.architecture.value} seed={c.seed}: RESUMED G={r.terminal_gap:.3f}",
                          flush=True)
                    continue
            except Exception:
                pass
        reset_call_stats()
        try:
            r = run(c)
        except Exception as e:
            print(f"  [{i}/{n}] {c.architecture.value} seed={c.seed}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        st = get_call_stats()
        r.config["llm_calls"], r.config["llm_failures"] = st["calls"], st["failures"]
        tmp = cp.with_suffix(".pkl.tmp"); tmp.write_bytes(pickle.dumps(r)); tmp.replace(cp)
        pc = getattr(r.epochs[-1], "per_class", {}) or {}
        nprop = sum(getattr(e, "n_proposals", 0) for e in r.epochs)
        deg = "  <-- DEGRADED" if st["failures"] else ""
        print(f"  [{i}/{n}] {c.architecture.value} seed={c.seed}: G={r.terminal_gap:.3f} "
              f"per_class={{reg:{pc.get('regularity',0):.2f} dist:{pc.get('distributional',0):.2f}}} "
              f"proposals={nprop} captures={r.capture_accepted_total} "
              f"calls={st['calls']} fail={st['failures']}{deg}", flush=True)
    print(f"Gameable arch [{model}] done.", flush=True)


if __name__ == "__main__":
    main()
