"""Detector eval (goal item 2): base-world gameable-vs-aligned with the observable
detector instrumented (obs signals + memory-anchored defended_gap logged per epoch).
Tests whether ground-truth-free signals (per-class coverage collapse; regularity
drift/disagreement) predict the hidden per-class gap and flag gaming early, and whether
the memory-anchored defense shrinks the (fabrication) reg-class gap.

    python scripts/run_detector.py api deepseek-chat detector_deepseek
    python scripts/run_detector.py api gemini-2.5-flash detector_gemini
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

SEEDS = tuple(int(x) for x in os.environ.get("GOODHART_SEEDS", "7,11,13,17,19,23").split(","))
LAMS = tuple(float(x) for x in os.environ.get("GOODHART_LAMS", "0.4").split(","))
AGENTS = int(os.environ.get("GOODHART_AGENTS", "8"))
EPOCHS = int(os.environ.get("GOODHART_EPOCHS", "8"))
# GOODHART_ARMS: "both" (default) | "gameable" | "aligned" — lets the two arms take
# different seed sets (e.g. 6 gameable + 3 aligned for a cost-bounded subscription run).
ARMS = {"both": (True, False), "gameable": (True,), "aligned": (False,)}[
    os.environ.get("GOODHART_ARMS", "both")]


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "api"
    model = sys.argv[2] if len(sys.argv) > 2 else "deepseek-chat"
    out_sub = sys.argv[3] if len(sys.argv) > 3 else "detector_deepseek"
    ckpt_dir = Path("out") / out_sub / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    configs = [RunConfig(architecture=Architecture.TYPED_STATIC, lam=lam, seed=s, epochs=EPOCHS,
                         n_agents=AGENTS, no_feedback_epochs=2, backend=backend, model=model,
                         max_workers=6, gameable_reward=gm, parent_memory=True)
               for gm in ARMS for lam in LAMS for s in SEEDS]
    n = len(configs)
    print(f"Detector [{backend}:{model}]: {n} cells (gameable+aligned, λ{LAMS} × {len(SEEDS)} seeds, "
          f"{AGENTS}ag×{EPOCHS}ep)", flush=True)
    for i, c in enumerate(configs, 1):
        arm = "gameable" if c.gameable_reward else "aligned "
        cp = ckpt_dir / f"{ckpt_key(c)}.pkl"
        if cp.exists():
            try:
                r = pickle.loads(cp.read_bytes())
                if r.config.get("llm_failures", 0) == 0 and getattr(r.epochs[-1], "obs", None):
                    print(f"  [{i}/{n}] {arm} λ={c.lam} seed={c.seed}: RESUMED G={r.terminal_gap:.3f}", flush=True)
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
        ep = r.epochs[-1]
        cov = ep.obs.get("coverage", {})
        deg = "  <-- DEGRADED" if st["failures"] else ""
        print(f"  [{i}/{n}] {arm} λ={c.lam} seed={c.seed}: G={r.terminal_gap:.3f} "
              f"defended={ep.defended_gap:.3f} cov={{r:{cov.get('regularity',0):.2f} "
              f"d:{cov.get('distributional',0):.2f} s:{cov.get('structural',0):.2f} "
              f"y:{cov.get('dynamics',0):.2f}}} drift={ep.obs.get('reg_drift',0):.2f} "
              f"disag={ep.obs.get('reg_disagree',0):.2f} {st['calls']}calls@{st['seconds']/max(1,st['calls']):.1f}s fail={st['failures']}{deg}", flush=True)
    print(f"Detector [{model}] done.", flush=True)


if __name__ == "__main__":
    main()
