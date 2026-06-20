"""03 — A real model decides (needs one API key).

Instead of a stipulated mock style, a real model is shown its in-world situation and the
in-world incentive and decides for itself what to report; honesty vs. gaming is endogenous.
Works with any OpenAI-compatible endpoint.

    OPENAI_API_KEY=...   python examples/03_real_model.py            # OpenAI
    # or point at another OpenAI-compatible provider:
    OPENAI_API_KEY=$DEEPSEEK_API_KEY OPENAI_BASE_URL=https://api.deepseek.com \\
        BSB_MODEL=deepseek-chat python examples/03_real_model.py
"""
import os, sys, pathlib, statistics as st
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from a clone, no install needed

from openai import OpenAI
from goodhart.env import OversightEnv, ServiceWorld
from goodhart.env.llm_agent import make_llm_decide

key = os.environ.get("OPENAI_API_KEY")
if not key:
    sys.exit("Set OPENAI_API_KEY (any OpenAI-compatible key) to run this example.")

client = OpenAI(api_key=key, base_url=os.environ.get("OPENAI_BASE_URL"))  # None -> OpenAI default
model = os.environ.get("BSB_MODEL", "gpt-4o")


class _Backend:  # minimal OpenAI-compatible backend kept dependency-light on purpose
    # (goodhart.llm's full backends pull in the research runtime; this needs only `openai`)
    def complete(self, prompt, system=None, max_tokens=512, label="", agent_id=""):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        r = client.chat.completions.create(model=model, messages=msgs, max_tokens=max_tokens)
        return type("R", (), {"text": r.choices[0].message.content or ""})()


def avg(series, k, last=4):
    return st.mean([pc.get(k, 0.0) for pc in series[-last:]])

decide = make_llm_decide(_Backend())
for gameable in (False, True):
    r = OversightEnv(ServiceWorld(n_agents=6), lam=0.4, gameable=gameable,
                     epochs=6, seed=7, decide=decide).run()
    arm = "gameable" if gameable else "aligned "
    print(f"{arm}  G_t={r.terminal_gap:.3f}  reg={avg(r.per_class,'regularity'):.2f} "
          f"time-varying={(avg(r.per_class,'distributional')+avg(r.per_class,'structural')+avg(r.per_class,'dynamics'))/3:.2f}")
