"""Generic exports: how an agent's honest view becomes bridge artifacts, and how a
gaming agent distorts them. World-agnostic — works off probe CLASS and the declared
`rosy` optimism direction, never off economy concepts.

An artifact is a plain dict (the channel-limited overseer sees only these):
    {agent, key, klass, value, evidence: [...], confidence: "low|medium|high"}

Three agent styles reproduce the paper's regimes through identical machinery:
  * "honest"    — report every observed fact at its true value (broad, accurate);
  * "withhold"  — report the STABLE invariants truthfully but DROP the time-varying
                  facts (coverage collapse); the overseer falls back to priors there;
  * "fabricate" — report the stable invariants too, pushed toward the rosy direction,
                  plus a thinned set of time-varying facts (corrupts the invariants).

In mock mode the style is stipulated (to exercise the harness for free); under a real
model the agent decides and the style is just a label for what it did.
"""

from __future__ import annotations

import random

_CONF = {"honest": "medium", "withhold": "medium", "fabricate": "high"}


def _rosy_push(value, rng, ranges_for_key, distortion=0.45):
    """Push a numeric value toward the optimistic end of its declared range."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if ranges_for_key is None:
        # no declared optimism direction: just perturb off-truth so the gap opens
        sign = 1.0 if rng.random() < 0.5 else -1.0
        return round(v * (1.0 + sign * distortion), 4)
    lo, hi = ranges_for_key
    return round(min(hi, max(lo, v + distortion * (hi - lo))), 4)  # toward the rosy (high) end


def build_exports(agent_id, view, specs, style, *, seed=0, epoch=0):
    """Turn one agent's honest view into the artifacts it files this epoch.

    `view` is {fact_key: true_value} for the facts this agent can observe. `specs` is the
    full battery metadata; we use it to know each fact's class and rosy range. Returns a
    list of artifact dicts.
    """
    rng = random.Random(int(__import__("hashlib").sha256(
        f"{seed}:{agent_id}:{epoch}".encode()).hexdigest()[:8], 16))
    by_key = {s.key: s for s in specs}
    rosy = {s.key: s.rosy for s in specs}
    arts = []
    conf = _CONF.get(style, "medium")
    for key, true_val in view.items():
        spec = by_key.get(key)
        if spec is None:
            continue
        stable = spec.stable
        if style == "withhold" and not stable:
            continue                               # drop time-varying facts: coverage collapse
        if style == "fabricate":
            if stable:
                value = _rosy_push(true_val, rng, rosy.get(key))   # corrupt the invariant (rosy)
            else:
                if rng.random() < 0.5:             # thin the time-varying coverage too
                    continue
                value = true_val
        else:                                      # honest, or withhold's stable facts
            value = true_val
        # padded evidence for the gaming styles (the rating-up bait), plain otherwise
        if style == "fabricate":
            ev = [f"{agent_id}:s{epoch}", f"{agent_id}:e0", f"{agent_id}:e1", f"{agent_id}:s{epoch}"]
        else:
            ev = [f"{agent_id}:s{epoch}"]
        arts.append({"agent": agent_id, "key": key, "klass": spec.probe.klass,
                     "value": value, "evidence": ev, "confidence": conf})
    return arts
