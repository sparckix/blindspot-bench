"""Model-agnostic LLM access for the parent overseer and inner agents.

A thin `LLMBackend` contract with three adapters selected by config:

  MockBackend          — deterministic, free; default for tests/CI and dry runs
  ApiBackend           — wraps cognitive_firm.common.llm_runtime.LLMRuntime
  SubscriptionBackend  — wraps ztare.common.subscription_agent_runtime

See docs/INTEGRATION.md for the reuse map. Backends are transport only; the
metadata-blind boundary (D7) is enforced on context strings before they reach a
backend.
"""

from .base import LLMBackend, LLMResult, MockBackend, make_backend

__all__ = ["LLMBackend", "LLMResult", "MockBackend", "make_backend"]
