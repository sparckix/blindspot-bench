"""Backend-construction tests for the LLM adapters.

HARD CONSTRAINT: these tests must never make a real, paid API or CLI call. They
exercise only construction and the fallback-to-mock path:

  * ``make_backend("mock")`` works and ``.complete`` is free.
  * ``make_backend("api")`` / ``make_backend("subscription")`` NEVER raise — they
    either return the real backend (when the operator's repos import cleanly) OR
    fall back to the deterministic ``MockBackend``.

``.complete`` is only invoked on a backend whose ``name == "mock"`` (free,
deterministic). When a real backend is constructed we assert its type/identity
but DO NOT call ``.complete`` (that would spend tokens / drive a CLI).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.llm.base import LLMBackend, LLMResult, MockBackend, make_backend


def test_mock_backend_completes_for_free():
    """``make_backend("mock")`` returns a MockBackend whose ``.complete``
    yields a non-empty ``LLMResult`` with no external calls."""
    backend = make_backend("mock")
    assert isinstance(backend, MockBackend)
    assert backend.name == "mock"
    result = backend.complete("hi", label="t")
    assert isinstance(result, LLMResult)
    assert result.text  # deterministic mock string, never empty
    assert result.backend == "mock"


@pytest.mark.parametrize("kind", ["api", "subscription"])
def test_real_backend_construction_never_raises(kind):
    """``make_backend("api"|"subscription")`` must NEVER raise: it returns either
    the real backend (operator's repos importable) or a MockBackend fallback.

    Either way the result is an ``LLMBackend``. We only call ``.complete`` when
    the fallback mock was selected (free); a real backend is asserted by type and
    its ``.complete`` is deliberately NOT invoked (no spend / no CLI dispatch).
    """
    # Construction must not raise regardless of whether the external runtimes
    # are importable — make_backend swallows construction errors and falls back.
    backend = make_backend(kind)
    assert isinstance(backend, LLMBackend)

    if backend.name == "mock":
        # Fallback path: the external runtime was unavailable. .complete is free.
        result = backend.complete("hi", label=f"{kind}-fallback")
        assert isinstance(result, LLMResult)
        assert result.text
        assert result.backend == "mock"
    else:
        # Real backend constructed cleanly. Assert identity only — never call
        # .complete here (it would spend tokens or drive a subscription CLI).
        assert backend.name == kind
        if kind == "api":
            from goodhart.llm.adapters import ApiBackend

            assert isinstance(backend, ApiBackend)
        else:
            from goodhart.llm.adapters import SubscriptionBackend

            assert isinstance(backend, SubscriptionBackend)


def test_api_backend_requires_model_id_but_falls_back():
    """``make_backend("api")`` with no ``model_id`` either falls back to mock
    (external runtime missing) or, if the runtime imports, surfaces the missing
    required kwarg as a mock fallback rather than crashing the caller."""
    # ApiBackend.__init__ requires model_id; make_backend passes **cfg, so an
    # omitted model_id raises a TypeError inside the guarded try, which
    # make_backend converts into a MockBackend fallback. Never raises.
    backend = make_backend("api")
    assert isinstance(backend, LLMBackend)
    if backend.name == "mock":
        assert isinstance(backend.complete("x", label="t"), LLMResult)
