"""The LLM backend contract + the deterministic MockBackend.

Every consumer (parent overseer, inner LLM agents) depends only on this contract,
never on a concrete provider:

    backend.complete(prompt, *, system="", max_tokens=, temperature=0.0,
                     label="", agent_id="") -> LLMResult(text, model, tokens...)

`make_backend("mock"|"api"|"subscription", ...)` constructs one. If the real
runtimes are not importable, `make_backend` falls back to MockBackend so the
apparatus always runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResult:
    text: str
    model: str = "mock"
    input_tokens: int = 0
    output_tokens: int = 0
    backend: str = "mock"
    meta: dict = field(default_factory=dict)


class LLMBackend(ABC):
    name: str = "base"

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        label: str = "request",
        agent_id: str = "",
    ) -> LLMResult:
        ...


class MockBackend(LLMBackend):
    """Deterministic stand-in. Returns a stable hash-derived string so transport
    is testable without spend. Consumers that need *useful* free behavior should
    use their heuristic policies (selected when the backend is mock) rather than
    relying on this text. If a `scripted` map is provided, an exact prompt match
    returns the scripted response (handy for deterministic unit tests)."""

    name = "mock"

    def __init__(self, scripted: dict | None = None):
        self.scripted = scripted or {}
        self.calls: list[dict] = []

    def complete(self, prompt, *, system="", max_tokens=1024, temperature=0.0,
                 label="request", agent_id="") -> LLMResult:
        self.calls.append({"prompt": prompt, "system": system, "label": label,
                           "agent_id": agent_id})
        if prompt in self.scripted:
            text = self.scripted[prompt]
        else:
            digest = hashlib.sha256((system + "\x00" + prompt).encode()).hexdigest()
            text = f"[mock:{label}:{digest[:12]}]"
        return LLMResult(text=text, model="mock", backend="mock",
                         input_tokens=len(prompt) // 4, output_tokens=len(text) // 4)


def bootstrap_paths() -> None:
    """Add the user's runtime repos to sys.path so the real adapters can import
    them. Controlled by env vars, with sensible home-dir defaults."""
    home = os.path.expanduser("~")
    candidates = [
        os.environ.get("GOODHART_COGNITIVE_FIRM_SRC", os.path.join(home, "cognitive-firm", "src")),
        os.environ.get("GOODHART_ZTARE_SRC", os.path.join(home, "figs_activist_loop", "src")),
    ]
    for path in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def make_backend(kind: str | None = None, **cfg) -> LLMBackend:
    """Construct a backend. `kind` defaults to env GOODHART_LLM_BACKEND or 'mock'.
    Falls back to MockBackend if a real adapter can't be constructed."""
    kind = (kind or os.environ.get("GOODHART_LLM_BACKEND", "mock")).lower()
    if kind == "mock":
        return MockBackend(**cfg)
    bootstrap_paths()
    try:
        if kind == "api":
            from .adapters import ApiBackend
            return ApiBackend(**cfg)
        if kind == "subscription":
            from .adapters import SubscriptionBackend
            return SubscriptionBackend(**cfg)
    except Exception as exc:  # pragma: no cover - depends on external repos
        sys.stderr.write(f"[goodhart.llm] {kind} backend unavailable ({exc}); "
                         f"falling back to mock.\n")
        return MockBackend()
    raise ValueError(f"unknown backend kind {kind!r}")


def extract_json(text: str) -> dict | list | None:
    """Best-effort extraction of the first JSON value (object OR array) from model
    text. Whichever of ``{`` / ``[`` appears EARLIEST is parsed first, so a
    top-level array is not mistaken for its first element."""
    text = text.strip()
    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        idx = text.find(opener)
        if idx != -1:
            candidates.append((idx, opener, closer))
    candidates.sort()  # earliest-appearing structure wins
    for start, opener, closer in candidates:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None
