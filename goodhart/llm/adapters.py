"""Concrete LLM backends that REUSE the operator's existing runtimes.

Two adapters, each a thin transport shim over a production-proven runtime in the
user's own repos (see ``docs/INTEGRATION.md`` for the reuse map):

  ApiBackend           wraps ``cognitive_firm.common.llm_runtime.LLMRuntime``
                       (Anthropic / OpenAI / Gemini / DeepSeek / OpenAI-compatible).
  SubscriptionBackend  wraps ``ztare.common.subscription_agent_runtime`` (Claude /
                       Codex CLIs driven by the operator's local subscription login,
                       with durable warm sessions).

These classes are imported LAZILY by ``goodhart.llm.base.make_backend`` and may
raise inside ``__init__`` when the external runtime is not importable;
``make_backend`` catches that and falls back to the deterministic ``MockBackend``,
so the apparatus always runs. The imports of the external runtimes therefore live
INSIDE ``__init__`` (after ``bootstrap_paths`` has put the repos on ``sys.path``).

Backends are transport only: the metadata-blind boundary (D7) is enforced on
context strings BEFORE they reach a backend, so reusing these runtimes does not
weaken the boundary guarantee.
"""

from __future__ import annotations

import os
import sys
import threading
import time

from .base import LLMBackend, LLMResult
from .sandbox import (assert_sealed_dir, make_sandbox_dir,
                      sealed_subscription_kwargs)

# Process-wide call accounting so a run can detect rate-limited / failed calls
# (which silently suppress an agent's exports and inflate that cell's gap as noise).
# Lock-guarded because parallel agent export calls increment it from worker threads.
# `seconds`/`max_s` give per-cell latency so a slow backend (e.g. cold-starting a
# subscription CLI per call) is visible rather than a mystery; set GOODHART_LLM_VERBOSE=1
# to also stream one line per call (label + wall-clock seconds).
CALL_STATS = {"calls": 0, "failures": 0, "seconds": 0.0, "max_s": 0.0}
_STATS_LOCK = threading.Lock()


def _record_call(failed: bool, seconds: float = 0.0, label: str = "") -> None:
    with _STATS_LOCK:
        CALL_STATS["calls"] += 1
        CALL_STATS["seconds"] += seconds
        if seconds > CALL_STATS["max_s"]:
            CALL_STATS["max_s"] = seconds
        if failed:
            CALL_STATS["failures"] += 1
    if os.environ.get("GOODHART_LLM_VERBOSE"):
        sys.stderr.write(f"[llm] {label or 'call'} {seconds:.1f}s"
                         f"{' FAIL' if failed else ''}\n")
        sys.stderr.flush()


def reset_call_stats() -> None:
    with _STATS_LOCK:
        CALL_STATS["calls"] = 0
        CALL_STATS["failures"] = 0
        CALL_STATS["seconds"] = 0.0
        CALL_STATS["max_s"] = 0.0


def get_call_stats() -> dict:
    with _STATS_LOCK:
        return dict(CALL_STATS)


class ApiBackend(LLMBackend):
    """API-key backed backend wrapping ``cognitive_firm``'s ``LLMRuntime``.

    A single ``LLMRuntime`` instance is constructed once and reused across calls.
    Each ``complete`` maps onto ``LLMRuntime.call_text`` and translates the
    returned ``LLMTextResponse`` into the apparatus's ``LLMResult`` contract.
    """

    name = "api"

    def __init__(
        self,
        model_id: str,
        fallback_model_ids: tuple = (),
        timeout_seconds: int = 300,
        max_tokens: int = 1024,
    ):
        """Construct an API backend pinned to ``model_id``.

        Args:
            model_id: Canonical provider model id (e.g. ``"claude-sonnet-4-6"``).
            fallback_model_ids: Optional cross-model fallback chain. An empty
                tuple is normalized to ``None`` at call time so ``LLMRuntime``
                applies its own default fallback chain.
            timeout_seconds: Per-call wall-clock budget passed to ``call_text``.
            max_tokens: Default output-token cap; overridable per ``complete``.

        Imports ``LLMRuntime`` here (assuming ``bootstrap_paths`` already ran).
        If the import fails the exception propagates so ``make_backend`` can fall
        back to the mock backend.
        """
        from cognitive_firm.common.llm_runtime import LLMRuntime

        self.model_id = model_id
        self.fallback = tuple(fallback_model_ids or ())
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self._runtime = LLMRuntime()

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
        """Run one API completion and return an ``LLMResult``.

        ``system`` is prepended to ``prompt`` because ``LLMRuntime.call_text``
        takes a single prompt string (no separate system channel). ``temperature``
        and ``agent_id`` are accepted for contract compatibility but are not part
        of ``call_text``'s knobs, so they are not forwarded.

        Token counts come from ``resp.usage`` defensively (``getattr`` with
        fallbacks across the documented ``LLMUsage`` field names), and the model
        name falls back to the requested ``model_id`` if the runtime omits it.
        """
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        _t0 = time.perf_counter()
        resp = self._runtime.call_text(
            full_prompt,
            model_id=self.model_id,
            fallback_model_ids=self.fallback or None,
            max_tokens=max_tokens,
            timeout_seconds=self.timeout_seconds,
            request_label=label,
        )
        _dt = time.perf_counter() - _t0
        usage = getattr(resp, "usage", None)
        input_tokens = 0
        output_tokens = 0
        if usage is not None:
            input_tokens = int(
                getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0
            )
            output_tokens = int(
                getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
                or 0
            )
        _record_call(not (getattr(resp, "text", "") or "").strip(), _dt,
                     f"api:{self.model_id}:{label}")
        return LLMResult(
            text=getattr(resp, "text", "") or "",
            model=getattr(resp, "model_name", None) or self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            backend="api",
            meta={
                "requested_model_id": getattr(resp, "requested_model_id", None),
                "effective_model_id": getattr(resp, "effective_model_id", None),
                "fallback_from_model_id": getattr(resp, "fallback_from_model_id", None),
            },
        )


class SubscriptionBackend(LLMBackend):
    """Subscription-CLI backend wrapping ``ztare``'s subscription agent runtime.

    Drives the operator's locally-authenticated ``claude`` / ``codex`` CLI via
    ``run_subscription_agent_with_recovery``. When ``warm`` is set and a
    ``session_dir`` is configured, durable warm sessions (keyed by runtime +
    agent id) are loaded before the run and persisted after, so successive
    dispatches for the same agent resume the same CLI conversation.
    """

    name = "subscription"

    def __init__(
        self,
        runtime: str = "claude",
        repo: str | None = None,
        session_dir: str | None = None,
        timeout_seconds: int = 300,
        warm: bool = True,
        sealed: bool = True,
    ):
        """Construct a subscription backend.

        Args:
            runtime: ``"claude"`` or ``"codex"`` (the subscription CLI to drive).
            repo: Working directory the CLI runs in. When ``sealed`` (the default
                and the only safe mode for this apparatus), an empty isolated
                sandbox directory OUTSIDE the project repo is created if ``repo``
                is None; a caller-supplied ``repo`` is asserted to be a safe,
                empty, non-repo directory or the constructor raises.
            session_dir: Directory for durable warm-session JSON. Warm sessions
                are only used when this is set AND ``warm`` is true.
            timeout_seconds: Per-dispatch wall-clock budget.
            warm: Whether to load/persist a warm session (vs always cold).
            sealed: Capability seal (D7 extended to capability). When true, ALL
                CLI tools are disabled and the cwd is an empty sandbox, so the
                agent cannot read the experiment from disk, the env, or the web.
                Disable ONLY for non-experiment uses; the apparatus runner never
                does.

        Imports the ztare runtime helpers here; an import failure propagates so
        ``make_backend`` falls back to the mock backend.
        """
        from ztare.common.subscription_agent_runtime import (
            get_or_create_warm_session,
            persist_warm_session,
            run_subscription_agent_with_recovery,
        )

        self.runtime = runtime
        self.sealed = sealed
        if sealed:
            if repo is None:
                repo = make_sandbox_dir(runtime)
            else:
                assert_sealed_dir(repo)  # caller-supplied cwd must also be safe
        self.repo = repo if repo is not None else "."
        self.session_dir = session_dir
        self.timeout_seconds = timeout_seconds
        self.warm = warm
        self._get_or_create_warm_session = get_or_create_warm_session
        self._persist_warm_session = persist_warm_session
        self._run = run_subscription_agent_with_recovery

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
        """Run one subscription-CLI dispatch and return an ``LLMResult``.

        The agent-scoped id is ``agent_id`` (preferred) or ``label`` as a
        fallback — it keys the durable warm session and scopes the CLI
        conversation. ``system`` is prepended to ``prompt`` (the CLIs take one
        prompt string). ``max_tokens`` and ``temperature`` are not knobs of the
        subscription runtime and are accepted only for contract compatibility.

        A nonzero CLI return code does NOT crash the apparatus: the stdout (which
        may still hold partial work) is returned as ``text`` and the failure is
        recorded in ``meta`` (return code + a tail of stderr).
        """
        scoped_id = agent_id or label or "agent"
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        session_state = None
        use_warm = bool(self.warm and self.session_dir)
        if use_warm:
            session_state = self._get_or_create_warm_session(
                self.session_dir,
                runtime=self.runtime,
                agent_id=scoped_id,
            )

        extra = sealed_subscription_kwargs(self.runtime) if self.sealed else {}
        _t0 = time.perf_counter()
        run = self._run(
            runtime=self.runtime,
            prompt=full_prompt,
            agent_id=scoped_id,
            repo=self.repo,
            session_state=session_state,
            timeout_seconds=self.timeout_seconds,
            **extra,
        )
        _dt = time.perf_counter() - _t0

        if use_warm:
            self._persist_warm_session(
                self.session_dir,
                runtime=self.runtime,
                agent_id=scoped_id,
                session_state=run.final_session_state,
            )

        result = run.result
        stderr = result.stderr or ""
        meta = {
            "returncode": result.returncode,
            "stderr_tail": stderr[-500:] if stderr else "",
            "agent_id": scoped_id,
            "recovery_note": run.recovery_note,
        }
        failed = result.returncode != 0 or not (result.stdout or "").strip()
        _record_call(failed, _dt, f"sub:{self.runtime}:{label}")
        if failed:
            meta["error"] = f"subscription {self.runtime} exited {result.returncode}"

        return LLMResult(
            text=result.stdout or "",
            model=self.runtime,
            backend="subscription",
            meta=meta,
        )
