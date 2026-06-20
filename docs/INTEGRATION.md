# LLM integration — reuse map

The agent/LLM layer reuses the user's existing, production-proven runtimes rather
than reinventing invocation. Two entry points, located by an optional `sys.path`
bootstrap (config below). If neither repo is importable, the apparatus degrades to
the deterministic `MockBackend` and stays fully runnable for free.

## API-based models (Anthropic / OpenAI / Gemini / DeepSeek / OpenAI-compatible)
- `from cognitive_firm.common.llm_runtime import LLMRuntime`
- `LLMRuntime().call_text(prompt, *, model_id, max_tokens=, timeout_seconds=, fallback_model_ids=, request_label=) -> LLMTextResponse`
  - `.text`, `.usage` (token counts), `.model_name`, `.fallback_from_model_id`
- Source: `cognitive-firm/src/cognitive_firm/common/llm_runtime.py:868`
- Model name → id map: `MODEL_MAP` (line 47); fallback chains: `FALLBACK_MODEL_CHAINS` (line 170)

## Subscription CLIs (Claude / Codex via local login tokens, warm sessions)
- `from ztare.common.subscription_agent_runtime import (`
  `  get_or_create_warm_session, run_subscription_agent_with_recovery, persist_warm_session)`
- `run_subscription_agent_with_recovery(*, runtime="claude"|"codex", prompt, agent_id, repo, session_state, timeout_seconds, ...) -> SubscriptionAgentRun`
  - `.result` (`subprocess.CompletedProcess`: `.stdout`/`.stderr`/`.returncode`), `.final_session_state`
- Source: `figs_activist_loop/src/ztare/common/subscription_agent_runtime.py:441`
- Warm sessions persist as JSON keyed by `(runtime, agent_id)` and survive restarts.

## Available credentials (names only; values never read)
- `figs_activist_loop/.env`: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY` (+ unrelated service tokens)
- Subscription auth for `claude`/`codex` is the CLI's own local login state (no env key).

## Backend selection in this repo (`goodhart.llm`)
- `MockBackend`     — deterministic, free; default for tests/CI and dry runs.
- `ApiBackend`      — wraps `LLMRuntime`; `model_id` per role (parent vs inner agent).
- `SubscriptionBackend` — wraps the ztare runtime; one `agent_id` per inner agent / parent.
- Config: `GOODHART_COGNITIVE_FIRM_SRC` and `GOODHART_ZTARE_SRC` point at the two `src/`
  dirs (defaults to `~/cognitive-firm/src` and `~/figs_activist_loop/src`); `GOODHART_LLM_BACKEND`
  selects `mock|api|subscription`.

## Boundary note (D7)
Backends are transport only. Inner-agent context strings are constructed and
verified against the metadata-blind blocklist BEFORE they reach any backend, so
reuse of these runtimes does not weaken the boundary guarantee.
