# Sandboxing inner agents — capability boundary

## Why this matters
The whole apparatus rests on the **metadata-blind boundary (D7)**: no inner agent
may learn anything about the parent, the probe battery, the ground truth, λ, or
the fact that it is being observed. The original boundary (`experimenter/boundary.py`)
verifies only the **prompt text** — necessary but NOT sufficient.

A real inner agent backed by a **subscription CLI** (`claude` / `codex`) ships with
tools — `Bash`, `Read`, `Grep`, `Glob`, `WebFetch`, … — and runs in a working
directory. If that agent could touch the filesystem it would simply `grep` this
repo and read the probe battery, the planted ground truth, and λ directly. That is
a total break of the experiment, independent of how clean the prompt is.

So the boundary must extend from **context** to **capability**.

## The seal (`goodhart/llm/sandbox.py`)
A sealed subscription agent runs with:

1. **All tools disabled** — `FULL_DISALLOWED_TOOLS` (deny-all: shell, file reads,
   web, sub-agents). The agent can only reason from its prompt. This alone removes
   filesystem AND environment-variable exfiltration (no `Bash` → no `env`).
2. **An empty, isolated cwd outside the repo** — `make_sandbox_dir()` creates a
   fresh temp directory with nothing in it. Even a tool that slipped through finds
   nothing to read.
3. **A read-only, no-network CLI sandbox** (`SEALED_CODEX_SANDBOX = "read-only"` for
   codex; web tools disallowed for claude).
4. **A hard refusal** — `assert_sealed_dir()` raises if the cwd is non-empty or lies
   inside the project repo. `SubscriptionBackend(sealed=True)` (the default) will not
   run an agent that can see the experiment.

## What is and isn't at risk
- **`ApiBackend` (Anthropic / OpenAI / Gemini / DeepSeek via `LLMRuntime`)** — pure
  text completions, no tools, no filesystem. **Inherently sealed.** Preferred for
  inner agents when you want zero capability surface.
- **`SubscriptionBackend` (claude / codex CLIs)** — sealed by default via the above.
  Never construct it with `sealed=False` for an experiment run, and never point its
  `repo` at anything but an empty sandbox dir.
- **`MockBackend`** — no external process at all.

## Operator checklist before a real run
- [ ] Inner-agent backend is `api`, or `subscription` with `sealed=True` (default).
- [ ] No `GOODHART_*` or experiment paths are needed in the agent subprocess env.
- [ ] Run from a directory where `assert_sealed_dir` passes (the runner enforces this).
- [ ] Spot-check the post-run boundary manifest: `RunResult.boundary_all_clear` is True
      and the context-hash manifest shows zero blocklist hits.
