"""Capability sealing for subscription-CLI agents.

The metadata-blind boundary (D7) as written guarantees only that no experiment
metadata appears in an agent's PROMPT. That is necessary but not sufficient when
an inner agent (or the parent) is a `claude`/`codex` subscription CLI: those CLIs
ship with tools (Bash, Read, Grep, Glob, WebFetch, ...) and run in a working
directory. A tool-enabled agent pointed anywhere near this repository could simply
`grep` the source and read the probe battery, the ground truth, λ, and the
experiment framing — collapsing the boundary completely.

This module extends the boundary from CONTEXT to CAPABILITY. A sealed subscription
agent runs with:

  1. ALL tools disabled — it can only reason from its prompt (no shell, no file
     reads, no web). This alone removes filesystem and env-var exfiltration.
  2. An isolated, EMPTY working directory that is NOT inside this repository — so
     even a tool that slipped through would find nothing to read.
  3. A read-only, no-network CLI sandbox mode (codex).

API backends (`ApiBackend` / `LLMRuntime`) are pure text completions with no tool
or filesystem access, so they are inherently sealed and need none of this.
"""

from __future__ import annotations

import os
import tempfile

# The full set of CLI tools to forbid for a sealed agent. The intent is total:
# a sealed agent has NO way to touch the filesystem, a shell, or the network. If
# the CLI gains new tools, add them here — the default posture is deny-all.
FULL_DISALLOWED_TOOLS = (
    "Bash", "Edit", "Write", "Read", "Grep", "Glob", "LS", "NotebookEdit",
    "NotebookRead", "WebSearch", "WebFetch", "Task", "TodoWrite", "MultiEdit",
    "Agent", "Computer", "mcp__",
)

# codex sandbox mode for sealed agents: no writes, no exec escalation, no network.
SEALED_CODEX_SANDBOX = "read-only"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def project_root() -> str:
    return _PROJECT_ROOT


def make_sandbox_dir(label: str = "agent", base: str | None = None) -> str:
    """Create a fresh, EMPTY working directory for a sealed agent and return it.

    The directory is created under the system temp dir (or `base`), never inside
    the project repository. The caller is responsible for cleanup; the directory
    is intentionally left empty so there is nothing for an agent to read.
    """
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
    parent = base or tempfile.gettempdir()
    path = tempfile.mkdtemp(prefix=f"goodhart-sealed-{safe_label}-", dir=parent)
    assert_sealed_dir(path)
    return path


def is_inside(path: str, root: str) -> bool:
    path = os.path.realpath(path)
    root = os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def assert_sealed_dir(path: str) -> None:
    """Refuse to proceed unless `path` is a safe sandbox: it must exist, be empty,
    and lie OUTSIDE the project repository. This is the hard guard that stops a
    real subscription agent from ever being run with sight of the experiment."""
    real = os.path.realpath(path)
    if not os.path.isdir(real):
        raise ValueError(f"sandbox dir does not exist: {path}")
    if is_inside(real, _PROJECT_ROOT):
        raise ValueError(
            f"REFUSING to run a sealed agent inside the project repo ({path}); "
            "an inner agent must never have the experiment on its filesystem.")
    if os.listdir(real):
        raise ValueError(f"sandbox dir is not empty: {path}")


def sealed_subscription_kwargs(runtime: str) -> dict:
    """Extra kwargs to pass to `run_subscription_agent_with_recovery` to seal an
    agent of the given runtime. claude → all tools disallowed; codex → read-only,
    no-network sandbox."""
    if runtime == "claude":
        return {"claude_disallowed_tools": list(FULL_DISALLOWED_TOOLS)}
    if runtime == "codex":
        return {"codex_sandbox": SEALED_CODEX_SANDBOX}
    return {}
