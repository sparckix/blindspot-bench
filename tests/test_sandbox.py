"""Capability-seal tests: a sealed agent must have no path to the experiment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goodhart.llm import sandbox


def test_disallowed_tools_are_total():
    # The deny-all posture must cover shell, file reads, and the web.
    for t in ("Bash", "Read", "Grep", "Glob", "WebFetch", "WebSearch"):
        assert t in sandbox.FULL_DISALLOWED_TOOLS


def test_sealed_kwargs_per_runtime():
    assert "claude_disallowed_tools" in sandbox.sealed_subscription_kwargs("claude")
    assert sandbox.sealed_subscription_kwargs("codex")["codex_sandbox"] == "read-only"


def test_make_sandbox_dir_is_empty_and_outside_repo():
    d = sandbox.make_sandbox_dir("test")
    try:
        assert os.path.isdir(d)
        assert os.listdir(d) == []                       # nothing to read
        assert not sandbox.is_inside(d, sandbox.project_root())
    finally:
        os.rmdir(d)


def test_refuses_dir_inside_repo():
    with pytest.raises(ValueError):
        sandbox.assert_sealed_dir(sandbox.project_root())


def test_refuses_nonempty_dir(tmp_path):
    (tmp_path / "secret.txt").write_text("the probe battery")
    with pytest.raises(ValueError):
        sandbox.assert_sealed_dir(str(tmp_path))


def test_subscription_backend_seals_or_skips():
    """If the ztare runtime is importable, a sealed SubscriptionBackend must run
    in an empty dir outside the repo and refuse a repo-internal cwd. If not
    importable, skip (the seal is still unit-tested above)."""
    try:
        from goodhart.llm.adapters import SubscriptionBackend
        from goodhart.llm.base import bootstrap_paths
        bootstrap_paths()
        be = SubscriptionBackend(runtime="claude")  # default sealed, sandbox cwd
    except Exception:
        pytest.skip("ztare subscription runtime not importable in this environment")
    assert be.sealed is True
    assert be.repo and not sandbox.is_inside(be.repo, sandbox.project_root())
    assert os.listdir(be.repo) == []
    with pytest.raises(ValueError):
        SubscriptionBackend(runtime="claude", repo=sandbox.project_root())
