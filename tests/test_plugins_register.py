"""Smoke test: each providers/<name>-acp/ plugin registers cleanly.

Requires a hermes-agent checkout on PYTHONPATH (this repo has no runtime
dependency of its own):

    HERMES_AGENT_PATH=/path/to/hermes-agent python -m pytest tests/
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROVIDERS_DIR = REPO_ROOT / "providers"


def _hermes_agent_on_path() -> bool:
    hermes_path = os.environ.get("HERMES_AGENT_PATH", "").strip()
    if hermes_path and hermes_path not in sys.path:
        sys.path.insert(0, hermes_path)
    try:
        import providers  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _hermes_agent_on_path(),
    reason="set HERMES_AGENT_PATH to a hermes-agent checkout to run these tests",
)


def _plugin_dirs():
    return sorted(p for p in PROVIDERS_DIR.iterdir() if (p / "__init__.py").exists())


@pytest.mark.parametrize("plugin_dir", _plugin_dirs(), ids=lambda p: p.name)
def test_plugin_registers_provider_and_agent(plugin_dir):
    from providers import _REGISTRY as provider_registry  # type: ignore[attr-defined]

    name = plugin_dir.name  # e.g. "claude-acp"
    agent_name = name.removesuffix("-acp")

    module_name = f"_hermes_acp_agents_test_{agent_name}"
    spec = importlib.util.spec_from_file_location(
            module_name,
            plugin_dir / "__init__.py",
            # Mirror core's loader (providers/__init__.py): plugins are
            # packages, so vendored sibling modules import relatively.
            submodule_search_locations=[str(plugin_dir)],
        )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    vendored_registry = None
    try:
        spec.loader.exec_module(module)
        # A seam-era plugin vendors its own agent registry as a sibling
        # module; a legacy plugin registers into core's (branch-only) one.
        vendored_registry = sys.modules.get(module_name + ".acp_agent_registry")
    finally:
        for suffix in ("", ".acp_agent_registry", ".acp_client"):
            sys.modules.pop(module_name + suffix, None)

    assert name in provider_registry, f"{name} did not call register_provider()"
    profile = provider_registry[name]
    assert profile.auth_type == "external_process"
    assert profile.base_url == f"acp://{agent_name}"

    if vendored_registry is not None:
        registry = vendored_registry.ACP_AGENT_REGISTRY
        get_entry = vendored_registry.get_agent_entry
    else:
        from agent.acp_agent_registry import (
            ACP_AGENT_REGISTRY as registry,
            get_agent_entry as get_entry,
        )
    assert agent_name in registry, f"{agent_name} did not call register_acp_agent()"
    entry = get_entry(agent_name)
    assert entry and entry.command
    assert entry.display_name
    assert entry.install_hint
