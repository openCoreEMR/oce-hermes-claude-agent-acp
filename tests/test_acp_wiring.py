"""End-to-end wiring for picker-driven model selection over a fake ACP server.

Ports the wiring suite reviewed on this repo's PR #2 to the vendored client:
``session/set_config_option`` emission, the confirmed-value read-back, and
``resolved_model`` reaching ``completion.model``. Drives ``_run_prompt`` and
``_create_chat_completion`` against a real subprocess speaking line-delimited
JSON-RPC, records every request it receives, and asserts the emitted protocol
rather than internal state.

The client under test is the one the ``claude-acp`` plugin ships — loaded
through the same package mechanics core's plugin loader uses, so an import
regression in the vendored modules fails here, not at runtime.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def _load_plugin_client():
    """Import providers/claude-acp as a package (core-loader mechanics)."""
    plugin_dir = REPO_ROOT / "providers" / "claude-acp"
    module_name = "_hermes_acp_agents_wiring_claude"
    spec = importlib.util.spec_from_file_location(
        module_name,
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    # The plugin imports its client lazily inside create_client(); import it
    # through the package so relative imports resolve the vendored siblings.
    client_mod = importlib.import_module(module_name + ".acp_client")
    return client_mod.ACPClient


FAKE_SERVER = textwrap.dedent(
    """
    import json, sys

    log_path = sys.argv[1]
    requests = []

    def log(entry):
        requests.append(entry)
        with open(log_path, "w") as f:
            json.dump(requests, f)

    current = "sonnet"
    def options():
        return [{
            "id": "model", "category": "model", "currentValue": current,
            "options": [
                {"value": "sonnet", "name": "Claude Sonnet 5"},
                {"value": "opus[1m]", "name": "Claude Opus 5"},
            ],
        }]

    for line in sys.stdin:
        msg = json.loads(line)
        log({"method": msg.get("method"), "params": msg.get("params")})
        rid = msg.get("id")
        method = msg.get("method")
        if method == "initialize":
            result = {"protocolVersion": 1}
        elif method == "session/new":
            result = {"sessionId": "s1", "configOptions": options()}
        elif method == "session/set_config_option":
            current = msg["params"]["value"]
            result = {"configOptions": options()}
        elif method == "session/prompt":
            note = {"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": "s1",
                "update": {"sessionUpdate": "agent_message_chunk",
                           "content": {"type": "text", "text": "hello from fake"}},
            }}
            sys.stdout.write(json.dumps(note) + "\\n"); sys.stdout.flush()
            result = {"stopReason": "end_turn"}
        else:
            result = {}
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}) + "\\n")
        sys.stdout.flush()
    """
)


@pytest.fixture()
def fake_agent(tmp_path):
    ACPClient = _load_plugin_client()
    server = tmp_path / "fake_acp_server.py"
    server.write_text(FAKE_SERVER)
    log = tmp_path / "requests.json"
    client = ACPClient(
        agent_name="claude",
        acp_cwd=str(tmp_path),
        acp_command=sys.executable,
        acp_args=[str(server), str(log)],
    )
    def requests():
        return json.loads(log.read_text())
    return client, requests


def test_client_declares_aux_wrap_skip_flags():
    # Without these, main's auxiliary client re-dispatches the stdio shim
    # through an HTTP wire adapter and compression/vision breaks.
    ACPClient = _load_plugin_client()
    assert ACPClient.HERMES_SKIP_TRANSPORT_WRAP is True
    assert ACPClient.HERMES_SKIP_ASYNC_WRAP is True


def test_explicit_pick_drives_set_config_option_and_labels_result(fake_agent):
    client, requests = fake_agent
    text, reasoning, resolved = client._run_prompt(
        "hi", timeout_seconds=20, model="opus-5"
    )
    assert text == "hello from fake"
    sets = [r for r in requests() if r["method"] == "session/set_config_option"]
    assert len(sets) == 1
    assert sets[0]["params"]["configId"] == "model"
    assert sets[0]["params"]["value"] == "opus[1m]"
    # Confirmed value read back out of the set response, not assumed.
    assert resolved == "opus[1m]"


def test_sentinel_pick_makes_no_set_call_and_reports_current(fake_agent):
    client, requests = fake_agent
    text, _, resolved = client._run_prompt(
        "hi", timeout_seconds=20, model="claude-acp"
    )
    assert text == "hello from fake"
    assert [r for r in requests() if r["method"] == "session/set_config_option"] == []
    assert resolved == "sonnet"


def test_resolved_model_reaches_completion_model(fake_agent):
    client, _ = fake_agent
    completion = client._create_chat_completion(
        model="opus-5",
        messages=[{"role": "user", "content": "hi"}],
        timeout=20,
    )
    assert completion.model == "opus[1m]"
