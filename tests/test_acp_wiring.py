"""The client's protocol behavior against a fake ACP agent in a real subprocess.

The client under test is the one the ``claude-acp`` plugin ships, loaded
through the same package mechanics core's plugin loader uses, so an import
regression in the vendored modules fails here, not at runtime. Tests assert the
protocol the agent received and what Hermes gets back, not internal state.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FAKE_AGENT = REPO_ROOT / "tests" / "fake_acp_agent.py"
SONNET = "claude-sonnet-5-5"
OPUS_1M = "claude-opus-5-5[1m]"
TOOLS = [{"type": "function", "function": {"name": "memory", "description": "d", "parameters": {}}}]


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


def _load_plugin():
    """Import providers/claude-acp as a package (core-loader mechanics)."""
    plugin_dir = REPO_ROOT / "providers" / "claude-acp"
    module_name = "_hermes_acp_agents_wiring_claude"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name,
            plugin_dir / "__init__.py",
            submodule_search_locations=[str(plugin_dir)],
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return (
        sys.modules[module_name],
        importlib.import_module(module_name + ".acp_client"),
        importlib.import_module(module_name + ".acp_session"),
    )


class Harness:
    def __init__(self, tmp_path, monkeypatch):
        self.plugin, self.client_mod, self.session_mod = _load_plugin()
        self.log = tmp_path / "requests.jsonl"
        self.tmp_path = tmp_path
        monkeypatch.setenv("FAKE_ACP_LOG", str(self.log))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("OCE_CLAUDE_ACP_USE_LOGIN", "1")
        monkeypatch.setenv("HERMES_SESSION_ID", "hermes-session-1")
        (tmp_path / "home").mkdir()
        self.messages = [{"role": "system", "content": "SYSTEM TEXT"}]

    def client(self):
        return self.client_mod.ACPClient(
            agent_name="claude",
            acp_cwd=str(self.tmp_path),
            acp_command=sys.executable,
            acp_args=[str(FAKE_AGENT)],
        )

    def requests(self, method=None):
        rows = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return [row for row in rows if method is None or row["method"] == method]

    def prompts(self):
        return ["".join(block["text"] for block in row["params"]["prompt"]) for row in self.requests("session/prompt")]

    def turn(self, text, *, model=SONNET, tools=TOOLS, timeout=20, client=None):
        """A Hermes turn: a new client each time, as after a rebuild, with the whole history."""
        self.messages.append({"role": "user", "content": text})
        completion = (client or self.client())._create_chat_completion(
            model=model, messages=self.messages, tools=tools, timeout=timeout)
        message = completion.choices[0].message
        stored = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            stored["tool_calls"] = [{"id": call.id} for call in message.tool_calls]
        self.messages.append(stored)
        return completion


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    built = Harness(tmp_path, monkeypatch)
    yield built
    built.session_mod.REGISTRY.close_all()


def _adapter_pids(harness):
    return {row["pid"] for row in harness.requests()}


def test_client_declares_aux_wrap_skip_flags():
    # Without these, main's auxiliary client re-dispatches the stdio shim
    # through an HTTP wire adapter and compression/vision breaks.
    _, client_mod, _ = _load_plugin()
    assert client_mod.ACPClient.HERMES_SKIP_TRANSPORT_WRAP is True
    assert client_mod.ACPClient.HERMES_SKIP_ASYNC_WRAP is True


def test_one_adapter_and_session_across_client_rebuilds(harness):
    harness.turn("first")
    harness.turn("second")
    assert len(_adapter_pids(harness)) == 1
    assert len(harness.requests("session/new")) == 1


def test_each_prompt_sends_only_what_claude_code_has_not_seen(harness):
    harness.turn("first")
    harness.turn("second")
    assert harness.prompts() == ["first", "second"]
    # The system prompt lives in Claude Code's system prompt, which survives compaction.
    meta = harness.requests("session/new")[0]["params"]["_meta"]
    assert "SYSTEM TEXT" in meta["systemPrompt"]["append"]
    assert "memory" in meta["systemPrompt"]["append"]


def test_a_second_hermes_session_gets_its_own_adapter(harness, monkeypatch):
    harness.turn("first")
    monkeypatch.setenv("HERMES_SESSION_ID", "hermes-session-2")
    harness.messages = []
    harness.turn("other")
    assert len(_adapter_pids(harness)) == 2


def test_a_call_without_tools_never_reaches_the_session(harness):
    harness.turn("first")
    title = harness.client()._create_chat_completion(
        model=SONNET, messages=[{"role": "user", "content": "make a title"}], timeout=20)
    assert title.choices[0].message.content == "echo: make a title"
    harness.turn("second")
    assert harness.prompts() == ["first", "make a title", "second"]
    assert len(_adapter_pids(harness)) == 2


def test_session_ignores_settings_files_and_forces_the_gate(harness):
    harness.turn("first")
    options = harness.requests("session/new")[0]["params"]["_meta"]["claudeCode"]["options"]
    assert options["settingSources"] == []
    assert {"Bash", "Read", "Write", "Edit"} <= set(options["settings"]["permissions"]["ask"])


def test_mode_is_default_on_every_new_and_load(harness):
    harness.turn("first")
    with pytest.raises(harness.session_mod.ACPTurnError):
        harness.turn("CRASH")
    harness.turn("after")
    assert len(harness.requests("session/load")) == 1
    assert [row["params"]["modeId"] for row in harness.requests("session/set_mode")] == ["default", "default"]


def test_exact_model_is_pinned_and_reported(harness):
    completion = harness.turn("first", model=OPUS_1M)
    sets = harness.requests("session/set_config_option")
    assert [row["params"]["value"] for row in sets] == [OPUS_1M]
    assert completion.model == OPUS_1M


@pytest.mark.parametrize("model", ["opus-5", "claude-opus-5-5", "sonnet", "claude-acp", ""])
def test_a_model_that_is_not_an_exact_match_fails_before_any_prompt(harness, model):
    with pytest.raises(harness.session_mod.ACPModelError, match="not offered"):
        harness.turn("first", model=model)
    assert harness.requests("session/prompt") == []


def test_model_is_pinned_again_after_a_load(harness):
    harness.turn("first", model=OPUS_1M)
    with pytest.raises(harness.session_mod.ACPTurnError):
        harness.turn("CRASH", model=OPUS_1M)
    harness.turn("after", model=OPUS_1M)
    assert [row["params"]["value"] for row in harness.requests("session/set_config_option")] == [OPUS_1M, OPUS_1M]


def test_a_crash_mid_turn_raises_a_report_and_never_resends(harness):
    with pytest.raises(harness.session_mod.ACPTurnError) as caught:
        harness.turn("CRASH")
    assert "Transcript:" in str(caught.value)
    assert "about to crash" in str(caught.value)
    assert harness.prompts() == ["CRASH"]
    verdict = harness.plugin._classify_api_error(caught.value)
    assert verdict == {"reason": "unknown", "retryable": False, "should_fallback": False}


def test_any_hermes_retry_of_a_started_turn_finds_nothing_to_send(harness):
    with pytest.raises(harness.session_mod.ACPTurnError):
        harness.turn("CRASH")
    with pytest.raises(harness.session_mod.ACPTurnError, match="Refused to resend"):
        harness.client()._create_chat_completion(model=SONNET, messages=harness.messages, tools=TOOLS, timeout=20)
    assert harness.prompts() == ["CRASH"]


def test_the_next_turn_after_a_crash_resumes_and_sends_only_itself(harness):
    with pytest.raises(harness.session_mod.ACPTurnError):
        harness.turn("CRASH")
    completion = harness.turn("after")
    assert harness.prompts() == ["CRASH", "after"]
    assert len(harness.requests("session/load")) == 1
    # The replay burst that session/load sends is not part of any reply.
    assert completion.choices[0].message.content == "echo: after"


def test_a_failed_setup_loses_the_adapter_and_the_next_turn_resumes(harness, monkeypatch):
    original = harness.session_mod.AdapterSession._set_mode_default
    calls = []

    def fail_once(session):
        calls.append(session.proc)
        if len(calls) == 1:
            raise TimeoutError("The adapter did not answer session/set_mode.")
        original(session)

    monkeypatch.setattr(harness.session_mod.AdapterSession, "_set_mode_default", fail_once)
    with pytest.raises(TimeoutError):
        harness.turn("first", model=OPUS_1M)
    assert calls[0].poll() is not None
    assert harness.turn("second", model=OPUS_1M).choices[0].message.content.startswith("echo:")
    assert len(harness.requests("session/load")) == 1
    assert [row["params"]["value"] for row in harness.requests("session/set_config_option")] == [OPUS_1M]


def test_an_adapter_that_answers_during_the_write_still_settles_the_prompt(harness, tmp_path):
    launch = harness.session_mod.Launch(command=sys.executable, args=(), cwd=str(tmp_path), build_env=dict)
    session = harness.session_mod.AdapterSession("race", launch)
    session._write = lambda message: session._route_response({"id": message["id"], "result": {}})
    session._prompt_settled.clear()
    session._send_request("session/prompt", {}, is_prompt=True)
    assert session._prompt_settled.is_set()
    assert session._open_prompt is None


def test_failures_before_the_prompt_keep_hermess_own_classification(harness):
    assert harness.plugin._classify_api_error(RuntimeError("spawn failed")) is None
    verdict = harness.plugin._classify_api_error(harness.session_mod.ACPModelError("no such model"))
    assert verdict["retryable"] is False


def _turn_in_thread(harness, client, text):
    outcome = {}

    def target():
        try:
            outcome["result"] = harness.turn(text, client=client, timeout=60)
        except Exception as exc:
            outcome["error"] = exc
        outcome["ended"] = time.monotonic()

    thread = threading.Thread(target=target)
    thread.start()
    return thread, outcome


def _wait_for_prompt(harness, count=1):
    deadline = time.monotonic() + 15
    while len(harness.requests("session/prompt")) < count:
        assert time.monotonic() < deadline, "no prompt reached the agent"
        time.sleep(0.02)


def test_cancel_sends_session_cancel_from_another_thread_and_raises_within_two_seconds(harness):
    client = harness.client()
    thread, outcome = _turn_in_thread(harness, client, "HANG")
    _wait_for_prompt(harness)
    canceled_at = time.monotonic()
    client.cancel()
    thread.join(5)
    assert outcome["ended"] - canceled_at < 2
    assert isinstance(outcome["error"], harness.session_mod.ACPTurnError)
    assert "Transcript:" in str(outcome["error"])
    assert harness.requests("session/cancel")


def test_cancel_before_the_prompt_is_on_the_wire_sends_no_prompt(harness):
    client = harness.client()
    original = client._launch
    client._launch = lambda: (client.cancel(), original())[1]
    # The turn does not exist until the call starts, so the cancel lands while it sets up.
    with pytest.raises(RuntimeError, match="before its prompt"):
        harness.turn("first", client=client)
    assert harness.requests("session/prompt") == []


def test_a_stopped_turn_does_not_repeat_and_the_next_turn_works(harness):
    client = harness.client()
    thread, outcome = _turn_in_thread(harness, client, "HANG")
    _wait_for_prompt(harness)
    client.cancel()
    thread.join(5)
    completion = harness.turn("after")
    assert completion.choices[0].message.content == "echo: after"
    assert harness.prompts() == ["HANG", "after"]
    assert len(_adapter_pids(harness)) == 1


def test_an_agent_that_ignores_cancel_loses_its_adapter(harness):
    client = harness.client()
    thread, outcome = _turn_in_thread(harness, client, "HANG-HARD")
    _wait_for_prompt(harness)
    client.cancel()
    thread.join(5)
    assert isinstance(outcome["error"], harness.session_mod.ACPTurnError)
    completion = harness.turn("after")
    assert completion.choices[0].message.content == "echo: after"
    assert len(_adapter_pids(harness)) == 2


def test_cancel_ends_what_the_turn_started_and_spares_claude_code(harness):
    client = harness.client()
    thread, _ = _turn_in_thread(harness, client, "CHILD 120")
    _wait_for_prompt(harness)
    deadline = time.monotonic() + 15
    while not harness.requests("_child"):
        assert time.monotonic() < deadline, "the fake agent started no child"
        time.sleep(0.02)
    direct = psutil.Process(harness.requests("_child")[0]["child"])
    grandchildren = direct.children()
    assert grandchildren
    client.cancel()
    thread.join(5)
    _, alive = psutil.wait_procs(grandchildren, timeout=5)
    assert alive == []
    assert direct.is_running()


def test_a_report_quotes_the_running_tool_call_when_there_is_no_text(harness, tmp_path):
    client = harness.client()
    thread, outcome = _turn_in_thread(harness, client, "HANG")
    _wait_for_prompt(harness)
    session = harness.session_mod.REGISTRY.get("hermes-session-1", None)
    transcript = Path(session._transcript_path())
    transcript.parent.mkdir(parents=True)
    block = {"type": "tool_use", "name": "Bash", "input": {"command": "make test"}}
    transcript.write_text(json.dumps({"type": "assistant", "message": {"content": [block]}}) + "\n")
    client.cancel()
    thread.join(5)
    assert "Last message: Bash:" in str(outcome["error"])
    assert "make test" in str(outcome["error"])


def test_a_torn_transcript_does_not_break_the_report(harness):
    harness.turn("first")
    session = harness.session_mod.REGISTRY.get("hermes-session-1", None)
    transcript = Path(session._transcript_path())
    transcript.parent.mkdir(parents=True)
    good = {"type": "assistant", "message": {"content": [{"type": "text", "text": "kept"}]}}
    odd = [[1, 2], {"type": "assistant", "message": "text"}, {"type": "assistant"}]
    rows = [json.dumps(row).encode() for row in (good, *odd)]
    transcript.write_bytes(b"\n".join(rows) + b'\n{"type": "assistant", "x": "\xe2\x82')
    assert "Last message: kept" in session._report("Claude Code was stopped.")


def test_a_deeply_nested_transcript_row_does_not_break_the_report(harness):
    harness.turn("first")
    session = harness.session_mod.REGISTRY.get("hermes-session-1", None)
    transcript = Path(session._transcript_path())
    transcript.parent.mkdir(parents=True)
    good = {"type": "assistant", "message": {"content": [{"type": "text", "text": "kept"}]}}
    nested = "[" * 200_000 + "]" * 200_000
    with pytest.raises(RecursionError):
        json.loads(nested)
    row = f'{{"type": "assistant", "message": {{"content": [{{"type": "tool_use", "name": "Bash", "input": {nested}}}]}}}}'
    transcript.write_text(f"{json.dumps(good)}\n{row}\n", encoding="utf-8")
    assert "Last message: kept" in session._report("Claude Code was stopped.")


def test_a_timeout_cancels_and_raises_without_a_retry(harness):
    with pytest.raises(harness.session_mod.ACPTurnError, match="stopped after 1 s"):
        harness.turn("HANG", timeout=1)
    assert harness.requests("session/cancel")
    assert harness.prompts() == ["HANG"]


def test_text_between_turns_is_held_for_the_next_reply_when_the_session_refuses_it(harness):
    harness.turn("BETWEEN late report")
    deadline = time.monotonic() + 5
    session = harness.session_mod.REGISTRY.get("hermes-session-1", None)
    while not session._between and time.monotonic() < deadline:
        time.sleep(0.02)
    completion = harness.turn("next")
    # This scratch home has no such session, so the note waits for the next reply.
    assert "late report" in completion.choices[0].message.content
    assert harness.prompts() == ["BETWEEN late report", "next"]


def test_hermes_tool_call_round_trip_sends_only_the_result(harness):
    completion = harness.turn("TOOLCALL")
    assert completion.choices[0].finish_reason == "tool_calls"
    harness.messages.append({"role": "tool", "tool_call_id": "t1", "content": "saved"})
    harness.client()._create_chat_completion(model=SONNET, messages=harness.messages, tools=TOOLS, timeout=20)
    assert harness.prompts() == ["TOOLCALL", "Tool result for t1:\nsaved"]


def test_a_rewritten_history_sends_a_bounded_memo(harness):
    harness.turn("first")
    harness.messages[1] = {"role": "user", "content": "a compressed summary"}
    harness.turn("second")
    assert "a compressed summary" in harness.prompts()[-1]
    assert "second" in harness.prompts()[-1]


def test_prompt_too_long_starts_a_new_session_next_turn_with_a_memo(harness):
    harness.turn("first")
    with pytest.raises(harness.session_mod.ACPTurnError, match="Prompt is too long"):
        harness.turn("TOOLONG")
    harness.messages.append({"role": "assistant", "content": "failed"})
    harness.turn("after")
    assert len(harness.requests("session/new")) == 2
    assert "first" in harness.prompts()[-1]


def test_a_new_process_resumes_from_the_persisted_session(harness):
    harness.turn("first")
    harness.session_mod.REGISTRY.close_all()
    harness.turn("second")
    assert len(harness.requests("session/load")) == 1
    assert harness.prompts() == ["first", "second"]


def _fail_state_writes(monkeypatch, *failing_calls):
    """Make the given state writes fail, counting from 1, and keep the others real."""
    real = os.replace
    calls = []

    def replace(source, target):
        calls.append(target)
        if len(calls) in failing_calls:
            raise OSError("No space left on device")
        real(source, target)

    monkeypatch.setattr(os, "replace", replace)
    return calls


def test_a_state_write_that_fails_after_the_prompt_is_not_retryable(harness, monkeypatch):
    harness.turn("first")
    calls = _fail_state_writes(monkeypatch, 2)
    with pytest.raises(harness.session_mod.ACPTurnError) as caught:
        harness.turn("second")
    assert harness.plugin._classify_api_error(caught.value)["should_fallback"] is False
    assert json.loads(calls[0].read_text())["awaiting_reply"] is True


def test_a_state_write_that_fails_before_the_prompt_sends_nothing(harness, monkeypatch):
    harness.turn("first")
    _fail_state_writes(monkeypatch, 1)
    with pytest.raises(OSError):
        harness.turn("second")
    assert harness.prompts() == ["first"]
    harness.turn("third")
    assert "second" in harness.prompts()[-1]


OTHER_TOOLS = [{"type": "function", "function": {"name": "recall", "description": "d", "parameters": {}}}]


def test_a_state_write_that_fails_before_the_prompt_keeps_the_tool_notice_for_the_retry(harness, monkeypatch):
    harness.turn("first")
    _fail_state_writes(monkeypatch, 1)
    with pytest.raises(OSError):
        harness.turn("second", tools=OTHER_TOOLS)
    harness.turn("third", tools=OTHER_TOOLS)
    assert "recall" in harness.prompts()[-1]


def test_a_loaded_session_still_gets_the_notice_for_a_changed_tool_list(harness):
    harness.turn("first")
    harness.session_mod.REGISTRY.close_all()
    harness.turn("second", tools=OTHER_TOOLS)
    assert len(harness.requests("session/load")) == 1
    assert "recall" in harness.prompts()[-1]


@pytest.mark.parametrize("block", ["<tool_call>{not json</tool_call>", '<tool_call>{"function": {}}</tool_call>'])
def test_a_tool_call_block_that_does_not_parse_stays_in_the_reply(harness, block):
    calls, text = sys.modules[harness.session_mod.__package__ + ".acp_format"].extract_tool_calls(f"before {block} after")
    assert calls == []
    assert block in text


@pytest.mark.parametrize(("raw", "expected"), [("", 8), ("16", 16), ("30m", 8)])
def test_a_bad_numeric_override_keeps_the_default(harness, monkeypatch, raw, expected):
    monkeypatch.setenv("OCE_TEST_NUMBER", raw)
    assert harness.session_mod._env_number("OCE_TEST_NUMBER", 8, int) == expected


def test_a_deleted_cwd_falls_back_to_hermes_home(harness, tmp_path):
    client = harness.client()
    client._acp_cwd = str(tmp_path / "gone")
    client._create_chat_completion(model=SONNET, messages=[{"role": "user", "content": "x"}], tools=TOOLS, timeout=20)
    assert harness.requests("session/new")[0]["params"]["cwd"] == str((tmp_path / "home").resolve())


def test_usage_reaches_hermes(harness):
    usage = harness.turn("first").usage
    assert usage.prompt_tokens == 15
    assert usage.completion_tokens == 2


def _permission(harness, spec):
    completion = harness.turn("PERMISSION " + json.dumps(spec))
    return json.loads(completion.choices[0].message.content)["outcome"]


def test_gate_allows_a_safe_command_once_never_always(harness):
    assert _permission(harness, {"name": "Bash", "command": "echo hi"}) == {"outcome": "selected", "optionId": "allow-once"}


def test_gate_refuses_a_floor_blocked_command(harness):
    assert _permission(harness, {"name": "Bash", "command": "rm -rf /"})["optionId"] == "reject"


def test_gate_refuses_a_write_to_hermess_env_file(harness, tmp_path):
    outcome = _permission(harness, {"name": "Write", "file_path": str(tmp_path / "home" / ".env"), "content": "X"})
    assert outcome["optionId"] == "reject"


def test_gate_refuses_a_read_of_a_project_env_file(harness, tmp_path):
    assert _permission(harness, {"name": "Read", "file_path": str(tmp_path / ".env")})["optionId"] == "reject"


def test_gate_refuses_a_write_to_ssh_config(harness):
    outcome = _permission(harness, {"name": "Write", "file_path": str(Path.home() / ".ssh" / "config"), "content": "X"})
    assert outcome["optionId"] == "reject"


def test_gate_allows_an_ordinary_write_and_read(harness, tmp_path):
    assert _permission(harness, {"name": "Write", "file_path": str(tmp_path / "ok.txt")})["optionId"] == "allow-once"
    assert _permission(harness, {"name": "Read", "file_path": str(tmp_path / "ok.txt")})["optionId"] == "allow-once"


def test_gate_refuses_a_tool_with_no_guard_such_as_mcp(harness):
    assert _permission(harness, {"name": "mcp__server__do_thing", "arg": 1})["optionId"] == "reject"


def test_gate_refuses_a_request_that_carries_no_command(harness):
    assert _permission(harness, {"name": "Bash"})["optionId"] == "reject"
