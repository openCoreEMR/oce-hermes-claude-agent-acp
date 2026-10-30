#!/usr/bin/env python3
"""Conformance probe: the plugin against the real adapter and real Claude Code, with a scripted model.

A stub Messages API stands in for Anthropic through ``ANTHROPIC_BASE_URL``, so
the run costs nothing and needs no credential. It plays each prompt's script,
including tool calls, and Claude Code really runs those tools. Rerun this for
every bump of the adapter, Claude Code, Hermes, or the plugin.

    HERMES_AGENT_PATH=/path/to/hermes-agent python tests/conformance_probe.py \\
        --adapter claude-agent-acp --record run-record.json

It exits 0 when no condition fired. ``CLAUDE_CODE_EXECUTABLE`` selects the
Claude Code binary. Conditions 6 (kanban), 7, 10, and 11's chat display need a
running gateway: ``tests/gateway_probe.py`` covers them on the agent.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
# Under an API key the adapter offers aliases. Under a plan login it offers
# full ids, including the [1m] ones, so tests/gateway_probe.py checks those.
MODEL = "sonnet"
TOOLS = [{"type": "function", "function": {"name": "memory", "description": "Save a fact.", "parameters": {}}}]
SCRIPT_MARK = "SCRIPT "
DEFAULT_REPLY = "STUB-REPLY"
CANCEL_LIMIT_SECONDS = 2.0
CHILD_LIMIT_SECONDS = 10.0
SETTLE_LIMIT_SECONDS = 120.0
BETWEEN_TURN_LIMIT_SECONDS = 60.0


def _content_text(content):
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content or () if isinstance(block, dict))


def _next_step(messages):
    """The script's step for this request: one step per assistant reply since the newest prompt, if it has a script."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        text = _content_text(message.get("content")) if message.get("role") == "user" else ""
        if not text.strip():
            continue
        if SCRIPT_MARK not in text:
            return {"text": DEFAULT_REPLY}
        steps = json.loads(text[text.index(SCRIPT_MARK) + len(SCRIPT_MARK):].splitlines()[0])
        done = sum(1 for later in messages[index + 1:] if later.get("role") == "assistant")
        return steps[done] if done < len(steps) else {"text": "DONE"}
    return {"text": DEFAULT_REPLY}


def _events(step, model):
    """Server-sent events for one step: a text reply, or a tool call."""
    usage = {"input_tokens": 1, "output_tokens": 1}
    message = {"id": "msg_stub", "type": "message", "role": "assistant", "model": model, "content": [],
               "stop_reason": None, "stop_sequence": None, "usage": usage}
    yield "message_start", {"type": "message_start", "message": message}
    if "tool" in step:
        block = {"type": "tool_use", "id": f"toolu_{time.monotonic_ns()}", "name": step["tool"], "input": {}}
        delta = {"type": "input_json_delta", "partial_json": json.dumps(step["input"])}
        stop_reason = "tool_use"
    else:
        block = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": step["text"]}
        stop_reason = "end_turn"
    yield "content_block_start", {"type": "content_block_start", "index": 0, "content_block": block}
    yield "content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}
    yield "content_block_stop", {"type": "content_block_stop", "index": 0}
    yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                            "usage": {"output_tokens": 1}}
    yield "message_stop", {"type": "message_stop"}


class _Stub(BaseHTTPRequestHandler):
    models: list[str] = []

    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        model = str(body.get("model"))
        _Stub.models.append(model)
        step = _next_step(body.get("messages") or [])
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for name, data in _events(step, model):
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())


def script(*steps):
    return SCRIPT_MARK + json.dumps(steps)


def bash(command, **extra):
    return {"tool": "Bash", "input": {"command": command, "description": "probe", **extra}}


def _load_plugin():
    hermes_path = os.environ.get("HERMES_AGENT_PATH", "").strip()
    if hermes_path:
        sys.path.insert(0, hermes_path)
    plugin_dir = REPO_ROOT / "providers" / "claude-acp"
    name = "_oce_claude_acp_conformance"
    spec = importlib.util.spec_from_file_location(name, plugin_dir / "__init__.py", submodule_search_locations=[str(plugin_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return importlib.import_module(name + ".acp_client"), importlib.import_module(name + ".acp_session")


class Probe:
    def __init__(self, adapter, base_url, workdir, home):
        self.client_mod, self.session_mod = _load_plugin()
        self.workdir = workdir
        self.home = home
        self.launch_args = [f"ANTHROPIC_BASE_URL={base_url}", "ANTHROPIC_API_KEY=sk-ant-stub", adapter]
        self.results = []
        self.counter = 0

    def client(self):
        return self.client_mod.ACPClient(
            agent_name="claude", acp_cwd=str(self.workdir), acp_command="env", acp_args=self.launch_args)

    def new_session(self):
        self.counter += 1
        os.environ["HERMES_SESSION_ID"] = f"conformance-{self.counter}"
        return [{"role": "system", "content": "Conformance probe."}]

    def call(self, messages, text, *, client=None, timeout=90.0, model=MODEL):
        messages.append({"role": "user", "content": text})
        completion = (client or self.client())._create_chat_completion(
            model=model, messages=messages, tools=TOOLS, timeout=timeout)
        reply = completion.choices[0].message
        stored = {"role": "assistant", "content": reply.content}
        if reply.tool_calls:
            stored["tool_calls"] = [{"id": call.id} for call in reply.tool_calls]
        messages.append(stored)
        return completion

    def session(self):
        return self.session_mod.REGISTRY.get(os.environ["HERMES_SESSION_ID"], None)

    def record(self, condition, fired, **detail):
        self.results.append({"condition": condition, "fired": fired, **detail})
        print(json.dumps(self.results[-1], default=str), flush=True)

    def call_in_thread(self, messages, text, client):
        outcome = {}

        def target():
            try:
                outcome["completion"] = self.call(messages, text, client=client)
            except Exception as exc:
                outcome["error"] = exc
            outcome["ended"] = time.monotonic()

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread, outcome

    # Condition 1: a stop ends the Bash child and leaves no later tool call.
    def condition_1(self):
        messages = self.new_session()
        tag = f"-c {100 + self.counter} -i 1 127.0.0.1"
        after = self.workdir / f"after-stop-{self.counter}"
        client = self.client()
        thread, outcome = self.call_in_thread(messages, script(bash(f"ping {tag}"), bash(f"touch {after}")), client)
        started = _wait(lambda: _running(tag), 60)
        stop_at = time.monotonic()
        client.cancel()
        thread.join(5)
        raised_in = outcome.get("ended", float("inf")) - stop_at
        gone = _wait(lambda: not _running(tag), CHILD_LIMIT_SECONDS)
        time.sleep(max(0.0, 11 - (time.monotonic() - stop_at)))
        report = str(outcome.get("error"))
        self.record(1, not (started and gone and raised_in < CANCEL_LIMIT_SECONDS and not after.exists()
                            and "Transcript:" in report),
                    started=started, child_gone=gone, raised_in=round(raised_in, 2), later_tool_ran=after.exists())
        self.call(messages, "Reply with a word.")

    # Conditions 2 and 3: a prompt during a background task settles, and the task's later report arrives.
    def conditions_2_and_3(self):
        messages = self.new_session()
        tag = f"-c {20 + self.counter} -i 1 127.0.0.1"
        self.call(messages, script(bash(f"ping {tag}", run_in_background=True), {"text": "STARTED"}))
        running = _running(tag)
        started = time.monotonic()
        follow_up = self.call(messages, "What is 2+2?")
        settled_in = time.monotonic() - started
        self.record(2, not (running and settled_in < SETTLE_LIMIT_SECONDS), background_running=running,
                    settled_in=round(settled_in, 2), reply=follow_up.choices[0].message.content)
        ended = _wait(lambda: not _running(tag), 90)
        session = self.session()
        arrived = _wait(lambda: session._between or session._undelivered or session._delivered_notes,
                        BETWEEN_TURN_LIMIT_SECONDS)
        self.record(3, not (ended and arrived), background_ended=ended, between_turn_text=bool(arrived))

    # Conditions 4 and 9: a killed adapter fails the turn, and nothing repeats its side effect.
    def conditions_4_and_9(self):
        messages = self.new_session()
        marker = self.workdir / f"kill-marker-{self.counter}"
        command = f"echo x >> {marker} ; ping -c 60 -i 1 127.0.0.1 >/dev/null"
        client = self.client()
        thread, outcome = self.call_in_thread(messages, script(bash(command)), client)
        wrote = _wait(marker.exists, 60)
        session = self.session()
        os.kill(session.proc.pid, signal.SIGKILL)
        thread.join(30)
        error = outcome.get("error")
        retried = False
        try:
            self.client()._create_chat_completion(model=MODEL, messages=messages, tools=TOOLS, timeout=30)
            retried = True
        except self.session_mod.ACPTurnError as exc:
            retried = "Refused to resend" not in str(exc)
        time.sleep(3)
        lines = marker.read_text().count("\n") if marker.exists() else 0
        self.record(4, not (wrote and isinstance(error, self.session_mod.ACPTurnError) and not retried),
                    wrote=wrote, error=str(error)[:200], resend_went_through=retried)
        self.record(9, lines != 1, marker_lines=lines)
        session.kill()

    # Condition 5: every decision matches the guard Hermes's own tool would meet.
    def condition_5(self):
        (self.workdir / ".env").write_text("PROBE=1\n")
        # Absent, so Claude Code's own read-before-write check cannot stand in for the gate.
        hermes_env = self.home / "hermes" / ".env"
        ssh_config = Path.home() / ".ssh" / "config"
        ok_file = self.workdir / "gate-ok.txt"
        safe_marker = self.workdir / "gate-safe"
        cases = (
            ("safe command", bash(f"touch {safe_marker}"), lambda: safe_marker.exists(), True),
            ("ordinary write", {"tool": "Write", "input": {"file_path": str(ok_file), "content": "ok"}}, ok_file.exists, True),
            ("write to Hermes .env", {"tool": "Write", "input": {"file_path": str(hermes_env), "content": "X=2"}},
             hermes_env.exists, False),
            ("write to ssh config", {"tool": "Write", "input": {"file_path": str(ssh_config), "content": "# x"}},
             ssh_config.exists, False),
        )
        for name, step, happened, expected in cases:
            messages = self.new_session()
            self.call(messages, script(step))
            occurred = bool(happened())
            self.record(5, occurred != expected, case=name, happened=occurred, expected=expected)
        # A read leaves no mark on disk, so the gate's own decision is the observable.
        gate = sys.modules[self.client_mod.__name__.rsplit(".", 1)[0] + ".acp_gate"]
        for name, tool, path, expected in (
            ("read of Hermes .env", "Read", hermes_env, False),
            ("read of a project .env", "Read", self.workdir / ".env", False),
            ("read of an ordinary file", "Read", ok_file, True),
            ("mcp tool", "mcp__s__t", ok_file, False),
        ):
            allowed, _ = gate.tool_call_allowed({"rawInput": {"file_path": str(path)}}, tool, str(self.workdir))
            self.record(5, allowed != expected, case=name, allowed=allowed, expected=expected)

    # Condition 6: a Hermes tool call in text form reaches Hermes, and its result comes back alone.
    def condition_6(self):
        messages = self.new_session()
        call = '<tool_call>{"id":"t1","type":"function","function":{"name":"memory","arguments":"{}"}}</tool_call>'
        completion = self.call(messages, script({"text": call}, {"text": "SAVED"}))
        calls = completion.choices[0].message.tool_calls
        messages.append({"role": "tool", "tool_call_id": "t1", "content": "ok"})
        follow_up = self.client()._create_chat_completion(model=MODEL, messages=messages, tools=TOOLS, timeout=60)
        self.record(6, not (calls and calls[0].function.name == "memory"), scope="text bridge only; cards need the gateway",
                    reply=follow_up.choices[0].message.content)

    # Condition 8: the exact option is served, and a loose id fails before any prompt.
    def condition_8(self):
        for model in ("sonnet", "opus"):
            messages = self.new_session()
            before = len(_Stub.models)
            completion = self.call(messages, "Reply with a word.", model=model)
            served = sorted(set(_Stub.models[before:]))
            self.record(8, completion.model != model or not any(model in name for name in served),
                        requested=model, served=served, usage_size=self.session().last_usage_size)
        for model in ("opus-5", "claude-opus-5-5", "son"):
            messages = self.new_session()
            try:
                self.call(messages, "Reply with a word.", model=model)
                refused = False
            except self.session_mod.ACPModelError:
                refused = True
            self.record(8, not refused, requested=model, refused=refused)

    # Condition 11's report: a turn that outlives its timeout raises the transcript path and last message.
    def condition_11(self):
        messages = self.new_session()
        tag = f"-c {300 + self.counter} -i 1 127.0.0.1"
        try:
            self.call(messages, script(bash(f"ping {tag}")), timeout=6.0)
            report = ""
        except self.session_mod.ACPTurnError as exc:
            report = str(exc)
        gone = _wait(lambda: not _running(tag), CHILD_LIMIT_SECONDS)
        session = self.session()
        self.record(11, not ("Transcript:" in report and "Last message:" in report and gone), scope="the report; the chat display needs the gateway",
                    report=report, report_chars=len(report), transcript_exists=Path(session._transcript_path()).exists())


def _running(fragment):
    for process in psutil.process_iter(["cmdline"]):
        command = " ".join(process.info["cmdline"] or ())
        if command.startswith("ping") and fragment in command:
            return True
    return False


def _wait(check, bound):
    end = time.monotonic() + bound
    while time.monotonic() < end:
        result = check()
        if result:
            return result
        time.sleep(0.25)
    return check()


def _version(command):
    try:
        return subprocess.run([*command, "--version"], capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def _run_record(args):
    from importlib.metadata import PackageNotFoundError, version
    try:
        hermes = version("hermes-agent")
    except PackageNotFoundError:
        hermes = "unknown"
    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    adapter_path = shutil.which(args.adapter)
    package = Path(adapter_path).resolve().parent.parent / "package.json" if adapter_path else None
    return {
        "hermes": hermes,
        "plugin_commit": commit or os.environ.get("OCE_PLUGIN_COMMIT", "unknown"),
        "adapter": json.loads(package.read_text())["version"] if package and package.exists() else "unknown",
        "claude_code": _version([os.environ.get("CLAUDE_CODE_EXECUTABLE", "claude")]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", default="claude-agent-acp")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--tries", type=int, default=1, help="Repeat the conditions that count tries.")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    root = Path(tempfile.mkdtemp(prefix="oce-acp-conformance-"))
    home = root / "home"
    workdir = root / "work"
    for path in (home / "hermes", home / ".ssh", workdir):
        path.mkdir(parents=True)
    os.environ.update({
        "HOME": str(home),
        "HERMES_HOME": str(home / "hermes"),
        "OCE_CLAUDE_ACP_USE_LOGIN": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    })
    probe = Probe(args.adapter, f"http://127.0.0.1:{server.server_address[1]}", workdir.resolve(), home)
    try:
        for _ in range(args.tries):
            probe.condition_1()
            probe.conditions_2_and_3()
            probe.conditions_4_and_9()
        probe.condition_5()
        probe.condition_6()
        probe.condition_8()
        probe.condition_11()
    finally:
        probe.session_mod.REGISTRY.close_all()
        server.shutdown()
    fired = [result for result in probe.results if result["fired"]]
    record = {"run": _run_record(args), "tries": args.tries, "fired": len(fired), "results": probe.results}
    if args.record:
        args.record.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    shutil.rmtree(root, ignore_errors=True)
    return 1 if fired else 0


if __name__ == "__main__":
    sys.exit(main())
