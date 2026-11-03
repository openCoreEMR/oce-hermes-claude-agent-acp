#!/usr/bin/env python3
"""A fake ACP agent that answers correctly and can be told to misbehave.

Extended from the fake server in PR #2's ``test_acp_wiring.py``. It speaks
line-delimited JSON-RPC on stdio and records each request in the file named by
``FAKE_ACP_LOG``. Each prompt's text picks its behavior:

``HANG``         never answer the prompt; a cancel settles it.
``HANG-HARD``    never answer the prompt, even after a cancel.
``CRASH``        exit in the middle of the turn.
``PERMISSION <json>``  ask for this tool call's permission, then report the outcome.
``CHILD <seconds>``  start a process tree, wait for a cancel, and leave the tree running.
``BETWEEN <text>``  finish the turn, then write text with no prompt in flight.
``WAKE <json>``  wait for a cancel, then ask for this tool call's permission with
                 no prompt in flight, as a background task's notification does.
``TOOLCALL``     reply with a Hermes tool call in text form.
``TOOLONG``      fail the prompt with Claude Code's context error.
``EMPTY``        end the turn with no text.
anything else    reply ``echo: <text>``.

``session/load`` works as the real adapter does: an unknown id is an error, a
known one replays history as updates and resets the model to the default.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

LOG = Path(os.environ["FAKE_ACP_LOG"])
SESSIONS = Path(os.environ["FAKE_ACP_LOG"] + ".sessions")
MODELS = ("default", "claude-sonnet-5-5", "claude-sonnet-5-5[1m]", "claude-opus-5-5[1m]")

current_model = "default"
current_mode = "acceptEdits"
write_lock = threading.Lock()
next_request = iter(range(1000, 100000))
waiting: dict[int, "threading.Event"] = {}
answers: dict[int, dict] = {}
canceled = threading.Event()


def log(entry):
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def send(message):
    with write_lock:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
        sys.stdout.flush()


def options():
    return [{
        "id": "model", "category": "model", "currentValue": current_model,
        "options": [{"value": value, "name": value} for value in MODELS],
    }]


def update(session_id, **fields):
    send({"method": "session/update", "params": {"sessionId": session_id, "update": fields}})


def say(session_id, text):
    update(session_id, sessionUpdate="agent_message_chunk", content={"type": "text", "text": text})


def ask_permission(session_id, spec):
    call_id = f"call-{next(next_request)}"
    name = spec.pop("name")
    update(session_id, sessionUpdate="tool_call", toolCallId=call_id, title=name, _meta={"claudeCode": {"toolName": name}})
    request_id = next(next_request)
    done = threading.Event()
    waiting[request_id] = done
    send({"id": request_id, "method": "session/request_permission", "params": {
        "sessionId": session_id,
        "toolCall": {"toolCallId": call_id, "title": name, "rawInput": spec},
        "options": [
            {"optionId": "allow-once", "kind": "allow_once", "name": "Allow"},
            {"optionId": "allow-always", "kind": "allow_always", "name": "Always"},
            {"optionId": "reject", "kind": "reject_once", "name": "Reject"},
        ],
    }})
    done.wait(30)
    log({"method": "_permission_answer", "answer": answers.get(request_id, {})})
    return answers.get(request_id, {})


def settle(request_id, session_id, text):
    say(session_id, text)
    update(session_id, sessionUpdate="usage_update", used=10, size=1_000_000 if current_model.endswith("[1m]") else 200_000)
    send({"id": request_id, "result": {"stopReason": "end_turn", "usage": {
        "inputTokens": 3, "outputTokens": 2, "cachedReadTokens": 5, "cachedWriteTokens": 7}}})


def run_prompt(request_id, session_id, text):
    canceled.clear()
    command, _, rest = text.partition(" ")
    if command == "HANG":
        canceled.wait()
        send({"id": request_id, "result": {"stopReason": "cancelled"}})
    elif command == "HANG-HARD":
        threading.Event().wait()
    elif command == "CRASH":
        say(session_id, "about to crash")
        os._exit(3)
    elif command == "PERMISSION":
        answer = ask_permission(session_id, json.loads(rest))
        settle(request_id, session_id, json.dumps(answer))
    elif command == "CHILD":
        # A shell and its child, as Claude Code's Bash tool leaves them.
        child = subprocess.Popen(["sh", "-c", f"sleep {rest} & wait"])
        time.sleep(0.2)
        log({"method": "_child", "child": child.pid})
        canceled.wait()
        send({"id": request_id, "result": {"stopReason": "cancelled"}})
    elif command == "WAKE":
        canceled.wait()
        send({"id": request_id, "result": {"stopReason": "cancelled"}})
        time.sleep(0.2)
        log({"method": "_wake", "answer": ask_permission(session_id, json.loads(rest))})
    elif command == "BETWEEN":
        settle(request_id, session_id, "turn done")
        time.sleep(0.2)
        say(session_id, rest)
    elif command == "TOOLCALL":
        settle(request_id, session_id, '<tool_call>{"id":"t1","type":"function","function":{"name":"memory","arguments":"{}"}}</tool_call>')
    elif command == "TOOLONG":
        send({"id": request_id, "error": {"code": -32000, "message": "Prompt is too long"}})
    elif command == "EMPTY":
        send({"id": request_id, "result": {"stopReason": "end_turn"}})
    else:
        settle(request_id, session_id, f"echo: {text}")


def known_sessions():
    return SESSIONS.read_text().split() if SESSIONS.exists() else []


def main():
    global current_model, current_mode
    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")
        params = message.get("params") or {}
        request_id = message.get("id")
        if method is None:
            answers[request_id] = message.get("result") or {}
            if request_id in waiting:
                waiting[request_id].set()
            continue
        log({"method": method, "params": params, "pid": os.getpid()})
        if method == "session/cancel":
            canceled.set()
            continue
        if method == "initialize":
            result = {"protocolVersion": 1}
        elif method == "session/new":
            session_id = f"fake-{os.getpid()}-{len(known_sessions())}"
            with SESSIONS.open("a", encoding="utf-8") as handle:
                handle.write(session_id + "\n")
            current_model = "default"
            result = {"sessionId": session_id, "configOptions": options()}
        elif method == "session/load":
            if params.get("sessionId") not in known_sessions():
                send({"id": request_id, "error": {"code": -32002, "message": "Session not found"}})
                continue
            say(params["sessionId"], "replayed history that must not reach a reply")
            current_model = "default"
            result = {"sessionId": params["sessionId"], "configOptions": options()}
        elif method == "session/set_config_option":
            current_model = params["value"]
            result = {"configOptions": options()}
        elif method == "session/set_mode":
            current_mode = params["modeId"]
            result = {}
        elif method == "session/prompt":
            text = "".join(block.get("text", "") for block in params["prompt"])
            threading.Thread(target=run_prompt, args=(request_id, params["sessionId"], text), daemon=True).start()
            continue
        else:
            send({"id": request_id, "error": {"code": -32601, "message": f"unknown method {method}"}})
            continue
        send({"id": request_id, "result": result})


main()
