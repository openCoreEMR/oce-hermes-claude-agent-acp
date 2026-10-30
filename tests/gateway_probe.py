#!/usr/bin/env python3
"""Probe Conditions 1 to 9 and 11 through a running Hermes gateway on the `claude` profile.

Unlike ``conformance_probe.py`` this spends real model turns on the plan, so
it is for the agent, not a build. Run it in the container the turns run in, with
the profile's ``HERMES_HOME``, from a scratch directory:

    HERMES_HOME=/opt/data/profiles/claude python gateway_probe.py --work /tmp/probe stop kill

Each line it prints is one JSON observation for the run record. It prints what
it saw and leaves the judgment to whoever reads the record.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil
from hermes_cli.gateway_client import connect_gateway
from hermes_constants import get_hermes_home

KANBAN_CARDS = (
    ("file", "Create a file named card-result.txt in your workspace containing the word done. Then finish the task."),
    ("summary", "Finish the task with a one sentence summary of what a kanban board is as the result."),
    ("read", "List the files in your workspace with your Bash tool, then finish the task reporting how many there are."),
    ("memory", "Save to your memory, with your memory tool, the fact that card probe value is 91. Then finish the task."),
    ("block", 'You need a human decision about which color to use. Block this task with the reason "which color?" instead of finishing it.'),
)
CARD_BOUND_SECONDS = 300


def emit(**fields):
    print(json.dumps(fields, default=str), flush=True)


def adapters():
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        command = " ".join(process.info["cmdline"] or ())
        if "claude-agent-acp" in command and process.info["pid"] != os.getpid():
            found.append(process.info["pid"])
    return sorted(found)


def running(fragment):
    for process in psutil.process_iter(["cmdline"]):
        command = " ".join(process.info["cmdline"] or ())
        if command.startswith("ping") and fragment in command:
            return True
    return False


def lines(path):
    return path.read_text().count("\n") if path.exists() else 0


async def wait_for(check, bound):
    end = time.monotonic() + bound
    while time.monotonic() < end:
        if check():
            return True
        await asyncio.sleep(0.5)
    return False


class Session:
    """One gateway session: prompts, stops, and model switches as live clients send them."""

    def __init__(self, client, sid):
        self.client, self.sid = client, sid
        self.admission = None
        self.approvals = []

    async def submit(self, text):
        receipt = await self.client.rpc("prompt.submit", session_id=self.sid, text=text,
                                        submission_id=str(uuid.uuid4()), _timeout=120)
        self.admission = receipt["admission_id"]

    async def complete(self, bound, answer=None):
        started = time.monotonic()
        receipt = {}
        while time.monotonic() - started < bound:
            await self._drain(answer)
            receipt = await self.client.rpc("prompt.receipt", session_id=self.sid, admission_id=self.admission)
            if receipt.get("status") in {"terminal", "unknown"}:
                return receipt.get("status"), receipt.get("outcome") or receipt.get("result"), round(time.monotonic() - started, 1)
            await asyncio.sleep(0.5)
        return "BOUND-EXCEEDED", receipt.get("status"), round(time.monotonic() - started, 1)

    async def _drain(self, answer):
        while not self.client.events.empty():
            frame = self.client.events.get_nowait()
            params = frame.get("params") or {}
            if params.get("type") != "approval.request" or params.get("session_id") != self.sid:
                continue
            payload = params.get("payload") or {}
            self.approvals.append(payload.get("command"))
            if answer is not None:
                await self.client.rpc("approval.respond", session_id=self.sid, choice=answer,
                                      execution_generation=payload["execution_generation"], prompt_id=payload["prompt_id"])

    async def turn(self, text, bound=300, answer=None):
        await self.submit(text)
        return await self.complete(bound, answer)

    async def switch_model(self, model, provider):
        snap = await self.client.rpc("session.resume", session_id=self.sid, _timeout=60)
        return await self.client.rpc("session.mutate", session_id=self.sid, request_id=uuid.uuid4().hex,
                                     expected_revision=snap["revision"], expected_generation=snap["execution_generation"],
                                     operation="model", payload={"model": model, "provider": provider}, _timeout=60)

    async def interrupt(self):
        snap = await self.client.rpc("session.resume", session_id=self.sid, _timeout=60)
        return await self.client.rpc("session.interrupt", session_id=self.sid, execution_generation=snap["execution_generation"])

    def rows(self):
        with sqlite3.connect(f"file:{get_hermes_home() / 'state.db'}?mode=ro", uri=True) as conn:
            return conn.execute(
                "select role, tool_name, substr(content,1,160), display_kind, display_metadata from messages "
                "where session_id=? order by id", (self.sid,)).fetchall()

    def tail(self, count=2):
        return [row[:3] for row in self.rows()][-count:]

    def acp_id(self):
        digest = hashlib.sha256(self.sid.encode()).hexdigest()[:32]
        path = get_hermes_home() / "claude-acp" / "sessions" / f"{digest}.json"
        return json.loads(path.read_text()).get("acp_id") if path.exists() else None


async def open_session(client):
    created = await client.rpc("session.create", _timeout=120)
    return Session(client, created["session_id"])


def transcript_models(acp_id):
    models = set()
    home = Path(os.environ.get("HERMES_CLAUDE_TRANSCRIPTS") or get_hermes_home() / "home" / ".claude" / "projects")
    for root in (home, Path.home() / ".claude" / "projects"):
        for path in root.glob(f"*/{acp_id}.jsonl"):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get("type") == "assistant":
                    models.add(row["message"].get("model"))
    return sorted(model for model in models if model)


def usage_sizes(acp_id, args):
    log = args.log or get_hermes_home() / "logs" / "agent.log"
    if not log.exists():
        return []
    return sorted({line.rsplit("size=", 1)[1].strip() for line in log.read_text(errors="replace").splitlines()
                   if f"acp={acp_id} " in line and "usage_update" in line and "size=" in line})


async def stop(client, args):
    """Condition 1: the Bash child is gone within 10 s of a stop, and no later tool runs."""
    session = await open_session(client)
    tag = f"-c 1{time.time_ns() % 100000} -i 1 127.0.0.1"
    after = args.work / f"after-stop-{time.time_ns()}"
    await session.submit(f"Use your Bash tool to run exactly: ping {tag} ; after it, use Bash again to run: touch {after} ; then reply DONE.")
    emit(step="c1-started", started=await wait_for(lambda: running(tag), 60), adapters=adapters())
    stop_at = time.monotonic()
    await session.interrupt()
    emit(step="c1-settled", result=await session.complete(30))
    emit(step="c1-child-gone", gone=await wait_for(lambda: not running(tag), 10), after=round(time.monotonic() - stop_at, 1))
    await asyncio.sleep(max(0, 12 - (time.monotonic() - stop_at)))
    emit(step="c1-later-tool", ran=after.exists())
    emit(step="c1-next-turn", result=await session.turn("Reply with exactly the word NEXT."), rows=session.tail())


async def background(client, args):
    """Conditions 2 and 3: a prompt during a background task settles, and the task's later report arrives."""
    session = await open_session(client)
    tag = f"-c {20 + time.time_ns() % 9} -i 1 127.0.0.1"
    emit(step="c2-start", result=await session.turn(
        f"Run this with your Bash tool, in the background (run_in_background true): ping {tag} . End your turn right "
        "after starting it, with the word STARTED. When it finishes, report the words BACKGROUND-REPORT and nothing else."))
    started = time.monotonic()
    emit(step="c2-followup", result=await session.turn("While that runs: what is 2+2? Reply with only the number.", bound=180),
         background_running=running(tag))

    def recorded():
        return [row for row in session.rows() if "BACKGROUND-REPORT" in (row[2] or "") and row[3]]

    found = await wait_for(recorded, 120)
    emit(step="c3", recorded=found, seconds=round(time.monotonic() - started, 1), rows=session.tail(3))


async def kill(client, args):
    """Conditions 4 and 9: kill the adapter during a tool call. The turn must not run again."""
    marker = args.work / f"kill-marker-{time.time_ns()}"
    session = await open_session(client)
    await session.submit(f"Use your Bash tool to run exactly this one command once: echo x >> {marker} ; ping -c 40 -i 1 127.0.0.1 >/dev/null ; then reply FINISHED.")
    await wait_for(lambda: lines(marker) > 0, 90)
    emit(step="c4-before-kill", marker_lines=lines(marker))
    for pid in adapters():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    emit(step="c4-settled", result=await session.complete(120), marker_lines=lines(marker))
    await asyncio.sleep(10)
    error = [json.loads(row[4]).get("error") for row in session.rows() if row[3] == "failed_turn"]
    emit(step="c4-later", marker_lines=lines(marker), chat_error=error[-1:] if error else [])
    emit(step="c4-next-turn", result=await session.turn("Reply with exactly the word RESUMED."), marker_lines=lines(marker))


async def models(client, args):
    """Condition 8: each listed id serves that base model, [1m] reports 1M, and a loose id fails."""
    for model in args.models:
        session = await open_session(client)
        await session.switch_model(model, "claude-acp")
        result = await session.turn("Reply with exactly the word MODEL.")
        acp_id = session.acp_id()
        emit(step="c8", requested=model, result=result, served=transcript_models(acp_id), sizes=usage_sizes(acp_id, args))
    session = await open_session(client)
    await session.switch_model("opus-5", "claude-acp")
    emit(step="c8-loose", result=await session.turn("Reply with exactly the word MODEL.", bound=60), rows=session.tail(1))


async def switchback(client, args):
    """Condition 7: a switch to `anthropic` takes, also right after a stop."""
    session = await open_session(client)
    emit(step="c7-acp", result=await session.turn("Reply with exactly the word ALPHA."))
    await session.switch_model(args.anthropic_model, "anthropic")
    emit(step="c7-anthropic", result=await session.turn("Reply with exactly the word BETA."), rows=session.tail(1))
    await session.switch_model(args.acp_model, "claude-acp")
    tag = f"-c 9{time.time_ns() % 1000} -i 1 127.0.0.1"
    await session.submit(f"Use your Bash tool to run: ping {tag} ; then reply DONE.")
    await wait_for(lambda: running(tag), 60)
    await session.interrupt()
    emit(step="c7-stopped", result=await session.complete(30))
    started = time.monotonic()
    await session.switch_model(args.anthropic_model, "anthropic")
    emit(step="c7-after-stop", result=await session.turn("Reply with exactly the word GAMMA."),
         seconds=round(time.monotonic() - started, 1), rows=session.tail(1))


async def gate(client, args):
    """Condition 5, interactive: the fixed probe list. Every flagged command is denied at its approval card."""
    home = get_hermes_home()
    (args.work / ".env").write_text("PROBE=1\n")
    probes = (
        ("safe-bash", f"Use your Bash tool to run: touch {args.work}/gate-safe . Reply whether it ran."),
        ("flagged-bash", f"Use your Bash tool, one attempt only, to run: rm -rf {args.work}/gate-nothing/ . Reply whether it ran."),
        ("floor-bash", "Use your Bash tool, one attempt only, to run: rm -rf / . Reply whether it ran."),
        ("write-hermes-env", f"Use your Write tool, one attempt only, to write X=1 to {home}/.env . Reply whether it succeeded. Do not try another way."),
        ("write-state-db", f"Use your Write tool, one attempt only, to write X to {home}/sessions/probe.txt . Reply whether it succeeded. Do not try another way."),
        ("read-hermes-env", f"Use your Read tool, one attempt only, to read {home}/.env . Reply only whether it succeeded."),
        ("read-project-env", f"Use your Read tool, one attempt only, to read {args.work}/.env . Reply only whether it succeeded."),
        ("write-ok", f"Use your Write tool to write OK to {args.work}/gate-write.txt . Reply whether it succeeded."),
        ("read-ok", f"Use your Read tool to read {args.work}/gate-write.txt . Reply with its contents."),
    )
    for name, text in probes:
        session = await open_session(client)
        result = await session.turn(text, bound=120, answer="deny")
        emit(step="c5", probe=name, result=result, approvals_asked=session.approvals, reply=session.tail(1))
    emit(step="c5-files", gate_safe=(args.work / "gate-safe").exists(), gate_write=(args.work / "gate-write.txt").exists())


def _hermes_kanban(*command):
    return subprocess.run(("hermes", "kanban", *command), capture_output=True, text=True, check=False).stdout


async def cards(client, args):
    """Condition 6: varied cards end in a Hermes kanban tool call, once each. Set --kanban-db to the board's database."""
    for shape, body in KANBAN_CARDS:
        created = json.loads(_hermes_kanban("create", f"probe {shape}", "--assignee", "claude", "--body", body, "--max-retries", "2", "--json"))
        task = created["id"]
        since = time.time()
        _hermes_kanban("dispatch", "--json")
        end = time.monotonic() + CARD_BOUND_SECONDS
        status = ""
        runs = 0
        while time.monotonic() < end:
            with sqlite3.connect(f"file:{args.kanban_db}?mode=ro", uri=True) as conn:
                status = conn.execute("select status from tasks where id=?", (task,)).fetchone()[0]
                runs = conn.execute("select count(*) from task_runs where task_id=?", (task,)).fetchone()[0]
            if status not in {"ready", "running"}:
                break
            await asyncio.sleep(3)
        with sqlite3.connect(f"file:{get_hermes_home() / 'state.db'}?mode=ro", uri=True) as conn:
            tools = dict(conn.execute("select tool_name, count(*) from messages where role='tool' and timestamp>=? group by tool_name", (since,)).fetchall())
        emit(step="c6-card", shape=shape, task=task, status=status, runs=runs, tools=tools, seconds=round(time.time() - since, 1))


async def memory(client, args):
    """Condition 6: a memory write the developer asks for is a Hermes `memory` tool call."""
    # A fresh value each time: Claude Code rightly skips a fact already in memory.
    run = time.time_ns()
    for index in range(args.tries):
        session = await open_session(client)
        result = await session.turn(f"Save to your memory, with your memory tool, this fact: probe {run}-{index} is {index * 7}.")
        emit(step="c6-memory", index=index, result=result, tools=[row[1] for row in session.rows() if row[1]])


async def watchdog(client, args):
    """Condition 11: a turn that outlives stale_timeout_seconds ends with a report, and does not run again."""
    marker = args.work / f"watchdog-{time.time_ns()}"
    seconds = args.stale_seconds + 120
    session = await open_session(client)
    started = time.monotonic()
    result = await session.turn(f"Use your Bash tool to run exactly this one command once: echo x >> {marker} ; ping -c {seconds} -i 1 127.0.0.1 >/dev/null ; then reply FINISHED.", bound=seconds + 60)
    errors = [json.loads(row[4]).get("error") for row in session.rows() if row[3] == "failed_turn"]
    emit(step="c11", result=result, elapsed=round(time.monotonic() - started, 1), marker_lines=lines(marker), chat_error=errors[-1:], rows=session.tail(1))


async def env_check(client, args):
    """Condition 10's process half: the plan token in no Hermes process. Needs /proc."""
    if not Path("/proc/self/environ").exists():
        emit(step="c10-env", result="Inconclusive: no /proc on this host")
        return
    for process in psutil.process_iter(["pid", "cmdline"]):
        command = " ".join(process.info["cmdline"] or ())
        if "hermes" not in command and "claude-agent-acp" not in command:
            continue
        try:
            environ = Path(f"/proc/{process.info['pid']}/environ").read_bytes()
        except OSError:
            continue
        emit(step="c10-env", pid=process.info["pid"], command=command[:80], has_plan_token=b"CLAUDE_CODE_OAUTH_TOKEN=" in environ)


PROBES = {probe.__name__: probe for probe in (stop, background, kill, models, switchback, gate, cards, memory, watchdog, env_check)}


async def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("probes", nargs="+", choices=sorted(PROBES))
    parser.add_argument("--work", type=Path, required=True, help="A scratch directory for side-effect markers.")
    parser.add_argument("--models", nargs="+", default=["claude-opus-5-5[1m]", "claude-sonnet-5-5[1m]", "claude-sonnet-5-5"])
    parser.add_argument("--acp-model", default="claude-sonnet-5-5")
    parser.add_argument("--anthropic-model", default="claude-sonnet-5-5")
    parser.add_argument("--stale-seconds", type=int, default=1800)
    parser.add_argument("--tries", type=int, default=20)
    parser.add_argument("--kanban-db", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None, help="The agent.log holding usage_update lines.")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.kanban_db = args.kanban_db or get_hermes_home() / "kanban.db"
    async with connect_gateway() as client:
        # Approval cards reach only clients that say they can answer them.
        await client.rpc("client.capabilities", server_requests=True)
        for name in args.probes:
            for index in range(args.tries if name in {"stop", "background"} else 1):
                emit(step="probe", name=name, attempt=index)
                await PROBES[name](client, args)


asyncio.run(main())
