"""One long-lived adapter and ACP session per Hermes session.

Hermes rebuilds its client often: after every interrupt, watchdog abort, and
model switch. The adapter must outlive those rebuilds, or a stop cannot reach
the running turn and background work dies with the call. So the adapter lives
here, in a module-level registry keyed by Hermes session id, and
``acp_client.ACPClient`` is a thin facade over it.

A session sends Claude Code only what it has not seen. Once a prompt is on the
wire its messages count as seen, so any Hermes retry of them finds nothing to
send and fails with a report instead of repeating the turn's side effects.
"""

from __future__ import annotations

import atexit
import contextvars
import hashlib
import itertools
import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import psutil

from .acp_format import (
    extract_tool_calls,
    message_digest,
    render_content,
    render_delta,
    render_memo,
    render_tool_spec_notice,
    reply_digest,
    system_append,
    tool_specs,
)
from .acp_gate import GATED_TOOLS, permission_outcome

logger = logging.getLogger(__name__)

_CONTROL_TIMEOUT_SECONDS = 30.0
_SESSION_TIMEOUT_SECONDS = 120.0
_TURN_LOCK_TIMEOUT_SECONDS = 30.0
# After a stop: how long Claude Code gets to end its own tools, and how long
# before a turn that still has not settled loses the whole adapter.
_REAP_AFTER_SECONDS = 3.0
_KILL_AFTER_SECONDS = 8.0
_SETTLE_WAIT_SECONDS = 10.0
_REPLAY_DROP_SECONDS = 1.0
_BETWEEN_TURN_QUIET_SECONDS = 3.0
_IDLE_SECONDS = float(os.environ.get("OCE_CLAUDE_ACP_IDLE_SECONDS") or 1800)
_MAX_SESSIONS = int(os.environ.get("OCE_CLAUDE_ACP_MAX_SESSIONS") or 8)
# Hermes shows at most 300 to 500 characters of a provider's error.
_REPORT_QUOTE_CHARS = 140
_MAX_TOOL_NAMES = 512
_PROMPT_TOO_LONG = "prompt is too long"


class ACPTurnError(RuntimeError):
    """A failure once ``session/prompt`` is on the wire. Never retried."""

    acp_prompt_sent = True


class ACPModelError(RuntimeError):
    """The requested model is not an exact match for one the adapter offers."""

    acp_model_refused = True


@dataclass
class Turn:
    """One call from Hermes. ``ACPClient.cancel`` reaches it from another thread."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    canceled: threading.Event = field(default_factory=threading.Event)
    session: "AdapterSession | None" = None
    prompt_sent: bool = False
    started_at: float = 0.0
    text_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)

    def cancel(self) -> None:
        with self.lock:
            self.canceled.set()
            session = self.session if self.prompt_sent else None
        if session is not None:
            session.stop_turn(self)


@dataclass
class TurnResult:
    text: str
    reasoning: str
    tool_calls: list[Any]
    model: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class Launch:
    command: str
    args: tuple[str, ...]
    cwd: str
    build_env: Callable[[], dict[str, str]]


def _state_path(key: str) -> Path:
    from hermes_constants import get_hermes_home

    digest = hashlib.sha256(key.encode("utf-8", "replace")).hexdigest()[:32]
    return get_hermes_home() / "claude-acp" / "sessions" / f"{digest}.json"


class AdapterSession:
    """A running adapter, its ACP session, and what Claude Code has already seen."""

    def __init__(self, key: str | None, launch: Launch):
        self.key = key
        self.launch = launch
        self.turn_lock = threading.Lock()
        self.last_used = time.monotonic()
        self.proc: subprocess.Popen[str] | None = None
        self.acp_id: str | None = None
        self.synced: list[str] | None = None
        self._awaiting_reply = False
        self._model_option: dict[str, Any] = {}
        self._mode = ""
        self._write_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._pending: dict[int, queue.SimpleQueue[dict[str, Any]]] = {}
        self._open_prompt: int | None = None
        self._prompt_settled = threading.Event()
        self._prompt_settled.set()
        self._turn: Turn | None = None
        self._drop_updates_until = 0.0
        self._dropping = False
        self._between_lock = threading.Lock()
        self._between: list[str] = []
        self._between_at = 0.0
        self._undelivered: list[str] = []
        self._delivered_notes: set[str] = set()
        self._context: contextvars.Context | None = None
        self._specs_digest = ""
        self.last_usage_size: int | None = None
        self._tool_names: dict[str, str] = {}
        self._stderr: deque[str] = deque(maxlen=40)
        self._env_home = ""
        self._state_file = _state_path(key) if key else None
        self._restore()

    # --- persistence: a restarted gateway resumes with session/load

    def _restore(self) -> None:
        if self._state_file is None or not self._state_file.exists():
            return
        try:
            state = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("claude-acp ignoring unreadable session state at %s", self._state_file)
            return
        self.acp_id = state.get("acp_id")
        self.synced = state.get("synced")
        self._awaiting_reply = bool(state.get("awaiting_reply"))

    def _persist(self) -> None:
        if self._state_file is None:
            return
        state = {"acp_id": self.acp_id, "synced": self.synced, "awaiting_reply": self._awaiting_reply}
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(json.dumps(state), encoding="utf-8")

    # --- the adapter process

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _spawn(self) -> None:
        env = self.launch.build_env()
        self._dropping = False
        self._open_prompt = None
        self._prompt_settled.set()
        self._env_home = env.get("CLAUDE_CONFIG_DIR") or str(Path(env.get("HOME", "~")) / ".claude")
        try:
            self.proc = subprocess.Popen(
                [self.launch.command, *self.launch.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self.launch.cwd,
                env=env,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Could not start the Claude Code ACP adapter '{self.launch.command}'.") from exc
        proc = self.proc
        threading.Thread(target=self._read_stdout, args=(proc,), daemon=True, name="claude-acp-stdout").start()
        threading.Thread(target=self._read_stderr, args=(proc,), daemon=True, name="claude-acp-stderr").start()
        self._request("initialize", {
            "protocolVersion": 1,
            "clientCapabilities": {},
            "clientInfo": {"name": "hermes-agent", "title": "Hermes Agent", "version": "0.0.0"},
        }, _CONTROL_TIMEOUT_SECONDS)

    def _read_stderr(self, proc: subprocess.Popen[str]) -> None:
        for line in proc.stderr or ():
            self._stderr.append(line.rstrip("\n"))

    def _read_stdout(self, proc: subprocess.Popen[str]) -> None:
        for line in proc.stdout or ():
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if isinstance(message, dict):
                self._route(message)
        # Wake every waiter, unless a new adapter already took over.
        if self.proc is proc or self.proc is None:
            self._prompt_settled.set()
            for reply in tuple(self._pending.values()):
                reply.put({"_exited": True})

    def _route(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if not isinstance(method, str):
            self._route_response(message)
        elif method == "session/update":
            self._on_update((message.get("params") or {}).get("update") or {})
        elif "id" in message:
            # Not on this thread: a human approval can take minutes, and the
            # reader must keep delivering the updates and the turn's response.
            context = (self._context or contextvars.copy_context()).copy()
            threading.Thread(target=context.run, args=(self._answer, message), daemon=True,
                             name="claude-acp-request").start()

    def _route_response(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        if request_id == self._open_prompt:
            self._open_prompt = None
            self._dropping = False
            self._prompt_settled.set()
        reply = self._pending.get(request_id)
        if reply is not None:
            reply.put(message)

    def _answer(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        response: dict[str, Any] = {"jsonrpc": "2.0", "id": message.get("id")}
        if method == "session/request_permission":
            params = message.get("params") or {}
            tool_call = params.get("toolCall")
            call_id = tool_call.get("toolCallId") if isinstance(tool_call, dict) else None
            response["result"] = permission_outcome(params, self._tool_names.get(str(call_id), ""), self.launch.cwd)
        else:
            response["error"] = {"code": -32601, "message": f"Hermes does not implement {method}."}
        self._write(response)

    def _write(self, payload: dict[str, Any]) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None:
            return
        try:
            with self._write_lock:
                proc.stdin.write(json.dumps(payload) + "\n")
                proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            logger.debug("claude-acp write after the adapter exited")

    def _send_request(self, method: str, params: dict[str, Any]) -> tuple[int, queue.SimpleQueue[dict[str, Any]]]:
        request_id = next(self._ids)
        reply: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        self._pending[request_id] = reply
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        return request_id, reply

    def _request(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        """A control-plane request: bounded, and safe to fail before any prompt."""
        request_id, reply = self._send_request(method, params)
        try:
            message = reply.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"The Claude Code ACP adapter did not answer {method} in {timeout:.0f} s.") from exc
        finally:
            self._pending.pop(request_id, None)
        if message.get("_exited"):
            raise RuntimeError(f"The Claude Code ACP adapter exited during {method}: {self._stderr_tail()}")
        if "error" in message:
            error = message["error"] or {}
            raise RuntimeError(f"Claude Code ACP {method} failed: {error.get('message') or error}")
        return message.get("result") or {}

    def _stderr_tail(self) -> str:
        return " ".join(self._stderr)[-300:] or "no output"

    def kill(self) -> None:
        """End the adapter and everything under it. The ACP session stays resumable."""
        proc = self.proc
        self.proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            family = [*psutil.Process(proc.pid).children(recursive=True)]
        except psutil.Error:
            family = []
        proc.kill()
        for child in family:
            try:
                child.kill()
            except psutil.Error:
                pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            logger.warning("claude-acp adapter %s did not exit", proc.pid)

    # --- updates

    def _on_update(self, update: dict[str, Any]) -> None:
        kind = str(update.get("sessionUpdate") or "")
        if kind == "usage_update":
            # A [1m] session must report a size of 1,000,000. This line is the evidence.
            self.last_usage_size = update.get("size")
            logger.info("claude-acp usage_update session=%s acp=%s used=%s size=%s",
                        self.key, self.acp_id, update.get("used"), update.get("size"))
            return
        if kind == "current_mode_update":
            self._mode = str(update.get("currentModeId") or "")
            return
        if kind in {"tool_call", "tool_call_update"}:
            self._note_tool_name(update)
            return
        if self._dropping or time.monotonic() < self._drop_updates_until:
            return
        content = update.get("content")
        text = str(content.get("text") or "") if isinstance(content, dict) else ""
        if not text:
            return
        turn = self._turn
        if kind == "agent_message_chunk":
            if turn is not None:
                turn.text_parts.append(text)
            else:
                with self._between_lock:
                    self._between.append(text)
                    self._between_at = time.monotonic()
        elif kind == "agent_thought_chunk" and turn is not None:
            turn.reasoning_parts.append(text)

    def _note_tool_name(self, update: dict[str, Any]) -> None:
        meta = update.get("_meta")
        claude_code = meta.get("claudeCode") if isinstance(meta, dict) else None
        name = claude_code.get("toolName") if isinstance(claude_code, dict) else None
        if isinstance(name, str) and name:
            if len(self._tool_names) >= _MAX_TOOL_NAMES:
                self._tool_names.clear()
            self._tool_names[str(update.get("toolCallId"))] = name

    def flush_between_turn(self, *, force: bool = False) -> None:
        """Move text Claude Code wrote with no prompt in flight into the Hermes session."""
        with self._between_lock:
            if not self._between or self._turn is not None:
                return
            if not force and time.monotonic() - self._between_at < _BETWEEN_TURN_QUIET_SECONDS:
                return
            text = "".join(self._between).strip()
            self._between.clear()
        if not text:
            return
        note = f"[Claude Code, between turns]\n{text}"
        context = (self._context or contextvars.copy_context()).copy()
        delivered = bool(self.key) and context.run(self._append_note, note)
        if delivered:
            self._delivered_notes.add(note)
        else:
            self._undelivered.append(note)

    def _append_note(self, note: str) -> bool:
        try:
            from hermes_state import SessionDB

            db = SessionDB()
            try:
                db.append_completion_note(self.key, uuid.uuid4().hex, note, "internal_notification",
                                          {"completion_note": ["claude-acp-between-turns"]})
            finally:
                db.close()
        except Exception as exc:
            logger.info("claude-acp holding between-turn text for the next reply: %s", exc)
            return False
        logger.info("claude-acp recorded between-turn text in session %s", self.key)
        return True

    # --- the ACP session

    def _session_meta(self, messages: list[dict[str, Any]], tools: Any) -> dict[str, Any]:
        system_text = "\n\n".join(
            render_content(message.get("content")) for message in messages if message.get("role") == "system")
        return {
            "systemPrompt": {"append": system_append(system_text, tool_specs(tools))},
            # Settings files could carry hooks and allow rules that skip the
            # gate. Hermes's system prompt already holds the project's context
            # files, so Claude Code loses no instructions by reading none.
            "claudeCode": {"options": {
                "settingSources": [],
                "settings": {"permissions": {"ask": list(GATED_TOOLS)}},
            }},
        }

    def _ensure_session(self, messages: list[dict[str, Any]], tools: Any) -> None:
        if self.alive and self.acp_id:
            return
        if not self.alive:
            self._spawn()
        params = {"cwd": self.launch.cwd, "mcpServers": [], "_meta": self._session_meta(messages, tools)}
        resumed: dict[str, Any] | None = None
        if self.acp_id:
            self._dropping = True  # session/load replays the history as updates
            try:
                resumed = self._request("session/load", {**params, "sessionId": self.acp_id}, _SESSION_TIMEOUT_SECONDS)
            except (RuntimeError, TimeoutError) as exc:
                logger.warning("claude-acp could not resume %s, starting fresh: %s", self.acp_id, exc)
                self.acp_id = None
                self.synced = None
            finally:
                self._dropping = False
                self._drop_updates_until = time.monotonic() + _REPLAY_DROP_SECONDS
        if resumed is None:
            created = self._request("session/new", params, _SESSION_TIMEOUT_SECONDS)
            self.acp_id = str(created.get("sessionId") or "") or None
            if self.acp_id is None:
                raise RuntimeError("The Claude Code ACP adapter returned no sessionId.")
            self.synced = None
            resumed = created
        self._specs_digest = self._digest_specs(tools)
        self._capture_model_option(resumed)
        self._set_mode_default()
        # session/load resets the model to the adapter's default, whatever it
        # reports, so the next pin must always send the set.
        self._model_option["currentValue"] = ""

    def _set_mode_default(self) -> None:
        self._request("session/set_mode", {"sessionId": self.acp_id, "modeId": "default"}, _CONTROL_TIMEOUT_SECONDS)
        self._mode = "default"

    def _capture_model_option(self, result: dict[str, Any]) -> None:
        for option in result.get("configOptions") or ():
            if isinstance(option, dict) and (option.get("category") == "model" or option.get("id") == "model"):
                self._model_option = option
                return

    def _pin_model(self, model: str) -> str:
        """Set exactly the model Hermes asked for, or fail before any prompt."""
        option = self._model_option
        offered = [str(choice.get("value")) for choice in option.get("options") or () if isinstance(choice, dict)]
        if model not in offered:
            raise ACPModelError(f"Model '{model}' is not offered by Claude Code. Available: {', '.join(offered) or 'none'}")
        if option.get("currentValue") == model:
            return model
        result = self._request("session/set_config_option", {
            "sessionId": self.acp_id, "configId": str(option.get("id") or "model"), "value": model,
        }, _CONTROL_TIMEOUT_SECONDS)
        self._capture_model_option(result)
        confirmed = str(self._model_option.get("currentValue") or "")
        if confirmed != model:
            raise ACPModelError(f"Claude Code confirmed model '{confirmed}' after a request for '{model}'.")
        return model

    @staticmethod
    def _digest_specs(tools: Any) -> str:
        return hashlib.sha256(json.dumps(tool_specs(tools), sort_keys=True, default=str).encode()).hexdigest()

    # --- what to send

    def _plan_prompt(self, messages: list[dict[str, Any]], tools: Any) -> tuple[str, list[str]]:
        body = [message for message in messages if message.get("role") != "system"
                and render_content(message.get("content")) not in self._delivered_notes]
        digests = [message_digest(message) for message in body]
        synced = self.synced
        if synced is None:
            return render_memo(body), digests
        if digests[:len(synced)] == synced:
            delta = body[len(synced):]
            if not delta:
                raise ACPTurnError(self._report("Refused to resend a turn Claude Code already started."))
            text = render_delta(delta)
        elif self._awaiting_reply and digests and synced and digests[-1] == synced[-1]:
            raise ACPTurnError(self._report("Refused to resend a turn Claude Code already started."))
        else:
            # Compression or /retry rewrote the history under this session.
            logger.info("claude-acp session %s history rewritten; sending a memo", self.key)
            text = render_memo(body)
        specs_digest = self._digest_specs(tools)
        if specs_digest != self._specs_digest:
            text = f"{render_tool_spec_notice(tool_specs(tools))}\n\n{text}"
            self._specs_digest = specs_digest
        return text, digests

    # --- a turn

    def run_turn(self, turn: Turn, *, messages: list[dict[str, Any]], tools: Any, model: str, timeout: float,
                 context: contextvars.Context) -> TurnResult:
        if not self.turn_lock.acquire(timeout=_TURN_LOCK_TIMEOUT_SECONDS):
            raise RuntimeError("Another turn is still running in this Claude Code session.")
        try:
            self._context = context
            self.last_used = time.monotonic()
            return self._run_turn_locked(turn, messages, tools, model, timeout)
        finally:
            self._turn = None
            self.last_used = time.monotonic()
            self.turn_lock.release()

    def _run_turn_locked(self, turn: Turn, messages: list[dict[str, Any]], tools: Any, model: str,
                         timeout: float) -> TurnResult:
        self.flush_between_turn(force=True)
        if not self._prompt_settled.wait(_SETTLE_WAIT_SECONDS):
            logger.warning("claude-acp session %s: a stopped turn never settled; restarting the adapter", self.key)
            self.kill()
            self._prompt_settled.set()
        self._ensure_session(messages, tools)
        served = self._pin_model(model)
        if self._mode != "default":
            self._set_mode_default()
        text, digests = self._plan_prompt(messages, tools)
        if self._undelivered:
            turn.text_parts.extend(f"{note}\n\n" for note in self._undelivered)
            self._undelivered.clear()
        prompt = {"sessionId": self.acp_id, "prompt": [{"type": "text", "text": text}]}

        with turn.lock:
            if turn.canceled.is_set():
                raise RuntimeError("The turn was stopped before its prompt was sent.")
            # Seen from this moment: a retry of these messages must find nothing to send.
            self.synced = digests
            self._awaiting_reply = True
            self._persist()
            turn.session = self
            turn.prompt_sent = True
            turn.started_at = time.time()
            self._turn = turn
            # Replay stragglers end here: what follows is this prompt's reply.
            self._drop_updates_until = 0.0
            self._prompt_settled.clear()
            request_id, reply = self._send_request("session/prompt", prompt)
            self._open_prompt = request_id
        try:
            response = self._await_prompt(turn, reply, timeout)
        finally:
            self._pending.pop(request_id, None)
        return self._finish(turn, response, served)

    def _await_prompt(self, turn: Turn, reply: queue.SimpleQueue[dict[str, Any]], timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            if turn.canceled.is_set():
                raise ACPTurnError(self._report("Claude Code was stopped.", turn))
            if time.monotonic() >= deadline:
                turn.cancel()
                raise ACPTurnError(self._report(f"Claude Code was stopped after {timeout:.0f} s.", turn))
            try:
                message = reply.get(timeout=0.1)
            except queue.Empty:
                continue
            if message.get("_exited"):
                self.kill()
                raise ACPTurnError(self._report("The Claude Code adapter exited during the turn.", turn))
            if "error" in message:
                error = message["error"] or {}
                detail = str(error.get("message") or error)
                if _PROMPT_TOO_LONG in detail.lower():
                    # Claude Code cannot continue this session. The next turn starts another.
                    self.acp_id = None
                    self.synced = None
                    self._persist()
                raise ACPTurnError(self._report(f"Claude Code failed: {detail[:120]}.", turn))
            return message.get("result") or {}

    def _finish(self, turn: Turn, response: dict[str, Any], served: str) -> TurnResult:
        if response.get("stopReason") == "cancelled":
            raise ACPTurnError(self._report("Claude Code canceled the turn.", turn))
        self._turn = None
        tool_calls, text = extract_tool_calls("".join(turn.text_parts))
        self.synced = [*(self.synced or ()), reply_digest(tool_calls)]
        self._awaiting_reply = False
        self._persist()
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        prompt_tokens = sum(int(usage.get(key) or 0) for key in ("inputTokens", "cachedReadTokens", "cachedWriteTokens"))
        return TurnResult(
            # An empty reply would send Hermes into its empty-response retry.
            text=text or ("" if tool_calls else "(Claude Code ended the turn with no text.)"),
            reasoning="".join(turn.reasoning_parts),
            tool_calls=tool_calls,
            model=served,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(usage.get("outputTokens") or 0),
        )

    # --- stopping

    def stop_turn(self, turn: Turn) -> None:
        """Send ``session/cancel`` and make sure the turn's processes end. Never blocks the caller."""
        self._dropping = True
        self._write({"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": self.acp_id}})
        threading.Thread(target=self._reap, args=(turn,), daemon=True, name="claude-acp-reap").start()

    def _reap(self, turn: Turn) -> None:
        proc = self.proc
        if proc is None:
            return
        self._prompt_settled.wait(_REAP_AFTER_SECONDS)
        self._kill_turn_children(proc, turn)
        if not self._prompt_settled.wait(max(0.0, _KILL_AFTER_SECONDS - _REAP_AFTER_SECONDS)):
            logger.warning("claude-acp session %s: the stopped turn did not settle; killing the adapter", self.key)
            self.kill()
            self._prompt_settled.set()

    @staticmethod
    def _kill_turn_children(proc: subprocess.Popen[str], turn: Turn) -> None:
        """Kill what this turn started, sparing Claude Code and what earlier turns left running."""
        try:
            root = psutil.Process(proc.pid)
            spared = {child.pid for child in root.children()}
            for child in root.children(recursive=True):
                if child.pid not in spared and child.create_time() >= turn.started_at:
                    logger.info("claude-acp killing %s left by a stopped turn", child.pid)
                    child.kill()
        except psutil.Error:
            logger.debug("claude-acp process sweep raced an exit", exc_info=True)

    # --- the report a failed turn leaves

    def _transcript_path(self) -> str:
        slug = re.sub(r"[^A-Za-z0-9]", "-", self.launch.cwd)
        return str(Path(self._env_home or "~/.claude", "projects", slug, f"{self.acp_id}.jsonl"))

    def _last_transcript_text(self) -> str:
        """The start of Claude Code's last message: its text, or the tool call it was running."""
        try:
            lines = Path(self._transcript_path()).read_text(encoding="utf-8").splitlines()
        except OSError:
            return ""
        for line in reversed(lines):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("type") != "assistant":
                continue
            content = (row.get("message") or {}).get("content")
            text = render_content(content)
            if text:
                return text
            for block in content if isinstance(content, list) else ():
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    return f"{block.get('name')}: {json.dumps(block.get('input'))}"
        return ""

    def _report(self, why: str, turn: Turn | None = None) -> str:
        """Hermes's chat shows only this error, so it says where the work is and how far it got."""
        text = "".join(turn.text_parts).strip() if turn is not None else ""
        quote = " ".join((text or self._last_transcript_text()).split())[:_REPORT_QUOTE_CHARS]
        # Hermes's own row for a failed turn says the request was not processed, which is not so here.
        return f"{why} Some actions may already have run. Transcript: {self._transcript_path()} Last message: {quote or '(none)'}"


class _Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, AdapterSession] = {}
        self._janitor: threading.Thread | None = None

    def get(self, key: str, launch: Launch) -> AdapterSession:
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = AdapterSession(key, launch)
                self._sessions[key] = session
            session.last_used = time.monotonic()
            if self._janitor is None:
                self._janitor = threading.Thread(target=self._sweep, daemon=True, name="claude-acp-janitor")
                self._janitor.start()
                atexit.register(self.close_all)
            return session

    def _sweep(self) -> None:
        while True:
            time.sleep(1)
            self.tick()

    def tick(self) -> None:
        with self._lock:
            sessions = tuple(self._sessions.items())
        now = time.monotonic()
        by_age = sorted(sessions, key=lambda item: item[1].last_used)
        excess = len(sessions) - _MAX_SESSIONS
        for index, (key, session) in enumerate(by_age):
            session.flush_between_turn()
            over_limit = index < excess
            if (over_limit or now - session.last_used > _IDLE_SECONDS) and session.turn_lock.acquire(blocking=False):
                try:
                    logger.info("claude-acp evicting idle session %s", key)
                    session.kill()
                    with self._lock:
                        self._sessions.pop(key, None)
                finally:
                    session.turn_lock.release()

    def close_all(self) -> None:
        with self._lock:
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.kill()


REGISTRY = _Registry()
