"""OpenAI-compatible facade that hands Hermes's turns to Claude Code over ACP.

Generalized from the original Copilot-only client (issue #5257). This client
holds no session state. Hermes rebuilds it after every interrupt and abort, so
the adapter and ACP session live in the registry in ``acp_session``. Permission
requests go through ``acp_gate``.
"""

from __future__ import annotations

import contextvars
import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .acp_agent_registry import (
    agent_env_unset,
    normalize_agent_name,
    resolve_agent_launch,
)
from .acp_models import pin_available_models
from .acp_session import REGISTRY, AdapterSession, Launch, Turn, TurnResult
from agent.runtime_cwd import resolve_agent_cwd
from tools.environments.local import hermes_subprocess_env

logger = logging.getLogger(__name__)

ACP_MARKER_PREFIX = "acp://"
_DEFAULT_TIMEOUT_SECONDS = 900.0


def extract_agent_from_url(base_url: Any) -> str | None:
    """Return the agent name from an ``acp://{agent}`` URL, else ``None``."""
    text = str(base_url or "").strip()
    if not text.lower().startswith(ACP_MARKER_PREFIX):
        return None
    name = normalize_agent_name(text[len(ACP_MARKER_PREFIX):].split("/", 1)[0])
    return name or None


def marker_base_url(agent_name: str) -> str:
    return f"{ACP_MARKER_PREFIX}{normalize_agent_name(agent_name)}"


def _resolve_home_dir() -> str:
    """Return a stable HOME for child ACP processes."""
    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok — POSIX fallback inside try/except (pwd import fails on Windows)
        if resolved:
            return resolved
    except Exception:
        pass

    # Last resort: /tmp (writable on any POSIX system). Avoids crashing the
    # subprocess with no HOME; callers can set HERMES_HOME explicitly if they
    # need a different writable dir.
    return "/tmp"


# Where the pod mounts the Claude Code plan token. It sits outside Hermes's
# model-key mount so no Hermes process ever loads it; only the adapter's
# environment gets it.
_TOKEN_FILE_DEFAULT = "/var/run/secrets/hermes-claude/CLAUDE_CODE_OAUTH_TOKEN"
_TOKEN_FILE_ENV = "OCE_CLAUDE_ACP_TOKEN_FILE"
# Local runs on a developer's machine sign in with that machine's login.
_USE_LOGIN_ENV = "OCE_CLAUDE_ACP_USE_LOGIN"
# Claude Code prefers each of these to CLAUDE_CODE_OAUTH_TOKEN.
_CREDENTIAL_SWITCHES = ("CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_FOUNDRY")
# Auto-compaction keeps a long session alive; nothing may turn it off.
_COMPACTION_SWITCHES = ("DISABLE_AUTO_COMPACT", "DISABLE_COMPACT")


def _read_plan_token() -> str | None:
    """The plan token for the adapter, or None when this run uses the machine's login."""
    if os.environ.get(_USE_LOGIN_ENV):
        return None
    path = Path(os.environ.get(_TOKEN_FILE_ENV) or _TOKEN_FILE_DEFAULT)
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Cannot read the Claude Code plan token at {path}: {exc.strerror}") from exc
    if not token:
        raise RuntimeError(f"The Claude Code plan token at {path} is empty.")
    return token


def _build_subprocess_env(env_unset: tuple[str, ...] = ()) -> dict[str, str]:
    # The adapter bills on the developer's plan, so it inherits no provider
    # credentials: the helper strips those and Tier-1 secrets alike.
    env = hermes_subprocess_env(inherit_credentials=False)
    home = _resolve_home_dir()
    env["HOME"] = home
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)
    # Per-agent session markers to strip (e.g. the Claude Code bridge won't
    # launch inside a parent Claude Code session) — declared on the registry
    # entry so this stays agent-agnostic.
    for marker in env_unset:
        env.pop(marker, None)
    for name in tuple(env):
        if name.startswith("ANTHROPIC_") or name in _CREDENTIAL_SWITCHES or name in _COMPACTION_SWITCHES:
            del env[name]
    token = _read_plan_token()
    if token is None:
        env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    else:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
        pin_available_models(env.get("CLAUDE_CONFIG_DIR") or Path(env["HOME"]) / ".claude")
    return env


def _hermes_session_id() -> str:
    """The id of the Hermes session this call belongs to, or "" outside one.

    ``run_agent`` publishes it through ``gateway.session_context``, and the
    worker thread that makes the call inherits that context.
    """
    try:
        from gateway.session_context import get_session_env
    except ImportError:
        return os.environ.get("HERMES_SESSION_ID", "")
    return get_session_env("HERMES_SESSION_ID", "")


def _session_cwd(explicit: str | None) -> str:
    """The working directory for a new adapter. A card's scratch workspace may be gone by now."""
    from hermes_constants import get_hermes_home

    try:
        path = Path(explicit) if explicit else resolve_agent_cwd()
    except OSError:
        path = get_hermes_home()
    if not path.is_dir():
        path = get_hermes_home()
    return str(path.resolve())


def _timeout_seconds(timeout: Any) -> float:
    """Hermes may pass an httpx.Timeout rather than a plain float."""
    if isinstance(timeout, (int, float)):
        return float(timeout)
    candidates = [getattr(timeout, attr, None) for attr in ("read", "write", "connect", "pool", "timeout")]
    numeric = [float(value) for value in candidates if isinstance(value, (int, float))]
    return max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS


def _completion(result: TurnResult) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.prompt_tokens + result.completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    message = SimpleNamespace(
        content=result.text,
        tool_calls=result.tool_calls,
        reasoning=result.reasoning or None,
        reasoning_content=result.reasoning or None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(message=message, finish_reason="tool_calls" if result.tool_calls else "stop")
    return SimpleNamespace(choices=[choice], usage=usage, model=result.model)


def _completion_to_stream_chunks(completion: SimpleNamespace) -> list[SimpleNamespace]:
    """Convert a one-shot ACP response into OpenAI-style stream chunks."""
    choice = completion.choices[0]
    message = choice.message
    tool_call_deltas = None
    if message.tool_calls:
        tool_call_deltas = [
            SimpleNamespace(
                index=index,
                id=getattr(tool_call, "id", None),
                type=getattr(tool_call, "type", "function"),
                function=SimpleNamespace(
                    name=getattr(tool_call.function, "name", None),
                    arguments=getattr(tool_call.function, "arguments", None),
                ),
            )
            for index, tool_call in enumerate(message.tool_calls)
        ]
    delta = SimpleNamespace(
        role="assistant",
        content=message.content or None,
        tool_calls=tool_call_deltas,
        reasoning_content=message.reasoning_content,
        reasoning=message.reasoning,
    )
    data_chunk = SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=choice.finish_reason)],
        model=completion.model,
        usage=None,
    )
    usage_chunk = SimpleNamespace(choices=[], model=completion.model, usage=completion.usage)
    return [data_chunk, usage_chunk]


class _ACPChatCompletions:
    def __init__(self, client: "ACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "ACPClient"):
        self.completions = _ACPChatCompletions(client)


class ACPClient:
    """Minimal OpenAI-client-compatible facade for Claude Code's ACP adapter."""

    # Declared for agent/auxiliary_client.py: this shim drives an ACP
    # subprocess over stdio, so it is already a complete client (never
    # re-dispatch it through a wire adapter) and is safe to call from async
    # code as-is. Without these flags the auxiliary client (compression,
    # vision) wraps it in an HTTP adapter and every aux call fails.
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self,
        *,
        agent_name: str = "claude",
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **_: Any,
    ):
        self.agent_name = extract_agent_from_url(base_url) or normalize_agent_name(agent_name) or "claude"
        self.api_key = api_key or f"{self.agent_name}-acp"
        self.base_url = base_url or marker_base_url(self.agent_name)
        self._default_headers = dict(default_headers or {})
        explicit_command = acp_command or command
        if explicit_command:
            try:
                _, resolved_args = resolve_agent_launch(self.agent_name)
            except ValueError:
                resolved_args = []
            self._acp_command = explicit_command
        else:
            self._acp_command, resolved_args = resolve_agent_launch(self.agent_name)
        self._acp_args = tuple(acp_args or args or resolved_args)
        self._acp_cwd = acp_cwd
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._turn: Turn | None = None
        self._turn_lock = threading.Lock()

    def close(self) -> None:
        """Hermes closes this client on every rebuild. The adapter must outlive that."""
        self.is_closed = True

    def cancel(self) -> None:
        """Hermes's interrupt and watchdog call this from another thread.

        The pending call raises within Hermes's 2 s wait, and Claude Code's
        processes stop whether or not it cooperates.
        """
        with self._turn_lock:
            turn = self._turn
        if turn is not None:
            turn.cancel()

    def _launch(self) -> Launch:
        env_unset = agent_env_unset(self.agent_name)
        return Launch(
            command=self._acp_command,
            args=self._acp_args,
            cwd=_session_cwd(self._acp_cwd),
            build_env=lambda: _build_subprocess_env(env_unset=env_unset),
        )

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        turn = Turn()
        with self._turn_lock:
            self._turn = turn
        context = contextvars.copy_context()
        session_id = _hermes_session_id()
        # A call with no tools is auxiliary work such as a title or a summary.
        # It must not land in the session's conversation with Claude Code.
        one_shot = not (session_id and tools)
        if one_shot:
            session = AdapterSession(None, self._launch())
        else:
            session = REGISTRY.get(session_id, self._launch())
        logger.info("claude-acp call: session=%s one_shot=%s", session_id, one_shot)
        try:
            result = session.run_turn(
                turn,
                messages=list(messages or ()),
                tools=tools,
                model=model or "",
                timeout=_timeout_seconds(timeout) if timeout is not None else _DEFAULT_TIMEOUT_SECONDS,
                context=context,
            )
        finally:
            with self._turn_lock:
                self._turn = None
            if one_shot:
                session.kill()
        completion = _completion(result)
        return _completion_to_stream_chunks(completion) if stream else completion
