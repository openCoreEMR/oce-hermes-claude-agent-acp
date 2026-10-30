"""The approval gate for Claude Code's tool calls.

Claude Code asks through ``session/request_permission``. The session forces
that question for every tool Hermes has a guard for (see ``GATED_TOOLS``), so
each call lands here and meets the guard Hermes's own tool would meet: shell
commands go to ``check_all_command_guards``, file tools to ``agent.file_safety``.

Anything without a guard is refused. That covers MCP tools, so no tool call
runs without a decision. Only one-shot options are ever selected:
``allow_always`` would write a rule that skips this gate from then on.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.file_safety import (
    get_read_block_error,
    get_write_denied_error,
    is_write_approval_required,
)

logger = logging.getLogger(__name__)

_SHELL_TOOLS = frozenset({"Bash"})
_READ_TOOLS = frozenset({"Read", "Grep", "Glob"})
_WRITE_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})
# Hermes's own web tools run with no gate, so these need none either.
_UNGATED_TOOLS = frozenset({"WebFetch", "WebSearch"})

# The session's `ask` rules: Claude Code would otherwise run reads inside the
# working directory and commands it considers read-only without asking.
GATED_TOOLS = tuple(sorted(_SHELL_TOOLS | _READ_TOOLS | _WRITE_TOOLS | _UNGATED_TOOLS))

_PATH_KEYS = ("file_path", "notebook_path", "path")


def _tool_path(raw_input: dict[str, Any], cwd: str) -> str:
    for key in _PATH_KEYS:
        value = raw_input.get(key)
        if isinstance(value, str) and value.strip():
            return str(Path(cwd, value.strip()))
    return ""


def _shell_allowed(raw_input: dict[str, Any]) -> tuple[bool, str]:
    command = str(raw_input.get("command") or "").strip()
    if not command:
        return False, "no command in the request"
    from tools.approval import check_all_command_guards

    # env_type="local": the adapter is a subprocess of this container, so its
    # commands act on us. A sandboxed backend name would skip the guards.
    verdict = check_all_command_guards(command, "local") or {}
    return bool(verdict.get("approved")), str(verdict.get("message") or command)


def _read_allowed(raw_input: dict[str, Any], cwd: str) -> tuple[bool, str]:
    path = _tool_path(raw_input, cwd)
    # A Grep or Glob with no path searches the working directory.
    blocked = get_read_block_error(path or cwd)
    return blocked is None, blocked or path or cwd


def _write_allowed(raw_input: dict[str, Any], cwd: str) -> tuple[bool, str]:
    path = _tool_path(raw_input, cwd)
    if not path:
        return False, "no path in the request"
    denied = get_write_denied_error(path)
    if denied:
        return False, denied
    # Hermes's file tools hold these for a human. This bridge has no card to
    # hold them on, so it refuses them as those tools do with no human present.
    if is_write_approval_required(path):
        return False, f"{path} requires interactive approval"
    return True, path


def tool_call_allowed(tool_call: Any, name: str, cwd: str) -> tuple[bool, str]:
    """Return ``(allowed, why)`` for a permission request's tool call.

    ``name`` comes from the ``tool_call`` update the adapter sends first: the
    request itself carries only a display title.
    """
    if not isinstance(tool_call, dict):
        return False, "malformed request"
    raw_input = tool_call.get("rawInput")
    raw_input = raw_input if isinstance(raw_input, dict) else {}
    if name in _SHELL_TOOLS:
        return _shell_allowed(raw_input)
    if name in _READ_TOOLS:
        return _read_allowed(raw_input, cwd)
    if name in _WRITE_TOOLS:
        return _write_allowed(raw_input, cwd)
    if name in _UNGATED_TOOLS:
        return True, name
    return False, f"no Hermes guard for tool {name or tool_call.get('title')!r}"


def _option_id(options: Any, kind: str) -> str | None:
    if not isinstance(options, list):
        return None
    for option in options:
        if isinstance(option, dict) and option.get("kind") == kind:
            return str(option.get("optionId") or "") or None
    return None


def permission_outcome(params: dict[str, Any], name: str, cwd: str) -> dict[str, Any]:
    """Build the ``session/request_permission`` result. Any failure refuses."""
    try:
        allowed, why = tool_call_allowed(params.get("toolCall"), name, cwd)
    except Exception:
        logger.exception("claude-acp permission check failed; refusing")
        allowed, why = False, "the check failed"
    logger.info("claude-acp permission %s: %s", "allowed" if allowed else "refused", why[:200])
    option_id = _option_id(params.get("options"), "allow_once" if allowed else "reject_once")
    if option_id is None:
        return {"outcome": {"outcome": "cancelled"}}
    return {"outcome": {"outcome": "selected", "optionId": option_id}}
