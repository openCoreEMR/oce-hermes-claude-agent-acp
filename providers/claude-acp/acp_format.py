"""Hermes messages to ACP prompt text, and Claude Code's reply back to tool calls."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}", re.DOTALL)

# A session that replaces one with history gets at most this much of it.
_MEMO_MAX_CHARS = 60_000

_BRIDGE_PREAMBLE = (
    "You are the active ACP agent backend for Hermes. Do the work with your own tools.\n"
    "Hermes's own tools are not available as native tools. To use one, write a "
    "<tool_call>{...}</tool_call> block holding one JSON object in OpenAI function-call shape, with "
    "id, type, and function{name,arguments}, where arguments is a JSON string. Hermes runs the call and "
    "sends you the result in your next prompt. Never edit Hermes's memory files or run the `hermes` "
    "command to stand in for these tools."
)


def tool_specs(tools: Any) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for tool in tools if isinstance(tools, list) else ():
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name.strip():
            specs.append({
                "name": name.strip(),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
    return specs


def render_tool_spec_notice(specs: list[dict[str, Any]]) -> str:
    return "Hermes tools available (OpenAI function schema):\n" + json.dumps(specs, ensure_ascii=False)


def system_append(system_text: str, specs: list[dict[str, Any]]) -> str:
    """The text appended to Claude Code's system prompt, which survives its auto-compaction."""
    sections = [_BRIDGE_PREAMBLE]
    if specs:
        sections.append(render_tool_spec_notice(specs))
    if system_text:
        sections.append("Hermes's system prompt for this session:\n\n" + system_text)
    return "\n\n".join(sections)


def render_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                parts.append(item["text"].strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _tool_call_ids(message: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for call in message.get("tool_calls") or ():
        call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        ids.append(str(call_id))
    return ids


def message_digest(message: dict[str, Any]) -> str:
    """A position-independent fingerprint of one Hermes message.

    An assistant message is identified by its tool call ids alone. Hermes
    cleans reply text before storing it, so hashing that text would make every
    reply look rewritten.
    """
    role = str(message.get("role") or "")
    if role == "assistant":
        body = "\n".join(sorted(_tool_call_ids(message)))
    else:
        body = f"{message.get('tool_call_id') or ''}\n{render_content(message.get('content'))}"
    return hashlib.sha256(f"{role}\0{body}".encode("utf-8", "replace")).hexdigest()[:24]


def reply_digest(tool_calls: list[ChatCompletionMessageToolCall]) -> str:
    return message_digest({"role": "assistant", "tool_calls": tool_calls})


def _render_one(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "")
    text = render_content(message.get("content"))
    if role == "tool":
        return f"Tool result for {message.get('tool_call_id') or 'unknown call'}:\n{text}"
    label = {"user": "User", "assistant": "Assistant"}.get(role, "Context")
    return f"{label}:\n{text}" if text else ""


def render_delta(messages: list[dict[str, Any]]) -> str:
    """Prompt text for messages Claude Code has not seen."""
    if len(messages) == 1 and messages[0].get("role") == "user":
        return render_content(messages[0].get("content"))
    return "\n\n".join(part for part in map(_render_one, messages) if part)


def render_memo(messages: list[dict[str, Any]]) -> str:
    """Prompt text for a session with no history of its own: the newest messages that fit, oldest first."""
    if len(messages) == 1:
        return render_delta(messages)
    kept: list[str] = []
    used = 0
    for part in reversed([part for part in map(_render_one, messages) if part]):
        if kept and used + len(part) > _MEMO_MAX_CHARS:
            kept.append("[Earlier messages omitted.]")
            break
        kept.append(part)
        used += len(part)
    return "Conversation so far:\n\n" + "\n\n".join(reversed(kept))


def _build_tool_call(*, call_id: str, name: str, arguments: str) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id,
        call_id=call_id,
        response_item_id=None,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def extract_tool_calls(text: str) -> tuple[list[ChatCompletionMessageToolCall], str]:
    """Split a reply into Hermes tool calls and the prose around them."""
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[ChatCompletionMessageToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    def try_add(raw_json: str) -> bool:
        try:
            obj = json.loads(raw_json)
        except ValueError:
            return False
        fn = obj.get("function") if isinstance(obj, dict) else None
        fn_name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(fn_name, str) or not fn_name.strip():
            return False
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"acp_call_{len(extracted) + 1}"
        extracted.append(_build_tool_call(call_id=call_id, name=fn_name.strip(), arguments=fn_args))
        return True

    # A rejected block stays in the reply, where Hermes and the user can see it.
    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        if try_add(match.group(1)):
            consumed_spans.append(match.span())

    # Only try the bare-JSON fallback when no XML blocks were found.
    if not extracted:
        for match in _TOOL_CALL_JSON_RE.finditer(text):
            if try_add(match.group(0)):
                consumed_spans.append(match.span())

    if not consumed_spans:
        return extracted, text.strip()

    parts: list[str] = []
    cursor = 0
    for start, end in sorted(consumed_spans):
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    parts.append(text[cursor:])
    return extracted, "\n".join(part.strip() for part in parts if part.strip()).strip()
