"""Claude Code ACP provider — standalone plugin against the core client seam.

Drives Claude Code through its official ACP adapter
(@agentclientprotocol/claude-agent-acp) as an external subprocess. Since
Hermes #101456 (the #74743 split) an external-process provider needs zero
core edits: this package vendors the generalized ACP client
(``acp_client.py``, from hermes-agent PR #68222 plus the model-selection
work reviewed on this repo's PR #2) and hands it to core through
``ProviderProfile.create_client()``. Core resolves the launch command from
the ``process_*`` fields below, and the auxiliary client (compression,
vision) honours the client's ``HERMES_SKIP_*`` flags.

Launch resolution: core checks ``shutil.which(process_command)`` before it
ever calls ``create_client``, and has no notion of fallback binaries — so
the fallback chain (maintained adapter name, then the older Zed name) is
resolved HERE at import time and the winner is pinned into
``process_command`` as an absolute path. A user overrides the binary with
``HERMES_CLAUDE_ACP_COMMAND`` (or argv with ``HERMES_CLAUDE_ACP_ARGS``);
core reads those env vars first, so an override needs no reinstall.
"""

import shutil

from providers import register_provider
from providers.base import ProviderProfile

from .acp_agent_registry import ACPAgentEntry, register_acp_agent
from .acp_models import MODELS

_COMMAND_CANDIDATES = (
    "claude-agent-acp",  # maintained: @agentclientprotocol/claude-agent-acp
    "claude-code-acp",   # older @zed-industries/claude-code-acp bin
)

_INSTALL_HINT = (
    "Install the maintained adapter: "
    "npm install -g @agentclientprotocol/claude-agent-acp "
    "(the older @zed-industries/claude-code-acp bin also works), or point "
    "HERMES_CLAUDE_ACP_COMMAND at the binary. "
    "Requires a Claude Code plan token file (OCE_CLAUDE_ACP_TOKEN_FILE) or, "
    "with OCE_CLAUDE_ACP_USE_LOGIN set, the machine's Claude Code login."
)


def _resolve_default_command() -> str:
    """First candidate on PATH, pinned absolute; else the primary name.

    Pinning the absolute path keeps the provider visible and spawnable in
    minimal-PATH processes (desktop-spawned backends inherit Finder's PATH,
    which usually lacks the npm prefix). When nothing resolves, the primary
    name is returned so core's own which() check raises the actionable
    "install it or set HERMES_CLAUDE_ACP_COMMAND" error instead of this
    module failing to import.
    """
    for name in _COMMAND_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    return _COMMAND_CANDIDATES[0]


# The vendored client resolves per-agent launch details (args, env_unset,
# display name) from its own registry; the entry is registered here, by the
# plugin that owns it.
register_acp_agent("claude", ACPAgentEntry(
    command=_resolve_default_command(),
    # The Claude Code bridge refuses to launch when it detects it is running
    # inside another Claude Code session (guard keyed off these markers);
    # strip them so Hermes can drive it from any environment.
    env_unset=("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT"),
    display_name="Claude Code",
    install_hint=_INSTALL_HINT,
    legacy_command_env_vars=("HERMES_CLAUDE_ACP_COMMAND",),
    legacy_args_env_var="HERMES_CLAUDE_ACP_ARGS",
))


def _classify_api_error(error, **_):
    """Decide Hermes's next step for this provider's failures.

    Once ``session/prompt`` is on the wire, Claude Code may have acted, so
    neither a retry nor a fallback provider may repeat the turn. Anything else
    keeps Hermes's own classification: nothing has started.
    """
    if getattr(error, "acp_prompt_sent", False):
        return {"reason": "unknown", "retryable": False, "should_fallback": False}
    if getattr(error, "acp_model_refused", False):
        return {"reason": "model_not_found", "retryable": False, "should_fallback": False}
    return None


class ClaudeACPProfile(ProviderProfile):
    """Claude Code ACP — external process, no REST models endpoint."""

    def create_client(self, **client_kwargs):
        """Build the vendored generalized ACP client instead of an HTTP one.

        Core passes ``command``/``args`` (resolved from the ``process_*``
        fields, env overrides applied) plus ``base_url``/``api_key``; the
        client accepts them directly and falls back to the registry entry
        above for anything omitted.
        """
        from .acp_client import ACPClient

        return ACPClient(agent_name="claude", **client_kwargs)

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return None


claude_acp = ClaudeACPProfile(
    name="claude-acp",
    aliases=("claude-acp-agent",),
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    display_name="Claude Code ACP",
    description="Claude Code ACP (spawns claude-agent-acp; reuses Claude Code login)",
    env_vars=("CLAUDE_ACP_BASE_URL",),  # credentials are managed by Claude Code's own login
    base_url="acp://claude",  # ACP internal scheme
    auth_type="external_process",
    # How to launch the CLI — the seam's contract (#101456). The command is
    # pre-resolved above so core's which() check passes on minimal PATHs.
    process_command=_resolve_default_command(),
    process_args=(),  # the adapter speaks ACP over stdio with no extra flags
    process_command_env_vars=("HERMES_CLAUDE_ACP_COMMAND",),
    process_args_env_var="HERMES_CLAUDE_ACP_ARGS",
    # The adapter's own option values. The client matches these exactly, so a
    # new or renamed model fails loudly until this list and the pin move with it.
    fallback_models=MODELS,
    classify_api_error=_classify_api_error,
)

register_provider(claude_acp)
