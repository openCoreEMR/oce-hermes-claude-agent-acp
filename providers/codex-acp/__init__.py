"""Codex CLI ACP provider — standalone plugin against the core client seam.

Drives the Codex CLI through its ACP adapter (@zed-industries/codex-acp) as
an external subprocess. Same architecture as the sibling ``claude-acp``
plugin: the generalized ACP client is vendored (``acp_client.py``) and
supplied through ``ProviderProfile.create_client()`` (Hermes #101456 seam);
see that plugin's docstring for the full rationale, including why the
launch command is pre-resolved at import time.
"""

import shutil

from providers import register_provider
from providers.base import ProviderProfile

from .acp_agent_registry import ACPAgentEntry, register_acp_agent

_COMMAND_CANDIDATES = ("codex-acp",)

_INSTALL_HINT = (
    "Install the official adapter: npm install -g @zed-industries/codex-acp "
    "(requires Codex CLI login or OPENAI_API_KEY), or point "
    "HERMES_CODEX_ACP_COMMAND at the binary."
)


def _resolve_default_command() -> str:
    for name in _COMMAND_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    return _COMMAND_CANDIDATES[0]


register_acp_agent("codex", ACPAgentEntry(
    command=_resolve_default_command(),
    display_name="Codex CLI",
    install_hint=_INSTALL_HINT,
    legacy_command_env_vars=("HERMES_CODEX_ACP_COMMAND",),
    legacy_args_env_var="HERMES_CODEX_ACP_ARGS",
))


class CodexACPProfile(ProviderProfile):
    """Codex CLI ACP — external process, no REST models endpoint."""

    def create_client(self, **client_kwargs):
        from .acp_client import ACPClient

        return ACPClient(agent_name="codex", **client_kwargs)

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return None


codex_acp = CodexACPProfile(
    name="codex-acp",
    aliases=("codex-acp-agent",),
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    display_name="Codex CLI ACP",
    description="Codex CLI ACP (spawns codex-acp; reuses Codex CLI login)",
    env_vars=("CODEX_ACP_BASE_URL",),  # credentials are managed by Codex CLI's own login
    base_url="acp://codex",  # ACP internal scheme
    auth_type="external_process",
    process_command=_resolve_default_command(),
    process_args=(),  # the adapter speaks ACP over stdio with no extra flags
    process_command_env_vars=("HERMES_CODEX_ACP_COMMAND",),
    process_args_env_var="HERMES_CODEX_ACP_ARGS",
    fallback_models=("codex-acp",),
)

register_provider(codex_acp)
