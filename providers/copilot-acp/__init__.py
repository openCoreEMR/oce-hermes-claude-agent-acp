"""GitHub Copilot ACP provider — standalone plugin against the core client seam.

Hermes core bundles its own ``copilot-acp`` plugin (a Copilot-specific
client); installing THIS plugin overrides it with the generalized ACP
client the sibling ``claude-acp``/``codex-acp`` plugins share — user
plugins are discovered after bundled ones, so this registration wins.
Skip installing this directory if the bundled client is all you need.
Same seam architecture as ``claude-acp``; see that plugin's docstring.
"""

import shutil

from providers import register_provider
from providers.base import ProviderProfile

from .acp_agent_registry import ACPAgentEntry, register_acp_agent

_INSTALL_HINT = (
    "Install GitHub Copilot CLI (npm install -g @github/copilot) or "
    "set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH."
)


def _resolve_default_command() -> str:
    return shutil.which("copilot") or "copilot"


register_acp_agent("copilot", ACPAgentEntry(
    command=_resolve_default_command(),
    args=("--acp", "--stdio"),
    display_name="GitHub Copilot",
    install_hint=_INSTALL_HINT,
    legacy_command_env_vars=("HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH"),
    legacy_args_env_var="HERMES_COPILOT_ACP_ARGS",
))


class CopilotACPProfile(ProviderProfile):
    """GitHub Copilot ACP — external process, no REST models endpoint."""

    def create_client(self, **client_kwargs):
        from .acp_client import ACPClient

        return ACPClient(agent_name="copilot", **client_kwargs)

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return None


copilot_acp = CopilotACPProfile(
    name="copilot-acp",
    aliases=("github-copilot-acp", "copilot-acp-agent"),
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    display_name="GitHub Copilot ACP",
    description="GitHub Copilot ACP (spawns copilot --acp --stdio)",
    env_vars=(),  # managed by the Copilot CLI's own login
    base_url="acp://copilot",  # ACP internal scheme
    auth_type="external_process",
    process_command=_resolve_default_command(),
    process_args=("--acp", "--stdio"),
    process_command_env_vars=("HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH"),
    process_args_env_var="HERMES_COPILOT_ACP_ARGS",
    fallback_models=("copilot-acp",),
)

register_provider(copilot_acp)
