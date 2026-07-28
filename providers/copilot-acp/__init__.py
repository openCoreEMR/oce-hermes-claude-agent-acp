"""GitHub Copilot ACP provider profile + agent registration.

copilot-acp ships bundled with Hermes core (agent/copilot_acp_client.py,
hermes_cli/auth.py's PROVIDER_REGISTRY, and a built-in "copilot" entry in
agent/acp_agent_registry.py) — this copy exists so hermes-acp-agents is a
complete, self-contained reference for all three launch agents and works
against older Hermes installs that predate the built-in entry. On a
current Hermes install, core's bundled copilot-acp plugin wins (loaded
first from the repo's own plugins/model-providers/, which takes priority
over $HERMES_HOME by import order) — this file only matters as a fallback
or a migration target.
"""

from providers import register_provider
from providers.base import ProviderProfile

from agent.acp_agent_registry import ACPAgentEntry, register_acp_agent


class ACPSubprocessProfile(ProviderProfile):
    """GitHub Copilot ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return None


register_acp_agent("copilot", ACPAgentEntry(
    command="copilot",
    args=("--acp", "--stdio"),
    display_name="GitHub Copilot",
    install_hint=(
        "Install GitHub Copilot CLI (npm install -g @github/copilot) or "
        "set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH."
    ),
    legacy_command_env_vars=("HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH"),
    legacy_args_env_var="HERMES_COPILOT_ACP_ARGS",
))

copilot_acp = ACPSubprocessProfile(
    name="copilot-acp",
    aliases=("github-copilot-acp", "copilot-acp-agent"),
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    display_name="GitHub Copilot ACP",
    description="GitHub Copilot ACP (spawns copilot --acp --stdio; reuses GitHub Copilot CLI login)",
    env_vars=("COPILOT_ACP_BASE_URL",),
    base_url="acp://copilot",  # ACP internal scheme
    auth_type="external_process",
    fallback_models=("copilot-acp",),
    picker_group="copilot",
)

register_provider(copilot_acp)
