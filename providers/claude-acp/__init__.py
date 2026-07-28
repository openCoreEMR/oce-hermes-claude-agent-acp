"""Claude Code ACP provider profile + agent registration.

Drives Claude Code through its official ACP adapter
(@agentclientprotocol/claude-agent-acp) as an external subprocess. Routing
happens in Hermes core's agent/acp_client.py via the acp://claude base
URL; this plugin only supplies the launch command (register_acp_agent)
and the provider profile (register_provider) core's generic derivation
needs to surface claude-acp in PROVIDER_REGISTRY, the CLI --provider flag,
CANONICAL_PROVIDERS, and the setup wizard / /model pickers.

Verification status: E2E tested against a running Hermes gateway
(claude-agent-acp backend, real chat traffic) — see the README table.
"""

from providers import register_provider
from providers.base import ProviderProfile

from agent.acp_agent_registry import ACPAgentEntry, register_acp_agent

register_acp_agent("claude", ACPAgentEntry(
    command="claude-agent-acp",
    command_fallbacks=("claude-code-acp",),
    # The Claude Code bridge refuses to launch when it detects it is
    # running inside another Claude Code session (guard keyed off these
    # markers); strip them so Hermes can drive it from any environment.
    env_unset=("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT"),
    display_name="Claude Code",
    install_hint=(
        "Install the maintained adapter: "
        "npm install -g @agentclientprotocol/claude-agent-acp "
        "(the older @zed-industries/claude-code-acp bin also works). "
        "Requires Claude Code CLI credentials or ANTHROPIC_API_KEY."
    ),
))


class ACPSubprocessProfile(ProviderProfile):
    """Claude Code ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return None


claude_acp = ACPSubprocessProfile(
    name="claude-acp",
    aliases=("claude-acp-agent",),
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    display_name="Claude Code ACP",
    description="Claude Code ACP (spawns claude-agent-acp; reuses Claude Code login)",
    env_vars=("CLAUDE_ACP_BASE_URL",),  # credentials are managed by Claude Code's own login
    base_url="acp://claude",  # ACP internal scheme
    auth_type="external_process",
    fallback_models=("claude-acp",),
    # Folds under the built-in "Anthropic" picker row alongside the native
    # anthropic provider — see hermes_cli.models.PROVIDER_GROUPS.
    picker_group="anthropic",
)

register_provider(claude_acp)
