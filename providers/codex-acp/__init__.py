"""Codex CLI ACP provider profile + agent registration.

Drives Codex CLI through its official ACP adapter
(@zed-industries/codex-acp) as an external subprocess. Routing happens in
Hermes core's agent/acp_client.py via the acp://codex base URL; this
plugin only supplies the launch command (register_acp_agent) and the
provider profile (register_provider) core's generic derivation needs to
surface codex-acp in PROVIDER_REGISTRY, the CLI --provider flag,
CANONICAL_PROVIDERS, and the setup wizard / /model pickers.

Verification status: reaches session/new and authenticates against a real
Codex CLI login — not yet exercised end-to-end with tool calls. See the
README table; help welcome closing that gap.
"""

from providers import register_provider
from providers.base import ProviderProfile

from agent.acp_agent_registry import ACPAgentEntry, register_acp_agent

register_acp_agent("codex", ACPAgentEntry(
    command="codex-acp",
    display_name="Codex CLI",
    install_hint=(
        "Install the official adapter: "
        "npm install -g @zed-industries/codex-acp "
        "(requires Codex CLI login or OPENAI_API_KEY)."
    ),
))


class ACPSubprocessProfile(ProviderProfile):
    """Codex CLI ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return None


codex_acp = ACPSubprocessProfile(
    name="codex-acp",
    aliases=("codex-acp-agent",),
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    display_name="Codex CLI ACP",
    description="Codex CLI ACP (spawns codex-acp; reuses Codex CLI login)",
    env_vars=("CODEX_ACP_BASE_URL",),  # credentials are managed by Codex CLI's own login
    base_url="acp://codex",  # ACP internal scheme
    auth_type="external_process",
    fallback_models=("codex-acp",),
    # Folds under the built-in "OpenAI" picker row alongside openai-codex /
    # openai-api — see hermes_cli.models.PROVIDER_GROUPS.
    picker_group="openai",
)

register_provider(codex_acp)
