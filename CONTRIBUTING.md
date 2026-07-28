# Contributing an agent

Adding a new ACP agent is one directory, two function calls. No changes to
Hermes core.

1. **Confirm the agent has an official ACP adapter.** Hermes only drives
   vendor CLIs through documented ACP interfaces (see the compliance
   discussion in [hermes-agent#3660](https://github.com/NousResearch/hermes-agent/issues/3660)) —
   not raw subprocess scraping. If the vendor doesn't publish one, this isn't
   the place for it.

2. **Create `providers/<name>-acp/`:**

   ```python
   # providers/<name>-acp/__init__.py
   from providers import register_provider
   from providers.base import ProviderProfile

   from agent.acp_agent_registry import ACPAgentEntry, register_acp_agent

   register_acp_agent("<name>", ACPAgentEntry(
       command="<the-agent's-acp-binary>",
       args=(),                       # default CLI args, if any
       command_fallbacks=(),          # alternate bin names to try, in order
       env_unset=(),                  # env vars to strip from the child (e.g. session markers)
       display_name="<Human Name>",
       install_hint="npm install -g <package> ...",
   ))


   class ACPSubprocessProfile(ProviderProfile):
       def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0):
           return None  # ACP subprocess picks its own model


   <name>_acp = ACPSubprocessProfile(
       name="<name>-acp",
       api_mode="chat_completions",
       display_name="<Human Name> ACP",
       description="<Human Name> ACP (spawns <binary>; reuses <vendor> login)",
       env_vars=("<NAME>_ACP_BASE_URL",),
       base_url="acp://<name>",
       auth_type="external_process",
       fallback_models=("<name>-acp",),
       picker_group="",  # optional: fold under an existing hermes_cli.models.PROVIDER_GROUPS id
   )

   register_provider(<name>_acp)
   ```

   Plus `providers/<name>-acp/plugin.yaml`:

   ```yaml
   name: <name>-acp-provider
   kind: model-provider
   version: 1.0.0
   description: <Human Name> via ACP subprocess
   author: your-github-handle
   ```

3. **Test it against a real Hermes checkout** (this repo has no runtime
   dependency of its own — it only makes sense loaded inside Hermes):

   ```bash
   git clone https://github.com/NousResearch/hermes-agent
   cd hermes-agent && uv sync
   HERMES_HOME=/tmp/acp-test uv run python -c "
   import sys; sys.path.insert(0, '/path/to/hermes-acp-agents/providers/<name>-acp')
   import __init__ as plugin
   from agent.acp_agent_registry import resolve_agent_launch
   print(resolve_agent_launch('<name>'))
   "
   ```

   Or symlink the plugin into `$HERMES_HOME/plugins/model-providers/` via
   `./install.sh --link` and run `hermes --provider <name>-acp -z "hi"` for
   real end-to-end verification — that's the bar the README's status column
   tracks. Prefer a real run against the live adapter over a mocked one; the
   `claude-acp` entry's verification (real gateway traffic, MCP tool calls)
   is the reference.

4. **Open a PR** adding the directory and updating the README's table (move
   your agent out of "Wanted" and into "Supported agents" with an honest
   status — "verified E2E" only if you actually ran it against the live
   adapter, not just imported it cleanly).

## Picker grouping

If the vendor already has a native (non-ACP) Hermes provider — e.g.
Anthropic's API-key provider, or OpenAI's Codex/API providers — set
`picker_group` to that existing `hermes_cli.models.PROVIDER_GROUPS` id (e.g.
`"anthropic"`, `"openai"`) so your agent folds under one vendor row in the
interactive pickers instead of cluttering the top-level list. Leave it empty
if there's no natural group; core creates one lazily from your profile's
`display_name`/`description` if the picker_group id doesn't exist yet.

## Removing HERMES_ACP_{NAME}_COMMAND overrides

Every agent, registered or not, can be launched with
`HERMES_ACP_{NAME}_COMMAND` / `HERMES_ACP_{NAME}_ARGS` env vars without a
plugin at all — that path already exists in core
(`agent/acp_agent_registry.py`). Only add a plugin here when you want the
agent to show up by default in Hermes's pickers with a proper display name,
install hint, and picker grouping.
