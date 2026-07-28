# hermes-acp-agents

Third-party ACP agent backends for [Hermes Agent](https://github.com/NousResearch/hermes-agent),
kept out of core per the [maintainer standalone-plugin
policy](https://github.com/NousResearch/hermes-agent/issues/5257#issuecomment-5088843393)
(the same policy that closed [#5258](https://github.com/NousResearch/hermes-agent/pull/5258)).

Hermes core ships the generic ACP client (`agent/acp_client.py`), the agent
registry mechanism (`agent/acp_agent_registry.py`), and one built-in agent
(`copilot`, predating this registry). Every other agent — including the ones
in this repo — is a plugin: a `providers/<name>-acp/__init__.py` that calls
`register_acp_agent()` (how to launch the agent's ACP adapter) and
`register_provider()` (the Hermes provider profile: display name, picker
grouping, fallback model hint). Core's auto-derivation (`hermes_cli/auth.py`,
`hermes_cli/models.py`, `hermes_cli/providers.py`) does the rest — no core
edits needed to add an agent.

See [NousResearch/hermes-agent#68222](https://github.com/NousResearch/hermes-agent/pull/68222)
for the core PR this repo depends on, and issue
[#5257](https://github.com/NousResearch/hermes-agent/issues/5257) for the
original generalized-ACP-client proposal.

## Install

```bash
git clone https://github.com/mvdbastos/hermes-acp-agents
cd hermes-acp-agents
./install.sh            # copies providers/*/ into $HERMES_HOME/plugins/model-providers/
./install.sh --link      # symlink instead, for tracking repo updates in place
```

Then install whichever agent's official adapter you want to use (see the
table below), and restart Hermes.

## Supported agents

| Provider | Ships | Spawns | Requires | Status |
|----------|-------|--------|----------|--------|
| `copilot-acp` | this repo (core also bundles it) | `copilot --acp --stdio` | GitHub Copilot CLI + `copilot login` | Maintained in Hermes core; copy here for older installs |
| `claude-acp` | this repo | `claude-agent-acp` (falls back to `claude-code-acp`) | `npm i -g @agentclientprotocol/claude-agent-acp` + Claude Code login (or `ANTHROPIC_API_KEY`) | **E2E verified** — real gateway traffic, MCP tool calls, approval-policy bridging |
| `codex-acp` | this repo | `codex-acp` | `npm i -g @zed-industries/codex-acp` + Codex CLI login (or `OPENAI_API_KEY`) | Reaches `session/new`, authenticates — not yet exercised with tool calls end-to-end |

## Wanted

Not yet in this repo — contributions welcome (see `CONTRIBUTING.md`):

- Gemini CLI (`gemini --experimental-acp`)
- Qwen Code (`qwen --experimental-acp`)
- Cursor, Kiro, KiloCode, OpenCode, Kimi, Cline, Amp, Droid, iFlow

`gemini-acp` and `qwen-acp` were dropped from the core PR because they'd
never been exercised against a live adapter — if you run one of these
day-to-day, a validated plugin (see the `claude-acp` entry for the bar) is
the fastest path to getting it supported.

## Usage

```bash
hermes --provider claude-acp --model claude-acp -z "your prompt"
```

Or set as default via `hermes setup` (agents appear nested under their
vendor's provider group — e.g. **Anthropic ▸ Claude Code ACP**), the gateway
`/model` picker, or manually:

```yaml
model:
  provider: "claude-acp"
  default: "claude-acp"
```

Trade-offs vs. native API providers, the `approvals.acp_mode` permission
bridge, and `HERMES_ACP_{NAME}_COMMAND`/`_ARGS` env overrides are documented
in Hermes core's [AI Providers guide](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/integrations/providers.md#acp-agent-backends-claude-code-and-more).

## License

MIT — same as Hermes Agent.
