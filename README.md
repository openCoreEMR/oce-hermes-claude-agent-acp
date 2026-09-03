# hermes-acp-agents

Third-party ACP agent backends for [Hermes Agent](https://github.com/NousResearch/hermes-agent),
kept out of core per the [maintainer standalone-plugin
policy](https://github.com/NousResearch/hermes-agent/issues/5257#issuecomment-5088843393)
(the same policy that closed [#5258](https://github.com/NousResearch/hermes-agent/pull/5258)).

Since hermes-agent [#101456](https://github.com/NousResearch/hermes-agent/pull/101456)
(part 2 of the #74743 split), an external-process provider needs **zero core
edits**: each `providers/<name>-acp/` package here is a
`ProviderProfile` subclass that describes its launch through the
`process_command` / `process_args` / `process_command_env_vars` /
`process_args_env_var` fields and supplies its client through
`ProviderProfile.create_client()`. The generalized ACP client (originally
proposed for core in [#68222](https://github.com/NousResearch/hermes-agent/pull/68222),
closed in favour of this seam) is **vendored** into each package
(`acp_client.py` + `acp_agent_registry.py`) — line-delimited JSON-RPC over
stdio, `session/set_config_option` model selection with confirmed read-back,
tool-call bridging, and the `HERMES_SKIP_TRANSPORT_WRAP` /
`HERMES_SKIP_ASYNC_WRAP` flags the auxiliary client (compression, vision)
honours.

Requires a Hermes build that includes #101456 (≥ 2026-09-02 main). For the
history: issue [#5257](https://github.com/NousResearch/hermes-agent/issues/5257)
was the original generalized-ACP-client proposal, PR #68222 its core
implementation, and the maintainer's closing note on #68222 the invitation
this re-ship answers.

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
