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
git clone https://github.com/openCoreEMR/oce-hermes-claude-agent-acp
cd oce-hermes-claude-agent-acp
./install.sh            # copies providers/*/ into $HERMES_HOME/plugins/model-providers/
./install.sh --link      # symlink instead, for tracking repo updates in place
```

Then install whichever agent's official adapter you want to use (see the
table below), and restart Hermes.

## Supported agents

This fork carries only `claude-acp`. It removed the Codex and Copilot providers, which it does not run. [Upstream](https://github.com/mvdbastos/hermes-acp-agents) still has them.

| Provider | Ships | Spawns | Requires | Status |
|----------|-------|--------|----------|--------|
| `claude-acp` | this repo | `claude-agent-acp` (falls back to `claude-code-acp`) | `npm i -g @agentclientprotocol/claude-agent-acp` + a Claude Code plan token or login | **E2E verified** — real gateway traffic and Hermes's approval guards; see [How `claude-acp` runs](#how-claude-acp-runs) |

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
hermes --provider claude-acp --model claude-sonnet-5-5 -z "your prompt"
```

Or set as default via `hermes setup` (agents appear nested under their
vendor's provider group — e.g. **Anthropic ▸ Claude Code ACP**), the gateway
`/model` picker, or manually:

```yaml
model:
  provider: "claude-acp"
  default: "claude-sonnet-5-5"
```

## How `claude-acp` runs

`claude-acp` differs from the other providers here. Each Hermes session gets one long-lived adapter and ACP session, held in a module-level registry, so they outlive Hermes's client rebuilds.

- **Prompts** carry only the messages Claude Code has not seen. Once a prompt is on the wire, its messages count as seen, and a retry of them fails instead of repeating the turn.
- **Errors** after a prompt is sent are never retried and never fall back to another provider. The error names Claude Code's transcript and quotes its last message.
- **Stopping** sends `session/cancel` from another thread, makes the pending call raise at once, and ends the processes the turn started. An adapter that still has not settled after 8 s is killed, and the next turn resumes with `session/load`.
- **Models** must match one of the adapter's option values exactly, including any `[1m]`. Under a plan login those are full ids; under an API key they are aliases such as `sonnet`.
- **Permissions** go through Hermes's own guards: `check_all_command_guards` for Bash, and `agent.file_safety` for Read, Grep, Glob, Write, and Edit. Claude Code's `ask` rules force those requests, the session reads no settings files, and only one-shot options are ever selected. A tool with no Hermes guard, such as an MCP tool, is refused.
- **Text between turns**, such as a background task's report, goes into the Hermes session as a note. If Hermes refuses the note, the next reply carries it.
- **Credentials:** the adapter gets `CLAUDE_CODE_OAUTH_TOKEN` from `/var/run/secrets/hermes-claude/CLAUDE_CODE_OAUTH_TOKEN` (or `OCE_CLAUDE_ACP_TOKEN_FILE`). `OCE_CLAUDE_ACP_USE_LOGIN=1` uses the machine's own Claude Code login instead. No other `ANTHROPIC_*` variable or cloud-provider switch reaches the adapter.

Set `compression.enabled: false` for a profile on this provider. Claude Code compacts its own session, and Hermes's compression would only rewrite history the session does not read. Keep the auxiliary routes on another provider. A call with no tools, as those are, runs in a one-off adapter so it cannot enter the session's conversation.

`OCE_CLAUDE_ACP_IDLE_SECONDS` (default 1800) and `OCE_CLAUDE_ACP_MAX_SESSIONS` (default 8) bound the adapters a gateway keeps.

## Testing `claude-acp`

```bash
HERMES_AGENT_PATH=/path/to/hermes-agent python -m pytest tests/
```

`tests/fake_acp_agent.py` is a fake ACP agent that can hang, crash, and request permissions.

`tests/conformance_probe.py` runs the real adapter and real Claude Code against a scripted stub of the Messages API, so it costs nothing and needs no credential. Rerun it for every bump of the adapter, Claude Code, Hermes, or the plugin.

`tests/gateway_probe.py` drives a running gateway and spends real model turns. It covers what needs a gateway: kanban cards, switching back to another provider, approval cards, and the watchdog.
