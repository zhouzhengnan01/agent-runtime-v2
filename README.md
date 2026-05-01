# JetLinks Agent Runtime v2

Stateless agent runtime driven by built-in JSON agent configs.

The v2 runtime is intentionally separate from `jetlinks-agent-runtime` v1. It treats the agent as stateless: each CLI or HTTP call supplies the full conversation context and runtime options, while server-side files are only artifact containers for uploads, workspaces, and outputs.

## Run

```bash
export LLM_BASE_URL="http://124.132.152.75:62091/v1"
# Optional if config/agents/<agent>.json already contains model.api_key
export LLM_API_KEY="..."
export LLM_MODEL="Qwen3.6-35B-A3B"

uvicorn app.main:app --reload --port 8010
```

Open:

```text
http://127.0.0.1:8010/static/workbench.html
```

## CLI

```bash
python -m app.cli list-agents
python -m app.cli show-agent artifact-generator
python -m app.cli run --agent artifact-generator --message "生成一个 JetLinks IoT 平台架构图"
python -m app.cli run --agent behavior-detector --message "人员翻越围栏进入禁区" --json
```

## Stateless Contract

- Agent configs are static inputs, not runtime state.
- Requests must include the messages needed for the current turn.
- The server does not recover hidden memory from previous turns.
- `thread_id` only scopes files under `.runtime/threads/<thread_id>`.
- To continue a conversation, the caller must send prior messages again.

## Runtime Architecture

The runtime has one common entrypoint, `AgentRuntime`, shared by HTTP, CLI, ACP WebSocket, and ACP stdio.
ACP switches agents by `agentName`; the runtime then loads the corresponding JSON config and applies the same routing rules.

```text
AgentRuntime
  ├─ WorkflowRouter
  │    └─ selects a registered workflow only when agent JSON + skill metadata match
  ├─ WorkflowRegistry
  │    └─ optional workflow plugins such as artifact_workflow and evidence_first_detection
  └─ ToolCallingAgentLoop
       └─ OpenAI tools -> LLM tool_calls -> ToolInvocationService -> tool result -> next LLM turn
```

The default runtime registers built-in workflow plugins through `WorkflowRegistry.builtin(...)`:

```text
artifact_workflow          -> ArtifactWorkflow
evidence_first_detection  -> ArtifactWorkflow with evidence-first behavior skill selection
```

Workflow plugins are optional. If a workflow name appears in agent JSON but is not registered in the current
`WorkflowRegistry`, the request falls back to `agent_loop` instead of entering a hardcoded branch.
This keeps deployment profiles flexible: a full runtime can enable specialized workflow plugins, while a lightweight
runtime can run only the generic tool-calling loop.

The generic agent loop is still capable of using tools and skills. It exposes only the tools/skills declared by the
current agent JSON, asks the configured model to produce OpenAI-compatible `tool_calls`, executes them through the
unified tool service, appends `role=tool` results to the conversation, and repeats up to
`runtime.max_tool_rounds`.

```text
LLM -> tool_calls -> ToolInvocationService -> role=tool result -> LLM
```

## Agent Config

Each agent is driven by `config/agents/*.json`. Put stable agent behavior and default model routing here:

- `model.model`, `model.base_url`, `model.api_key`, `model.temperature`, `model.max_tokens`
- `runtime.stateless`, `runtime.max_tool_rounds`, `runtime.max_retries`, `runtime.require_verification`
- `tools`: named MCP/manual tools exposed to the model in `agent_loop`
- `skills`: skill-backed tools exposed to the model and allowed for workflow skill selection
- `workflows`: optional workflow plugin names such as `agent_loop`, `artifact_workflow`, or `evidence_first_detection`
- `quality`
- `prompts.system`

Environment variables still work as runtime overrides:

- `LLM_MODEL` overrides `model.model`
- `LLM_BASE_URL` overrides `model.base_url`
- `LLM_API_KEY` overrides `model.api_key`

The model config resolution order is:

```text
runtime_options -> environment variables -> agent JSON
```

It is valid to put `api_key` directly in a private agent JSON file:

```json
{
  "model": {
    "provider": "openai_compatible",
    "model": "Qwen3.6-35B-A3B",
    "base_url": "http://124.132.152.75:62091/v1",
    "api_key": "your-key"
  }
}
```

For committed or shared configs, prefer omitting `model.api_key` and using `LLM_API_KEY` so secrets are not checked in.
If the OpenAI-compatible endpoint does not require authentication, `api_key` can be omitted.

For local private credentials, use an ignored override file next to the public agent config:

```text
config/agents/default.json        # public default config
config/agents/default.local.json  # private local overrides, ignored by git
```

The local file can be partial; it is deep-merged over the public agent JSON:

```json
{
  "model": {
    "api_key": "your-key"
  }
}
```

`config/agents/*.local.json` is ignored by `.gitignore`, so private model keys can stay on the machine without being
uploaded with the shared config. Agent detail payloads and `python -m app.cli show-agent` redact `model.api_key` as
`********`; the runtime still uses the real value from the merged config.

Typical workflow configuration:

```json
{
  "workflows": {
    "default": "agent_loop",
    "generation": "artifact_workflow",
    "vision_behavior": "evidence_first_detection"
  }
}
```

`default: agent_loop` gives a Hermes-like general agent loop. Specialized agents can set
`default: artifact_workflow` or `default: evidence_first_detection` when deterministic workflow behavior is preferred.

## Tools, Skills, And MCP

Skills and MCP/manual tools share one protocol-neutral tool layer:

```text
Skill manifests + config/mcp/tools.json
  -> ToolRegistry
  -> ToolInvocationService
  -> MCP tools/list and tools/call
  -> Agent tool-calling loop
```

Skill manifests from `config/skills/*.json` and plugin manifests are exposed as skill-backed tools through
`SkillToolProvider`. Custom MCP/manual tools live in:

```text
config/mcp/tools.json
```

The same `ToolInvocationService` is used by:

- MCP JSON-RPC `tools/list` and `tools/call`
- HTTP management endpoints under `/api/mcp/tools`
- the generic `ToolCallingAgentLoop`

The default agent also exposes a first batch of Hermes-inspired local tools through the same registry:

```text
local_read_file     -> read UTF-8 text from the current thread workspace
local_write_file    -> write UTF-8 text into the current thread workspace
local_search_text   -> search text files in the current thread workspace
local_todo          -> maintain a thread-scoped todo list
local_shell_command -> run shell commands in the thread workspace, disabled by default
```

These local tools are intentionally scoped to `.runtime/threads/<thread_id>/user-data/workspace` so model-driven file
operations do not get unrestricted host filesystem access. `local_shell_command` is only exposed when explicitly
enabled:

```bash
export LOCAL_SHELL_TOOL_ENABLED=true
```

Useful tool endpoints:

```text
GET  /api/mcp/tools
GET  /api/mcp/tools/{tool_name}
PUT  /api/mcp/tools/{tool_name}
POST /mcp
```

## Protocol Strategy

The runtime core emits typed events once, then transports adapt those events for different clients.

- Native Web Workbench: supports both HTTP POST plus SSE via `/api/agents/{agent}/runs/stream` and ACP-shaped WebSocket via `/api/acp/ws`.
- A2A adapter: reuse the same event stream and map runtime events to A2A streaming task/status/artifact messages.
- ACP adapter: keep the JSON-RPC lifecycle (`initialize`, `new_session`, `prompt`, `session/update`) and support WebSocket for browser clients.
- ACP stdio: run `python -m app.cli acp-stdio --agent default` for editor-style stdio integrations.
- WebSocket is the right transport for future bidirectional sessions, such as live cancellation, terminal input, approval prompts, or collaborative multi-agent control.

This keeps the stateless agent core transport-agnostic: SSE, ACP WebSocket, ACP stdio, and A2A all adapt the same typed runtime events.

## Sandbox Strategy

Sandbox execution is selective. Normal chat, LLM spec planning, and behavior-detection text checks stay in the local runtime. Only skills mapped to a sandbox profile are eligible for OpenSandbox execution.

Skill manifests live at:

```text
config/skills/*.json
```

Each manifest declares the platform-facing contract:

- `input_schema` for the structured spec accepted by the skill
- `output_schema` for expected artifacts or JSON results
- `quality_template` for verifier/planner hints
- `sandbox.enabled`, `sandbox.profile`, `sandbox.adapter_command`, and fallback behavior

Reusable sandbox runtime profiles live at:

```text
config/sandbox/profiles.json
```

The default mappings are:

```text
drawio-generation -> drawio
pptx-generation   -> office
excel-generation  -> office
```

Useful environment variables:

```bash
export SANDBOX_PROVIDER=local                 # local | opensandbox
export SANDBOX_PROFILE_CONFIG=config/sandbox/profiles.json
export SANDBOX_SKILLS=drawio-generation       # optional allowlist; omit to use all configured mappings
export SANDBOX_FALLBACK_TO_LOCAL=true
export SANDBOX_EXECUTOR_ENABLED=false         # default; set true only after OpenSandbox Server/images are ready
export OPENSANDBOX_DOMAIN=127.0.0.1:8080
export OPENSANDBOX_PROTOCOL=http
```

OpenSandbox skill images use a single adapter entrypoint:

```text
/opt/jetlinks/skills/run_skill.py --request /mnt/user-data/workspace/request.json --outputs /mnt/user-data/outputs
```

The runtime writes a `skill-run.v1` request JSON, runs the manifest/profile `adapter_command`,
then copies files from `/mnt/user-data/outputs` back into the thread artifact store.

The platform can read the skill contracts through:

```text
GET /api/skills
GET /api/skills/{skill_name}
```

Each workflow emits a `sandbox.policy` event after skill selection. This event records the resolved profile, whether the skill is eligible for sandbox execution, and why the current run uses local or sandbox execution.
