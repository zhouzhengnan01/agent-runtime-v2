# JetLinks Agent Runtime v2

JetLinks Agent Runtime v2 是一套轻量、无状态的智能体运行时。它以 `config/agents/*.json`
里的智能体配置为入口，把模型调用、工具调用、Skill 执行、Workflow 插件、ACP/MCP 协议和
文件产物管理串成一个统一的运行闭环。

v2 与原 `jetlinks-agent-runtime` v1 是不同架构：v2 默认不保存隐藏对话记忆，每次 CLI 或 HTTP
请求都需要带上当前轮所需的完整上下文；服务端文件主要用于上传文件、线程工作区和输出产物。

## 快速启动

```bash
export LLM_BASE_URL="http://124.132.152.75:62091/v1"
# 如果 config/agents/<agent>.json 或 <agent>.local.json 已配置 model.api_key，这里可以不设置
export LLM_API_KEY="..."
export LLM_MODEL="Qwen3.6-35B-A3B"

uvicorn app.main:app --reload --port 8010
```

打开工作台：

```text
http://127.0.0.1:8010/static/workbench.html
```

## 命令行

```bash
python -m app.cli list-agents
python -m app.cli show-agent artifact-generator
python -m app.cli run --agent default --message "你能做什么"
python -m app.cli run --agent artifact-generator --message "生成一个 JetLinks IoT 平台架构图"
python -m app.cli run --agent behavior-detector --message "人员翻越围栏进入禁区" --json
python -m app.cli acp-stdio --agent default
```

## 当前能力边界

当前 `agent-v2` 已经具备轻量 agent runtime 的核心能力：

- 完整的基础 agent loop：模型输出 `tool_calls`，运行时执行工具，再把 `role=tool` 结果回填给模型继续推理。
- JSON 配置驱动：不同 agent 通过 `config/agents/*.json` 定义模型、工具、Skill、Workflow 和运行参数。
- 统一工具层：Skill、MCP/manual 工具、本地 workspace 工具都通过 `ToolRegistry` 和 `ToolInvocationService` 暴露。
- Skill 插件：支持 drawio、pptx、excel、xmind、markdown、deliverables、behavior-detection 等内置 Skill。
- Workflow 插件：`artifact_workflow`、`evidence_first_detection` 通过 `WorkflowRegistry` 注册，可选启用。
- 多协议入口：HTTP、SSE、CLI、ACP WebSocket、ACP stdio、MCP HTTP。
- 线程级文件工作区：每个 `thread_id` 有自己的 workspace、uploads、outputs。
- 本地私密配置：支持 `config/agents/*.local.json` 覆盖公开 agent JSON，并默认忽略上传。

当前还没有完整实现这些 Hermes/OpenCode 级能力：

- 多 provider 适配：目前主要是 OpenAI-compatible API，还没有 Anthropic、Gemini、Bedrock、OpenRouter 等独立 adapter。
- 长期 Memory：v2 默认无状态，不保存跨会话长期记忆。
- 完整 Session Resume：`thread_id` 只管理文件和产物，不恢复隐藏历史对话。
- Cron/Gateway/Telegram/Discord/Slack/Email 等多平台常驻接入。
- 子智能体 `delegate_task` 协作。
- 浏览器自动化和网页抓取工具。
- 上下文压缩、fallback model、stream 健康检查、复杂断流恢复等长任务增强能力。

这些能力可以继续参考 `/Users/chenhao/Desktop/code/opencode/hermes-agent` 分批接入，但不建议一次性整包搬入。

## 无状态约定

- Agent 配置是静态输入，不是运行时状态。
- 请求必须包含当前轮需要的 messages。
- 服务端不会从隐藏记忆中恢复历史对话。
- `thread_id` 只用于隔离 `.runtime/threads/<thread_id>` 下的文件。
- 如果要继续一段对话，调用方需要把前文 messages 再次传入。

## 运行时架构

所有入口都共享同一个 `AgentRuntime`：

```text
HTTP / CLI / ACP WebSocket / ACP stdio
  -> AgentRuntime
     ├─ WorkflowRouter
     │    └─ 根据 agent JSON、已注册 Workflow、Skill 元信息选择可用 workflow
     ├─ WorkflowRegistry
     │    └─ 注册可选 workflow 插件，例如 artifact_workflow、evidence_first_detection
     └─ ToolCallingAgentLoop
          └─ LLM tools -> tool_calls -> ToolInvocationService -> role=tool result -> 下一轮 LLM
```

默认运行时通过 `WorkflowRegistry.builtin(...)` 注册两个内置 workflow：

```text
artifact_workflow          -> ArtifactWorkflow
evidence_first_detection  -> 带 evidence-first 行为检测策略的 ArtifactWorkflow
```

Workflow 是插件，不是硬编码主流程。若 agent JSON 声明了某个 workflow，但当前 `WorkflowRegistry`
没有注册它，请求会回退到通用 `agent_loop`，而不是进入固定分支。

通用 agent loop 只会向模型暴露当前 agent JSON 声明的 `tools` 和 `skills`。模型返回
OpenAI-compatible `tool_calls` 后，运行时通过统一工具服务执行工具，把工具结果以 `role=tool`
消息追加回对话，并最多循环 `runtime.max_tool_rounds` 轮。

```text
LLM -> tool_calls -> ToolInvocationService -> role=tool result -> LLM
```

## Agent 配置

每个智能体由 `config/agents/*.json` 驱动。稳定的智能体行为建议放在这里：

- `model.model`、`model.base_url`、`model.api_key`、`model.temperature`、`model.max_tokens`
- `runtime.stateless`、`runtime.max_tool_rounds`、`runtime.max_retries`、`runtime.require_verification`
- `tools`：暴露给 `agent_loop` 的 MCP/manual/local 工具
- `skills`：暴露给模型的 Skill-backed tools，同时也用于 Workflow Skill 选择
- `workflows`：可选 Workflow 插件名，例如 `agent_loop`、`artifact_workflow`、`evidence_first_detection`
- `quality`
- `prompts.system`

运行时配置优先级如下：

```text
runtime_options -> 环境变量 -> agent JSON
```

环境变量仍然可以覆盖 agent JSON：

- `LLM_MODEL` 覆盖 `model.model`
- `LLM_BASE_URL` 覆盖 `model.base_url`
- `LLM_API_KEY` 覆盖 `model.api_key`

私有部署时可以直接把 `api_key` 放在本地 agent JSON 中：

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

如果配置会提交或共享，建议不要把真实 key 放进公开 JSON，而是使用本地覆盖文件。

## 本地私密配置

公开配置和本地私密配置可以分开：

```text
config/agents/default.json        # 公开默认配置，可以上传
config/agents/default.local.json  # 本地私密覆盖，默认被 git 忽略
```

`*.local.json` 可以只写需要覆盖的字段，加载时会深度合并到公开 agent JSON 上：

```json
{
  "model": {
    "api_key": "your-key"
  }
}
```

`config/agents/*.local.json` 已加入 `.gitignore`，不会上传到 GitHub。HTTP agent 详情接口和
`python -m app.cli show-agent` 会把 `model.api_key` 脱敏为 `********`，运行时内部仍然使用真实值。

如果你遇到：

```text
401 Unauthorized for /v1/chat/completions
```

通常说明模型服务要求鉴权，但当前运行时没有读到 `api_key`，或者 key 不正确。

## Workflow 配置

典型配置：

```json
{
  "workflows": {
    "default": "agent_loop",
    "generation": "artifact_workflow",
    "vision_behavior": "evidence_first_detection"
  }
}
```

`default: agent_loop` 表示默认走通用 Hermes-like 工具调用循环。更确定性的生成类智能体可以把
`default` 设置成 `artifact_workflow`；行为检测智能体可以设置成 `evidence_first_detection`。

## 工具、Skill 与 MCP

当前工具层是协议无关的：

```text
Skill manifests + config/mcp/tools.json + 内置 local tools
  -> ToolRegistry
  -> ToolInvocationService
  -> MCP tools/list 和 tools/call
  -> Agent tool-calling loop
```

同一个 `ToolInvocationService` 被这些入口复用：

- MCP JSON-RPC `tools/list` 和 `tools/call`
- HTTP 管理接口 `/api/mcp/tools`
- 通用 `ToolCallingAgentLoop`

### Skill 工具

`config/skills/*.json` 和插件 manifest 会通过 `SkillToolProvider` 暴露为工具。当前内置 Skill 包括：

```text
drawio-generation
pptx-generation
excel-generation
xmind-generation
markdown-rendering
deliverables-export
behavior-detection
```

### MCP/manual 工具

自定义 MCP/manual 工具配置在：

```text
config/mcp/tools.json
```

工具接口：

```text
GET  /api/mcp/tools
GET  /api/mcp/tools/{tool_name}
PUT  /api/mcp/tools/{tool_name}
POST /mcp
```

### 本地 Workspace 工具

参考 Hermes 的 file/search/todo/terminal 工具后，v2 当前先接入了一批低风险本地工具。它们都限制在
当前线程的 workspace 目录下：

```text
.runtime/threads/<thread_id>/user-data/workspace
```

默认 agent 已声明这些工具：

```text
local_read_file     -> 读取当前线程 workspace 内的 UTF-8 文本文件
local_write_file    -> 写入当前线程 workspace 内的 UTF-8 文本文件
local_search_text   -> 搜索当前线程 workspace 内的文本文件
local_todo          -> 维护当前线程的 todo 列表
local_shell_command -> 在当前线程 workspace 内执行 shell 命令，默认关闭
```

`local_shell_command` 默认不暴露，只有显式启用后才会出现在工具列表中：

```bash
export LOCAL_SHELL_TOOL_ENABLED=true
```

这样做是为了避免模型一上来就获得宿主机任意命令执行权限。后续如果要做更强能力，建议继续增加：

- 权限审批
- 命令 allowlist / blocklist
- 超时与输出裁剪策略
- 操作审计日志

## 协议入口

运行时核心只产生一套 typed events，不同协议负责适配这些事件：

- Web Workbench：`/static/workbench.html`
- HTTP：`POST /api/agents/{agent}/runs`
- SSE：`POST /api/agents/{agent}/runs/stream`
- ACP WebSocket：`/api/acp/ws`
- ACP stdio：`python -m app.cli acp-stdio --agent default`
- MCP HTTP：`POST /mcp`

ACP WebSocket 支持通过 `agentName` 切换当前智能体。ACP stdio 用于编辑器或本地 agent 客户端的 stdio
集成场景。

## Sandbox 策略

Sandbox 执行是选择性的。普通对话、LLM spec planning、行为检测文本判断默认在本地 runtime 中完成。
只有 Skill manifest 显式映射到 sandbox profile 时，才会进入 OpenSandbox 执行路径。

Skill manifest 位置：

```text
config/skills/*.json
```

每个 manifest 描述平台侧契约：

- `input_schema`：Skill 接收的结构化输入
- `output_schema`：预期输出、产物或 JSON 结果
- `quality_template`：给 planner/verifier 的质量提示
- `sandbox.enabled`、`sandbox.profile`、`sandbox.adapter_command`、fallback 策略

Sandbox profile 位置：

```text
config/sandbox/profiles.json
```

默认映射：

```text
drawio-generation -> drawio
pptx-generation   -> office
excel-generation  -> office
```

相关环境变量：

```bash
export SANDBOX_PROVIDER=local                 # local | opensandbox
export SANDBOX_PROFILE_CONFIG=config/sandbox/profiles.json
export SANDBOX_SKILLS=drawio-generation       # 可选 allowlist；不设置则使用所有配置映射
export SANDBOX_FALLBACK_TO_LOCAL=true
export SANDBOX_EXECUTOR_ENABLED=false         # 默认关闭；OpenSandbox Server/images 准备好后再设 true
export OPENSANDBOX_DOMAIN=127.0.0.1:8080
export OPENSANDBOX_PROTOCOL=http
```

OpenSandbox Skill 镜像使用统一 adapter 入口：

```text
/opt/jetlinks/skills/run_skill.py --request /mnt/user-data/workspace/request.json --outputs /mnt/user-data/outputs
```

运行时会写入 `skill-run.v1` 请求 JSON，执行 manifest/profile 中的 `adapter_command`，再把
`/mnt/user-data/outputs` 中的文件复制回当前 thread 的 artifact store。

平台可以读取 Skill 契约：

```text
GET /api/skills
GET /api/skills/{skill_name}
```

每个 workflow 在 Skill 选择后都会发出 `sandbox.policy` 事件，记录解析出的 profile、是否具备 sandbox
资格，以及本轮为何使用 local 或 sandbox 执行。

## 与 Hermes Agent 的参考关系

`/Users/chenhao/Desktop/code/opencode/hermes-agent` 可以作为后续增强参考，但当前 v2 不会直接整包照搬。
已经参考并接入的第一批能力是 workspace 级本地工具：

- 文件读写
- 文本搜索
- 线程 todo
- 默认关闭的 workspace shell

后续可继续参考 Hermes 分批补齐：

1. 多 provider adapter 和统一 `NormalizedResponse`
2. 长期 Memory 与 Session Resume
3. Context 压缩和工具结果裁剪
4. Browser/Web tools
5. Cron/Gateway 和多平台消息接入
6. 子智能体 delegation

优先建议继续保持“可选、插件化、agent JSON 可控”的方式接入，避免把通用助手的高权限能力直接写进主流程。

## 验证

当前分支提交前使用以下命令验证：

```bash
python -m ruff check app tests
python -m mypy app
python -m pytest -q
```
