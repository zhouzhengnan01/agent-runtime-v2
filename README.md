# JetLinks Agent Runtime v2

JetLinks Agent Runtime v2 是一套轻量、无状态的智能体运行时。它以 `config/agents/*.json`
里的智能体配置为入口，把模型调用、工具调用、Skill 执行、Workflow 插件、ACP/MCP 协议和
文件产物管理串成一个统一的运行闭环。

v2 与原 `jetlinks-agent-runtime` v1 是不同架构：v2 默认不保存隐藏对话记忆，每次 CLI 或 HTTP
请求都需要带上当前轮所需的完整上下文；服务端文件主要用于上传文件、线程工作区、输出产物和可选
长期记忆。

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
python -m app.cli run --agent artifact-generator --workflow artifact_workflow --message "生成一个 JetLinks IoT 平台架构图"
python -m app.cli run --agent behavior-detector --workflow evidence_first_detection --message "人员翻越围栏进入禁区" --json
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
- Workbench 应用中心：通过预置模板一键套用 agent、workflow、Skill、MCP 工具和示例提示词。
- 轻量 Cron：支持配置定时运行 agent 任务、预览触发时间、手动运行和查看运行历史。
- 线程级文件工作区：每个 `thread_id` 有自己的 workspace、uploads、outputs。
- 可选本地 Memory v1：通过 agent JSON 开关控制，提供记忆写入、搜索、删除和可选上下文注入。
- 本地私密配置：支持 `config/agents/*.local.json` 覆盖公开 agent JSON，并默认忽略上传。

当前还没有完整实现这些 Hermes/OpenCode 级能力：

- 多 provider 适配：目前主要是 OpenAI-compatible API，还没有 Anthropic、Gemini、Bedrock、OpenRouter 等独立 adapter。
- 完整 Memory 系统：目前只有本地 JSON Memory v1，还没有向量检索、自动总结、权限隔离和外部存储后端。
- 完整 Session Resume：`thread_id` 只管理文件和产物，不恢复隐藏历史对话。
- Gateway/Telegram/Discord/Slack/Email 等多平台常驻接入；Cron 已有轻量配置和内置调度器。
- 子智能体 `delegate_task` 协作。
- 浏览器自动化和网页抓取工具。
- 上下文压缩、fallback model、stream 健康检查、复杂断流恢复等长任务增强能力。

这些能力可以继续参考 `/Users/chenhao/Desktop/code/opencode/hermes-agent` 分批接入，但不建议一次性整包搬入。

## 无状态约定

- Agent 配置是静态输入，不是运行时状态。
- 请求必须包含当前轮需要的 messages。
- 服务端不会从隐藏历史中恢复完整对话；只有 agent JSON 显式开启的 Memory 会参与运行。
- `thread_id` 只用于隔离 `.runtime/threads/<thread_id>` 下的文件。
- 如果要继续一段对话，调用方需要把前文 messages 再次传入。
- Memory 是长期偏好和事实记录，不等同于 chat history 或 session resume。

## 运行时架构

所有入口都共享同一个 `AgentRuntime`：

```text
HTTP / CLI / ACP WebSocket / ACP stdio
  -> AgentRuntime
     ├─ 默认：ToolCallingAgentLoop
     │    └─ LLM tools -> tool_calls -> ToolInvocationService -> role=tool result -> 下一轮 LLM
     └─ 显式 workflow：WorkflowRegistry.get(runtime_options.workflow)
          └─ artifact_workflow、evidence_first_detection 等可选插件
```

默认运行时通过 `WorkflowRegistry.builtin(...)` 注册两个内置 workflow：

```text
artifact_workflow          -> ArtifactWorkflow
evidence_first_detection  -> 带 evidence-first 行为检测策略的 ArtifactWorkflow
```

Workflow 是插件，不是硬编码主流程。默认请求不会根据用户文本、Skill 简介或 agent JSON 自动切换
workflow；调用方显式传入 `runtime_options.workflow` 时，`AgentRuntime` 会从 `WorkflowRegistry`
中取出对应插件执行。如果显式指定的 workflow 没有注册，请求会直接失败并返回清晰错误，避免调用方
误以为已经进入某个插件流程。

另外，页面或协议层如果显式传入 `runtime_options.selected_skills`，运行时会把它视为一次用户确认的
能力选择：generation Skill 会进入 `artifact_workflow`，非 generation Skill 会进入
`evidence_first_detection`，并继续走 Spec 构建、Sandbox 策略、Skill 执行、Verifier 和可选 retry
闭环，而不是绕过 workflow 直接执行 Skill。`runtime_options.workflow` 的优先级仍然最高。

通用 agent loop 只会向模型暴露当前 agent JSON 声明的 `tools` 和 `skills`。模型返回
OpenAI-compatible `tool_calls` 后，运行时通过统一工具服务执行工具，把工具结果以 `role=tool`
消息追加回对话，并最多循环 `runtime.max_tool_rounds` 轮。

为避免文件读取、搜索、Skill 输出等大结果撑爆模型上下文，工具结果回填给模型前会按
`runtime.max_tool_result_chars` 裁剪（默认 20000 字符）。裁剪只影响进入下一轮 LLM 的
`role=tool` JSON；运行时事件里的 `structured_content` 仍保留工具执行层返回的结构化信息，便于审计和调试。

```text
LLM -> tool_calls -> ToolInvocationService -> role=tool result -> LLM
```

## Agent 配置

每个智能体由 `config/agents/*.json` 驱动。稳定的智能体行为建议放在这里：

- `model.model`、`model.base_url`、`model.api_key`、`model.api_key_enc`、`model.tool_choice`、`model.temperature`、`model.max_tokens`、`model.request_timeout_seconds`
- `runtime.stateless`、`runtime.max_tool_rounds`、`runtime.max_tool_result_chars`、`runtime.max_retries`、`runtime.require_verification`
- `tools`：暴露给 `agent_loop` 的 MCP/manual/local 工具
- `skills`：暴露给模型的 Skill-backed tools，同时也用于 Workflow Skill 选择
- `workflows`：可选 Workflow 插件名声明，例如 `agent_loop`、`artifact_workflow`、`evidence_first_detection`。
  该字段不再驱动默认文本自动路由，默认主流程始终是 `agent_loop`；但显式 `selected_skills` 会使用
  `generation` / `vision_behavior` 映射，未配置时分别回落到内置 `artifact_workflow` /
  `evidence_first_detection`。
- `memory`：是否启用长期记忆、记忆作用域、最大注入条数和是否注入系统提示
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
- `LLM_REQUEST_TIMEOUT_SECONDS` 覆盖模型请求超时时间，默认 120 秒，取值会限制在 1 到 3600 秒之间

`model.tool_choice` 用于控制是否向 OpenAI-compatible API 发送 `tools` 和 `tool_choice=auto`：

```json
{
  "model": {
    "tool_choice": "none"
  }
}
```

- `auto`：默认值，模型服务需要支持 tool calling。
- `none`：不向模型发送 tools，适合当前只支持普通聊天或未开启 tool parser 的模型服务。

如果模型服务返回：

```text
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

说明 provider 还没有开启自动工具调用解析，需要把当前 agent 的 `model.tool_choice` 设为 `none`，或者在模型服务侧开启 tool parser。当前 `default` agent 已设为 `none`。

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

如果配置会提交或共享，建议不要把真实 key 放进公开 JSON，而是使用本地覆盖文件。运行时也支持
`api_key_enc`，可以把 key 加密后写入本地覆盖文件。

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

也可以使用加密写入命令：

```bash
.venv/bin/python -m app.cli secrets set-api-key --agent default --value "your-key"
# 或者
uv run python -m app.cli secrets set-api-key --agent default --value "your-key"
```

命令会写入：

```text
config/agents/default.local.json
.runtime/secrets/master.key
```

`default.local.json` 中保存的是 `model.api_key_enc`：

```json
{
  "model": {
    "api_key_enc": "enc.fernet.v1...."
  }
}
```

运行时加载 agent 配置时会自动解密成 `model.api_key`，之后再调用 OpenAI-compatible API。

加密使用 `cryptography.Fernet`。解密主密钥来源优先级：

```text
JETLINKS_AGENT_SECRET_KEY -> .runtime/secrets/master.key
```

`config/agents/*.local.json` 和 `.runtime/` 都已加入 `.gitignore`，不会上传到 GitHub。HTTP agent
详情接口和 `python -m app.cli show-agent` 会把 `model.api_key`、`model.api_key_enc` 脱敏为
`********`，运行时内部仍然使用真实值。

注意：如果把 `api_key_enc` 提交到 GitHub，就必须确保 `JETLINKS_AGENT_SECRET_KEY` 或
`.runtime/secrets/master.key` 没有一起泄漏；加密串和主密钥同时泄漏就等同于明文泄漏。

启用 `api_key_enc` 后，运行服务的 Python 环境必须安装 `cryptography`。建议使用项目虚拟环境或 uv 启动：

```bash
.venv/bin/python -m uvicorn app.main:app --reload --port 8010
# 或者
uv run uvicorn app.main:app --reload --port 8010
```

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

`default: agent_loop` 表示默认走通用 Hermes-like 工具调用循环。`generation` 和
`vision_behavior` 只是声明可选插件名，便于页面、CLI、ACP 或后续推荐器明确选择。

真正触发 workflow 的方式有两种。

第一种是直接指定 workflow：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "可以帮我画一个原型图吗"
    }
  ],
  "runtime_options": {
    "thread_id": "local-demo",
    "workflow": "artifact_workflow"
  }
}
```

不传 `runtime_options.workflow` 时，即使用户文本里出现“原型图 / drawio / ppt / 摔倒 / 翻越”等词，
运行时也不会仅凭文本自动进入 workflow，而是继续走主 `agent_loop`。如果模型服务支持 tool calling，
模型可以在 agent loop 内根据已暴露的 Skill、MCP 和本地工具自主调用能力；如果模型服务暂时不支持
tool calling，可以通过页面能力按钮、CLI `--workflow` 或 ACP `runtimeOptions.workflow` 显式启用
确定性 workflow。

第二种是本轮显式选择 Skill：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "生成一份项目汇报材料"
    }
  ],
  "runtime_options": {
    "thread_id": "local-demo",
    "selected_skills": ["pptx-generation"]
  }
}
```

这种情况下，运行时会使用该 Skill 对应的 workflow 闭环执行。这样页面上的“选择 Skill”按钮可以做到
所选即所用，同时不会绕过校验、Sandbox 和 retry 逻辑。

## 应用中心 / 智能体模板

如果不希望每次在页面里反复选择 agent、Workflow、Skill 和 MCP 工具，可以把常用组合沉淀成
Workbench 应用模板。模板配置保存在：

```text
config/apps/templates.json
```

当前内置了通用助手、IoT 架构图、项目汇报 PPT、Excel 模板、技术文档、XMind 任务拆解、行为识别
安全助手和产物生成全家桶等模板。每个模板可声明：

```json
{
  "name": "iot-architecture-diagram",
  "title": "IoT 架构图专家",
  "category": "generation",
  "agent_name": "artifact-generator",
  "workflow": "artifact_workflow",
  "selected_skills": ["drawio-generation"],
  "selected_mcp_tools": [],
  "prompt_examples": ["生成一份 JetLinks IoT 平台架构 Draw.io 图"],
  "tags": ["Draw.io", "架构图"]
}
```

接口：

```text
GET /api/apps/templates
GET /api/apps/templates/{template_name}
```

Workbench 左侧新增 **应用中心** 页面，也可以从输入框工具栏点 **应用** 快捷进入。点击“使用此应用”后，
页面会自动切换到对应 agent，写入本轮 `runtime_options.workflow`、`selected_skills`、
`selected_mcp_tools`，并填入第一个示例提示词；之后直接发送即可使用已经配置好的智能体。模板目前是
轻量 JSON 配置，不会修改 agent 源配置，适合把常用场景预置成可点击入口。

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

管理类写接口可以通过环境变量开启轻量鉴权：

```bash
export RUNTIME_API_TOKEN="your-admin-token"
```

设置后，下列接口需要请求头 `Authorization: Bearer <token>`：

```text
PUT  /api/mcp/tools/{tool_name}
POST /api/skills/plugins
PUT  /api/skills/{skill_name}
PUT  /api/skills/{skill_name}/files/{file_id}
```

未设置 `RUNTIME_API_TOKEN` 时保持本地开发兼容，不强制鉴权。

运行类接口默认保留开发模式的无鉴权行为；生产部署可显式启用 runtime 鉴权：

```bash
export RUNTIME_AUTH_MODE=production
export RUNTIME_API_TOKEN="your-runtime-token"
```

启用后，`POST /api/agents/{agent_name}/runs`、`POST /api/agents/{agent_name}/runs/stream`
和 `GET /api/acp/ws` 需要携带 `Authorization: Bearer <token>`，WebSocket 也支持
`/api/acp/ws?token=<token>` 便于浏览器客户端接入。


### Cron 定时任务

运行时内置轻量 Cron 任务配置，任务保存在：

```text
config/cron/jobs.json
```

接口：

```text
GET    /api/cron/jobs
GET    /api/cron/jobs/{job_name}
GET    /api/cron/jobs/{job_name}/runs
POST   /api/cron/preview
PUT    /api/cron/jobs/{job_name}
DELETE /api/cron/jobs/{job_name}
POST   /api/cron/jobs/{job_name}/run
GET    /api/cron/scheduler/status
```

写入、删除和手动运行接口复用 `RUNTIME_API_TOKEN` 管理鉴权。Cron 表达式使用 5 字段格式：

```text
minute hour day-of-month month day-of-week
```

支持 `*`、`*/n`、范围、列表，以及英文月份/星期缩写。任务会按配置的 `timezone` 计算下一次运行时间，
并通过现有 `AgentRuntime` 执行指定 `agent_name`、`prompt` 和 `runtime_options`。
每次运行会追加记录到 `config/cron/runs/{job_name}.jsonl`，便于页面展示最近运行历史。

Workbench 左侧新增 **Cron 计划** 页面，可创建、编辑、启停、删除、立即运行任务，查看接下来 5 次触发预览
和最近运行历史。页面侧提供 Admin Token 输入框，保存后会自动为管理接口附加 `Authorization: Bearer <token>`。
内置调度器默认启用；
如需只保留配置管理和手动运行，可以设置：

```bash
export CRON_SCHEDULER_ENABLED=false
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
present_files       -> 列出当前线程 outputs 文件，可选列出 workspace 文件
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

### 运行时选择工具策略

`runtime_options.selected_mcp_tools` 用于“本轮临时把某些已注册工具暴露给 agent loop”。它不是任意工具
执行入口，只影响本轮发给模型的 OpenAI-compatible `tools` 列表：

- 只会暴露已经注册且 `enabled=true` 的工具；未知工具会被忽略。
- 当前允许临时选择 `manual`、`local`、`memory`、`skill` 来源的工具。
- 如果 agent 关闭了 `memory.enabled`，memory 工具仍不会暴露。
- `local_shell_command` 默认 disabled；只有 `LOCAL_SHELL_TOOL_ENABLED=true` 后才会进入可选列表。
- 当 agent JSON 的 `model.tool_choice` 为 `none`，但本轮选择了 MCP 工具时，运行时会把本轮
  `tool_choice` 临时切到 `auto`，避免“用户选了工具但模型完全看不到工具”的问题。

### 本地 Memory 工具

Memory v1 是一个轻量本地持久化能力，不是完整会话恢复系统。它用于保存跨请求仍然有价值的用户偏好、
项目事实和长期任务信息，存储位置为：

```text
.runtime/memory/<agent>.json
.runtime/memory/global.json
```

默认 agent 已开启 Memory，并声明了这些工具：

```text
memory_remember -> 写入一条长期记忆
memory_search   -> 搜索当前 agent 或 global 作用域下的记忆
memory_forget   -> 按 id 删除一条记忆
```

`memory_clear` 已在工具层实现，但默认 agent 不暴露，避免模型误清空全部记忆。

agent JSON 示例：

```json
{
  "tools": ["memory_remember", "memory_search", "memory_forget"],
  "memory": {
    "enabled": true,
    "scope": "agent",
    "max_items": 20,
    "inject_context": true,
    "markdown_enabled": true,
    "markdown_writable_scopes": ["session"],
    "markdown_max_chars": 12000
  }
}
```

字段说明：

- `enabled`：是否启用 memory 工具；关闭时即使 `tools` 声明了 memory 工具，也不会暴露给模型。
- `scope`：`agent` 表示每个智能体独立记忆；`global` 表示多个智能体共享 `.runtime/memory/global.json`。
- `max_items`：注入系统提示时最多带入多少条最近记忆。
- `inject_context`：是否在每次 agent loop 调用模型前，把最近记忆追加到系统提示中。
- `markdown_enabled`：是否允许该 agent 暴露 Markdown 文件夹记忆工具。
- `markdown_writable_scopes`：Markdown 记忆允许写入的 scope。默认建议只开放 `session`，避免模型误写项目级、
  用户级或全局长期记忆。
- `markdown_max_chars`：Markdown 记忆默认读取裁剪长度。

建议把 Memory 当成“长期事实和偏好”，不要把完整聊天记录塞进去。完整 Session Resume 后续应单独设计，
例如按 thread 保存 history、摘要压缩、工具结果裁剪和上下文恢复策略。

### Markdown 文件夹记忆

除 JSON Memory v1 外，运行时还提供一套人类可读的 Markdown 文件夹记忆系统。它适合保存会话摘要、
项目背景、决策记录、用户偏好说明和可人工编辑的长期知识。目录按 scope 分层，方便直接定位对应记忆：

```text
.runtime/memory-md/
  global/
  users/
    <user_id 或 agent-<agent_name>>/
      *.md
      projects/
        <project_id>/
          **/*.md
      sessions/
        <thread_id>/
          **/*.md
```

请求可通过 `runtime_options.user_id`、`runtime_options.project_id` 和 `runtime_options.thread_id` 控制隔离：

- `global`：全局 Markdown 记忆。
- `user`：用户级 Markdown 记忆；未传 `user_id` 时回落到 `agent-<agent_name>`。
- `project`：用户下的项目级 Markdown 记忆。
- `session`：用户下的会话级 Markdown 记忆，天然按 `thread_id` 隔离。

内置工具：

```text
memory_md_list     -> 列出某个 scope/目录下的 .md 记忆
memory_md_read     -> 读取 .md 记忆，支持 max_chars 裁剪
memory_md_append   -> 追加写入 .md 记忆，适合会话工作记录
memory_md_search   -> 跨 global/user/project/session 搜索 .md 记忆
memory_md_compress -> 把较大的 .md 记忆抽取压缩成 summary.md
```

安全边界：

- 只允许访问 `.md` 文件。
- 禁止 `../`、绝对路径、home-relative 路径等路径穿越。
- 所有路径都会被限制在当前 `scope` 对应目录内。
- 默认 agent 只允许写入 `session` 级 Markdown 记忆；`project`、`user`、`global` 建议在用户确认或管理员授权后
  再加入 `memory.markdown_writable_scopes`。
- `memory_md_compress` 是本地抽取式压缩，不依赖 LLM；它会保留标题、重要 bullet、包含关键词的段落、
  “决定 / 结论 / TODO / 问题 / 风险 / 偏好”等高价值行，并强制遵守 `max_chars`。

典型请求：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "把这次讨论追加到会话记忆"
    }
  ],
  "runtime_options": {
    "user_id": "u_chenhao",
    "project_id": "jetlinks-agent-runtime-v2",
    "thread_id": "memory-design",
    "selected_mcp_tools": ["memory_md_append", "memory_md_search", "memory_md_compress"]
  }
}
```

## 协议入口

运行时核心只产生一套 typed events，不同协议负责适配这些事件：

- Web Workbench：`/static/workbench.html`
- HTTP：`POST /api/agents/{agent}/runs`
- SSE：`POST /api/agents/{agent}/runs/stream`
- Apps：`GET /api/apps/templates`
- ACP WebSocket：`/api/acp/ws`
- ACP stdio：`python -m app.cli acp-stdio --agent default`
- MCP HTTP：`POST /mcp`

ACP WebSocket 支持通过 `agentName` 切换当前智能体，也支持通过 `runtimeOptions.workflow` 显式启用
workflow。ACP stdio 用于编辑器或本地 agent 客户端的 stdio 集成场景，同样默认进入 `agent_loop`。

agent backend 分两类：

```text
local:
  backend.type = local
  使用 JetLinks 自己的 ToolCallingAgentLoop 作为主智能体闭环。

external:
  backend.type = acp_stdio
  代理 Codex 或其他 ACP stdio agent。
```

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
- 本地 Memory v1 工具

后续可继续参考 Hermes 分批补齐：

1. 多 provider adapter 和统一 `NormalizedResponse`
2. Memory 增强和 Session Resume
3. Context 压缩（工具结果裁剪已具备基础版本）
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
