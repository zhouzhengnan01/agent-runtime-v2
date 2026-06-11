# JetLinks Agent Runtime v2

JetLinks Agent Runtime v2 是一套轻量、无状态的智能体运行时。它以 `config/agents/*.json`
里的智能体配置为入口，把模型调用、工具调用、Skill 执行、Workflow 插件、ACP/MCP 协议和
文件产物管理串成一个统一的运行闭环。

v2 与原 `jetlinks-agent-runtime` v1 是不同架构：v2 默认不保存隐藏对话记忆，每次 CLI 或 HTTP
请求都需要带上当前轮所需的完整上下文；服务端文件主要用于上传文件、线程工作区、输出产物和可选
长期记忆。

## 快速启动

推荐使用仓库自带脚本启动，脚本会写入 PID、日志并等待 `/health` 正常：

```bash
./install-deps.sh
APP_PORT=18012 ./start.sh
APP_PORT=18012 ./status.sh
```

打开 Workbench：

```text
http://127.0.0.1:18012/workbench
```

### 模型配置与密钥

当前项目约定：Workbench 应用从 `config/apps/*.json` 的应用模板 `models` 读取模型连接配置，
包括 `model`、`base_url`、`api_key`、`tool_choice`、`temperature` 和 `max_tokens`。
也就是说，点击“使用此应用”后，前端会传 `app_template_name`，服务端再按该应用模板里的
`models[].api_key` 发起模型调用。

如果需要本地保存加密后的模型 key，可以把明文 `api_key` 改为 `api_key_enc` 并配合本地 secrets 主密钥。完整说明见
[`docs/secrets.md`](docs/secrets.md)；Git 中可见的目录说明见
[`config/secrets/README.md`](config/secrets/README.md)。真实主密钥会在运行时生成到
`.runtime/secrets/master.key`，该目录已被 `.gitignore` 忽略，不会提交到 GitHub。

也可以直接用 uvicorn 开发调试：

```bash
uvicorn app.main:app --reload --port 8010
```

```text
http://127.0.0.1:8010/workbench
```

### 本地 Docker 镜像启动

如果希望用容器启动当前工作区代码，可以先构建本地 Python 3.12 镜像：

```bash
cd /Users/chenhao/Desktop/code/jetlinks-official/jetlinks-agent-runtime-agent-v2

docker build \
  --build-arg BASE_IMAGE=docker.m.daocloud.io/library/python:3.12-slim \
  -t jetlinks-agent-runtime-v2:local-py312 .
```

启动容器（联调模式，默认会挂载当前源码目录，改代码后重启即可生效）：

```bash
APP_PORT=18013 \
APP_HOST=127.0.0.1 \
JETLINKS_AGENT_IMAGE=jetlinks-agent-runtime-v2:local-py312 \
JETLINKS_AGENT_PULL_IMAGE=false \
JETLINKS_AGENT_CONTAINER_NAME=jetlinks-agent-runtime-v2-local-image \
JETLINKS_AGENT_CONTAINER_PORT=8000 \
./up.sh --docker
```

如果不希望目标机器拉取或挂载源码，需要使用已经把 `app/config/plugins/static` 打进镜像的版本，并关闭源码挂载：

```bash
APP_PORT=18013 \
APP_HOST=127.0.0.1 \
JETLINKS_AGENT_IMAGE=jetlinks-agent-runtime-v2:local-py312 \
JETLINKS_AGENT_PULL_IMAGE=false \
JETLINKS_AGENT_MOUNT_CODE=false \
JETLINKS_AGENT_CONTAINER_NAME=jetlinks-agent-runtime-v2-image-only \
JETLINKS_AGENT_CONTAINER_PORT=8000 \
./up.sh --docker
```

远端仓库镜像同理，把 `JETLINKS_AGENT_IMAGE` 换成仓库地址即可；目标机器只需要有 `up.sh/runtime-env.sh/status.sh/stop.sh`
这几个启动脚本和 Docker，不需要完整源码目录。

如果希望把镜像里的内置代码、配置和 skill 文件同步到本地目录，再用本地目录挂载启动，可以先执行：

```bash
JETLINKS_AGENT_SYNC_TARGET_DIR=/opt/jetlinks-agent-runtime-v2 \
./sync-image-code.sh
```

`sync-image-code.sh` 可以单独拷到目标机器执行；如果没有显式传 `JETLINKS_AGENT_IMAGE`，脚本会根据当前机器架构自动选择镜像：

```text
x86_64/amd64   -> registry.cn-hangzhou.aliyuncs.com/koudaimao/jetlinks-agent-runtime-v2:stable-amd64
aarch64/arm64  -> registry.cn-hangzhou.aliyuncs.com/koudaimao/jetlinks-agent-runtime-v2:stable-arm64
```

如果需要指定完整镜像地址，也可以显式传：

```bash
./sync-image-code.sh \
  --image registry.cn-hangzhou.aliyuncs.com/koudaimao/jetlinks-agent-runtime-v2:stable-arm64 \
  --target /opt/jetlinks-agent-runtime-v2
```

如果目标目录为空，它会把镜像里的代码、配置、skill 和启动脚本都同步出来。
然后进入同步出来的目录，用默认挂载模式启动：

```bash
cd /opt/jetlinks-agent-runtime-v2

APP_PORT=18013 \
APP_HOST=127.0.0.1 \
JETLINKS_AGENT_IMAGE=jetlinks-agent-runtime-v2:local-py312 \
JETLINKS_AGENT_PULL_IMAGE=false \
JETLINKS_AGENT_MOUNT_CODE=true \
JETLINKS_AGENT_CONTAINER_NAME=jetlinks-agent-runtime-v2-local-code \
JETLINKS_AGENT_CONTAINER_PORT=8000 \
./up.sh --docker
```

`sync-image-code.sh` 默认从镜像内的 `/workspace/code/jetlinks-agent-runtime-agent-v2` 同步；如果镜像工作目录不同，可以设置
`JETLINKS_AGENT_SYNC_SOURCE_DIR` 覆盖。注意 Docker 的 bind mount 不会自动把镜像里的已有文件复制到宿主机目录；
必须先同步，再挂载运行。

启动后访问：

```bash
curl http://127.0.0.1:18013/health
```

正常返回：

```json
{"status":"ok","service":"jetlinks-agent-runtime-v2"}
```

默认情况下，`up.sh --docker` 会自动把当前项目真实路径挂载到容器内，并把容器工作目录设置为挂载目录：

```text
/Users/chenhao/code/jetlinks-official/jetlinks-agent-runtime-agent-v2
  -> /workspace/code/jetlinks-agent-runtime-agent-v2
```

这里的宿主机路径来自 `pwd -P`，所以即使从 `/Users/chenhao/Desktop/code/...` 进入项目，
实际挂载路径也可能显示为 `/Users/chenhao/code/...`。

可以用下面命令确认实际挂载：

```bash
docker inspect jetlinks-agent-runtime-v2-local-image \
  --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}'
```

如果设置了 `JETLINKS_AGENT_MOUNT_CODE=false`，上面的命令不会输出源码挂载，服务会直接使用镜像内置代码。

查看状态：

```bash
APP_PORT=18013 \
APP_HOST=127.0.0.1 \
JETLINKS_AGENT_IMAGE=jetlinks-agent-runtime-v2:local-py312 \
JETLINKS_AGENT_CONTAINER_NAME=jetlinks-agent-runtime-v2-local-image \
./status.sh --docker
```

停止容器：

```bash
JETLINKS_AGENT_IMAGE=jetlinks-agent-runtime-v2:local-py312 \
JETLINKS_AGENT_CONTAINER_NAME=jetlinks-agent-runtime-v2-local-image \
./stop.sh --docker
```

### 一键进程管理脚本

仓库根目录提供了面向本地部署、Java `ProcessBuilder` 或其他进程管理器调用的脚本：

```bash
./start.sh    # 后台启动，推荐入口
./up.sh       # 后台启动，兼容旧入口
./status.sh   # 输出机器可读状态
./stop.sh     # 停止服务
./restart.sh  # 重启服务，内部调用 stop + start
```

首次部署或 Python 依赖变化后先执行：

```bash
./install-deps.sh
```

`install-deps.sh` 会创建 `.venv` 并安装 `requirements.txt`。默认使用 `python3` 创建虚拟环境，
也可以指定：

```bash
BASE_PYTHON=/usr/bin/python3.12 VENV_DIR=.venv ./install-deps.sh
```

默认监听地址是 `0.0.0.0:8000`；本机访问 Workbench 为：

```text
http://127.0.0.1:8000/workbench
```

`/workbench` 直接返回静态 Workbench；同一页面也保留在 `/static/workbench.html`，便于排障和直接访问。

运行状态文件默认写入 `.runtime/server/`：

```text
.runtime/server/server.pid
.runtime/server/server.log
```

可通过环境变量覆盖启动参数，适合 Java 调用前注入：

```bash
APP_HOST=0.0.0.0 APP_PORT=8010 ./start.sh
APP_PORT=8010 ./status.sh
APP_PORT=8010 ./restart.sh
APP_PORT=8010 ./stop.sh
```

常用变量：

- `APP_HOST`：监听地址，默认 `0.0.0.0`
- `APP_PORT`：监听端口，默认 `8000`
- `APP_MODULE`：ASGI 应用，默认 `app.main:app`
- `APP_WORKERS`：uvicorn workers，默认 `auto`
- `APP_WORKERS_AUTO_MAX`：`APP_WORKERS=auto` 时的上限，默认 `4`
- `APP_LOG_LEVEL`：uvicorn 日志级别，默认 `info`
- `PYTHON_BIN`：Python 解释器，默认优先 `.venv/bin/python`
- `RUNTIME_DIR`：PID 和日志目录，默认 `.runtime/server`
- `PID_FILE`：PID 文件路径
- `LOG_FILE`：日志文件路径
- `APP_START_TIMEOUT_SECONDS`：启动健康检查超时，默认 `30`
- `APP_STOP_TIMEOUT_SECONDS`：停止等待超时，默认 `15`
- `BASE_PYTHON`：`install-deps.sh` 创建虚拟环境时使用的 Python，默认 `python3`
- `VENV_DIR`：虚拟环境目录，默认 `.venv`

`status.sh` 输出适合程序解析：

```text
status=running pid=12345 health=ok url=http://127.0.0.1:8000 log=.runtime/server/server.log
```

退出码约定：

- `0`：运行中且 `/health` 正常
- `2`：进程存在但健康检查失败
- `3`：未运行或 PID 文件过期

模型由 `config/apps/*.json` 里的应用模板 `models` 提供；
Workbench 选择应用模板后会把 `app_template_name` 传给运行时，由服务端读取模板模型配置。

如果 `up.sh` 发现当前 Python 环境缺少依赖，会返回退出码 `10` 并提示先执行：

```bash
./install-deps.sh
```

#### Java 调用示例

Java 侧可以直接通过 `ProcessBuilder` 调用这些脚本。下面示例展示首次安装依赖、启动、查询状态、
重启和停止：

```java
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public class AgentRuntimeManager {
    private final File workDir;

    public AgentRuntimeManager(String repoPath) {
        this.workDir = new File(repoPath);
    }

    public CommandResult installDeps() throws IOException, InterruptedException {
        return run("./install-deps.sh", Map.of(
            "BASE_PYTHON", "python3",
            "VENV_DIR", ".venv"
        ));
    }

    public CommandResult start() throws IOException, InterruptedException {
        return run("./start.sh", Map.of(
            "APP_HOST", "127.0.0.1",
            "APP_PORT", "8000"
        ));
    }

    public CommandResult status() throws IOException, InterruptedException {
        return run("./status.sh", Map.of(
            "APP_PORT", "8000"
        ));
    }

    public CommandResult restart() throws IOException, InterruptedException {
        return run("./restart.sh", Map.of(
            "APP_PORT", "8000"
        ));
    }

    public CommandResult stop() throws IOException, InterruptedException {
        return run("./stop.sh", Map.of(
            "APP_PORT", "8000"
        ));
    }

    private CommandResult run(String command, Map<String, String> env)
        throws IOException, InterruptedException {
        ProcessBuilder builder = new ProcessBuilder(command);
        builder.directory(workDir);
        builder.redirectErrorStream(true);
        Map<String, String> processEnv = builder.environment();
        processEnv.putAll(env);

        Process process = builder.start();
        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
            new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append('\n');
            }
        int exitCode = process.waitFor();
        return new CommandResult(exitCode, output.toString());
    }

    public static Map<String, String> parseStatus(String output) {
        Map<String, String> result = new HashMap<>();
        for (String part : output.trim().split("\\s+")) {
            int index = part.indexOf('=');
            if (index > 0) {
                result.put(part.substring(0, index), part.substring(index + 1));
            }
        }
        return result;
    }

    public record CommandResult(int exitCode, String output) {}
}
```

调用示例：

```java
AgentRuntimeManager manager = new AgentRuntimeManager(
    "/Users/chenhao/Desktop/code/jetlinks-official/jetlinks-agent-runtime-agent-v2-branch"
);

CommandResult install = manager.installDeps();
if (install.exitCode() != 0) {
    throw new IllegalStateException("install failed: " + install.output());
}

CommandResult start = manager.start();
if (start.exitCode() != 0) {
    throw new IllegalStateException("start failed: " + start.output());
}

CommandResult status = manager.status();
Map<String, String> fields = AgentRuntimeManager.parseStatus(status.output());
System.out.println(fields.get("status"));
System.out.println(fields.get("url"));
```

Java 侧建议逻辑：

1. 首次部署或升级后调用 `./install-deps.sh`
2. 调用 `./start.sh`
3. 调用 `./status.sh`，退出码为 `0` 且 `status=running` 才认为启动成功
4. 需要重载时调用 `./restart.sh`
5. 停止时调用 `./stop.sh`

## 命令行

```bash
python -m app.cli list-agents
python -m app.cli show-agent default
python -m app.cli run --agent default --message "你能做什么"
python -m app.cli run --agent default --workflow artifact_workflow --message "生成一个 JetLinks IoT 平台架构图"
python -m app.cli run --agent default --workflow evidence_first_detection --message "人员翻越围栏进入禁区" --json
python -m app.cli acp-stdio --agent default
```

## 当前能力边界

当前 `agent-v2` 已经具备轻量 agent runtime 的核心能力：

- 完整的基础 agent loop：模型输出 `tool_calls`，运行时执行工具，再把 `role=tool` 结果回填给模型继续推理。
- JSON 配置驱动：通用 agent 通过 `config/agents/default.json` 定义工具、Skill、运行行为和提示词；模型参数由
  `config/apps/*.json` 的 `runtime_options` 定义。
- 统一工具层：Skill、MCP/manual 工具、本地 workspace 工具都通过 `ToolRegistry` 和 `ToolInvocationService` 暴露。
- Skill 插件：支持 drawio、pptx、excel、xmind、markdown、deliverables、behavior-detection 等内置 Skill。
- Workflow 插件：`artifact_workflow`、`evidence_first_detection` 通过 `WorkflowRegistry` 注册，可选启用。
- 多协议入口：HTTP、SSE、CLI、ACP WebSocket、ACP stdio、MCP HTTP。
- Workbench 应用中心：通过预置模板一键套用 default agent、workflow、Skill、MCP 工具和示例提示词。
- 轻量 Cron：支持配置定时运行 agent 任务、预览触发时间、手动运行和查看运行历史。
- 线程级文件工作区：每个 `thread_id` 有自己的 workspace、uploads、outputs。
- 可选本地 Memory v1：通过 agent JSON 开关控制，提供记忆写入、搜索、删除和可选上下文注入。
- Hermes-like 上下文压缩：当请求或工具闭环消息过长时，可按 agent JSON 配置保留头尾消息，并把中间历史压缩成交接摘要。
- 轻量 `delegate_task`：可把聚焦子任务交给一次无工具子模型调用，父 agent 只接收子任务摘要。
- 本地私密配置：支持 `config/agents/*.local.json` 覆盖公开 agent JSON，并默认忽略上传。

当前还没有完整实现这些 Hermes/OpenCode 级能力：

- 多 provider 适配：目前主要是 OpenAI-compatible API，还没有 Anthropic、Gemini、Bedrock、OpenRouter 等独立 adapter。
- 完整 Memory 系统：目前只有本地 JSON Memory v1，还没有向量检索、自动总结、权限隔离和外部存储后端。
- 基础 Session Resume：Runtime 会按 `thread_id` 为 agent loop 和 workflow 保存并恢复会话消息；更完整的摘要压缩、权限隔离和外部存储后端仍待增强。
- Gateway/Telegram/Discord/Slack/Email 等多平台常驻接入；Cron 已有轻量配置和内置调度器。
- 完整 Hermes 式多层子智能体协作；当前只有轻量无工具 `delegate_task`，还没有子 agent 工具集、暂停、中断和观测面板。
- 浏览器自动化和网页抓取工具。
- fallback model、stream 健康检查、复杂断流恢复等长任务增强能力。

这些能力可以继续参考 `/Users/chenhao/Desktop/code/opencode/hermes-agent` 分批接入，但不建议一次性整包搬入。

## 会话状态约定

- Agent 配置是静态输入，不是运行时状态。
- `thread_id` 是会话 ID，用于隔离 `.runtime/threads/<thread_id>` 下的消息历史、工作区和产物。
- Runtime 会把合并后的对话保存到 `.runtime/threads/<thread_id>/memory/conversation.jsonl`，并生成可读 transcript：`.runtime/threads/<thread_id>/memory/conversation.md`。
- 如果调用方只传当前轮 messages，服务端会自动加载同一 `thread_id` 的已保存历史再进入模型。
- 如果调用方已经传入完整历史，服务端会做前后缀重叠检测，避免把同一批消息重复追加。
- 文件成果仍存储在 `.runtime/threads/<thread_id>/outputs/`，并通过同目录下的 `manifest.json` 建索引。
- Memory v1 仍用于长期偏好和事实记录，不等同于完整 chat history；会话历史由 thread 目录内的 conversation 文件负责。

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

`runtime_options.selected_skills` 只表示本轮选择或允许的 Skill，不会单独触发 workflow。需要确定性
workflow 闭环时，调用方必须显式传入 `runtime_options.workflow`，并可同时传入
`runtime_options.selected_skills` 约束 workflow 使用的 Skill。

通用 agent loop 只会向模型暴露当前 agent JSON 声明的 `tools` 和 `skills`。模型返回
OpenAI-compatible `tool_calls` 后，运行时通过统一工具服务执行工具，把工具结果以 `role=tool`
消息追加回对话，并最多循环 `runtime.max_tool_rounds` 轮。

为避免文件读取、搜索、Skill 输出等大结果撑爆模型上下文，工具结果回填给模型前会按
`runtime.max_tool_result_chars` 裁剪（默认 20000 字符）。如果 agent JSON 开启
`runtime.context_compression_enabled`，运行时还会在每次模型调用前检查当前对话 JSON 长度；超过
`runtime.context_max_chars` 时保留前 `runtime.context_keep_first_messages` 条和后
`runtime.context_keep_last_messages` 条消息，并把中间消息压缩成一条 reference-only 交接摘要。
裁剪只影响进入下一轮 LLM 的
`role=tool` JSON；运行时事件里的 `structured_content` 仍保留工具执行层返回的结构化信息，便于审计和调试。

```text
LLM -> tool_calls -> ToolInvocationService -> role=tool result -> LLM
```

## Agent 配置

通用智能体由 `config/agents/default.json` 驱动。稳定的智能体行为建议放在这里：

- `runtime.stateless`、`runtime.max_tool_rounds`、`runtime.max_tool_result_chars`、`runtime.context_compression_enabled`、
  `runtime.context_max_chars`、`runtime.context_keep_first_messages`、`runtime.context_keep_last_messages`、
  `runtime.max_retries`、`runtime.require_verification`
- `tools`：暴露给 `agent_loop` 的 MCP/manual/local 工具
- `skills`：暴露给模型的 Skill-backed tools，也可被显式 workflow 根据 `selected_skills` 使用
- `memory`：是否启用长期记忆、记忆作用域、最大注入条数和是否注入系统提示
- `quality`
- `prompts.system`

模型配置不要写在 `config/agents/default.json`。应用中心入口应把模型参数写在 `config/apps/*.json` 的
`models` 中，例如：

```json
{
  "model_tags": ["chat", "reasoning", "tool_call"],
  "models": [
    {
      "name": "gpt-5.5",
      "features": ["vision", "reasoning", "chat"],
      "priority_features": ["chat", "reasoning"],
      "provider": "openai-compatible",
      "model": "gpt-5.5",
      "base_url": "http://192.168.35.29:9100/api/llm/openai/v1/providers/<provider-id>/",
      "api_key": "your-model-key",
      "tool_choice": "auto",
      "default_model": "gpt-5.5",
      "temperature": 0.4,
      "max_tokens": 2048
    }
  ]
}
```

`model_tags` 是给应用中心、Java 模型绑定和前端筛选使用的模型能力标签，不会作为运行时调用参数传给模型，
也不会进入 Skill 的 `input_schema`。Skill manifest 也可以声明同名顶层字段，用来表达该 skill 需要哪类模型能力；
真正生效的模型连接参数由应用模板的 `models` 绑定。当前支持的标签：

```text
chat
reasoning
vision
embedding
tool_call
image_generation
video_generation
audio_generation
text_to_speech
speech_to_text
vision_segmentation
rerank
```

历史配置里的 `tool_calling` 会归一化为 `tool_call`；`vision_segmentation` 和 `rerank` 已加入允许列表。

运行时模型配置优先级如下：

```text
请求显式 runtime_options -> 应用模板 models
```

当前内置应用模板采用 `models[].api_key` 明文配置；Workbench 使用应用模板时，运行时会优先使用该
app model 中的 `model`、`base_url`、`api_key` 和采样参数。模型连接参数不从环境变量读取。

`tool_choice` 属于模型调用兼容性开关。需要按应用控制时，放在应用模板 `models` 中；不建议再写入默认 agent JSON。

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

说明 provider 还没有开启自动工具调用解析，需要在模型服务侧开启 tool parser，或者把对应模型入口切到不使用工具调用的模式。
如果要把同一套配置发布到公开仓库或交付给外部环境，再把 `api_key` 改成 `api_key_enc`；
当前本地应用模板以 `api_key` 为准。

## 本地私密配置

公开 agent 配置和本地私密配置可以分开：

```text
config/agents/default.json        # 公开默认配置，可以上传
config/agents/default.local.json  # 本地私密覆盖，默认被 git 忽略
```

`*.local.json` 可以只写需要覆盖的字段，加载时会深度合并到公开 agent JSON 上。历史版本允许在这里覆盖
`model.api_key` / `model.api_key_enc`；应用中心模型配置迁移到 `config/apps/*.json` 后，优先在应用模板
`models` 中配置模型参数。

> 这部分只保留核心流程。完整密钥目录、加解密主密钥、迁移和部署说明见
> [`docs/secrets.md`](docs/secrets.md)。

```json
{
  "runtime": {
    "max_tool_rounds": 8
  }
}
```

如果你仍在旧的 `*.local.json` 中临时写入明文 `model.api_key`，运行时第一次加载该 agent 时会自动把
明文改写为 `model.api_key_enc` 并删除 `model.api_key`。本次加载仍然会在内存中使用该 key；
后续加载会从 `api_key_enc` 自动解密。

这适合迁移场景：新机器上只需要临时写一次明文 local config，启动或执行一次 `show-agent`
后，本地文件会自动变成加密形态：

```json
{
  "model": {
    "model": "gpt-5.4-mini",
    "base_url": "https://rehdasu.cn/v1",
    "api_key": "your-key"
  }
}
```

加载后会自动写回为：

```json
{
  "model": {
    "model": "gpt-5.4-mini",
    "base_url": "https://rehdasu.cn/v1",
    "api_key_enc": "enc.fernet.v1...."
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

如果密钥是写在应用模板配置里，也就是 `config/apps/*.json` 的 `models[].api_key_enc`，不要使用
`set-api-key --agent ...` 生成。应用模板密钥绑定的是 app 名称和模型名称，使用：

```bash
uv run python -m app.cli secrets encrypt-app-api-key \
  --app "_debug-1c81a222b0fa9000" \
  --model "gpt-5.5" \
  --value "your-key"
```

命令会输出可直接放入 `models[].api_key_enc` 的密文。这里的 `--model` 必须和 app JSON 中运行时选中的
`models[].name` 一致；如果没有 `name`，再使用 `models[].model` 或 `models[].default_model`。运行时解密时会使用：

```text
app:<app-name>:model:<model-name>:api_key
```

`set-api-key --agent default` 生成的是 agent local override 密文，purpose 是：

```text
agent:default:model:api_key
```

两者不能混用。

加密使用 `cryptography.Fernet`。解密主密钥来源优先级：

```text
JETLINKS_AGENT_SECRET_KEY -> .runtime/secrets/master.key
```

### 加解密主密钥

`api_key_enc` 不是独立可解密的密文，必须配合加密时使用的主密钥才能还原。推荐生产、测试等固定部署环境显式配置
`JETLINKS_AGENT_SECRET_KEY`，这样重启、换目录或重新拉代码后仍然能解密历史 `api_key_enc`：

```bash
export JETLINKS_AGENT_SECRET_KEY="replace-with-a-long-random-secret"
uv run python -m app.cli secrets set-api-key --agent default --value "your-key"
uv run python -m app.cli show-agent default
```

`JETLINKS_AGENT_SECRET_KEY` 可以是 Fernet key，也可以是普通长随机字符串；如果是普通字符串，运行时会先通过
SHA-256 派生出 Fernet 可用密钥。只要字符串内容不变，已有 `api_key_enc` 就可以继续解密。

如果没有配置 `JETLINKS_AGENT_SECRET_KEY`，首次加密或首次加载明文 `model.api_key` 时会自动生成本地文件：

```text
.runtime/secrets/master.key
```

这个文件默认只适合单机本地开发使用。迁移到新机器时，如果需要沿用旧的 `api_key_enc`，必须同时迁移
`.runtime/secrets/master.key`，或者在新环境配置与旧环境一致的 `JETLINKS_AGENT_SECRET_KEY`。如果主密钥丢失或更换，
旧的 `api_key_enc` 无法解密，需要重新执行 `secrets set-api-key` 或临时写入明文 `model.api_key` 让运行时重新加密。

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

Workflow 是平台级插件，由 `WorkflowRegistry` 注册和发现，不写在 `config/agents/default.json` 里。
默认请求始终走通用 agent loop；真正触发 workflow 的方式是直接指定 workflow：

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

本轮也可以同时显式选择 Skill：

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
    "workflow": "artifact_workflow",
    "selected_skills": ["pptx-generation"]
  }
}
```

这种情况下，`artifact_workflow` 会优先使用 `pptx-generation`。如果只传 `selected_skills` 而不传
`workflow`，请求仍然走默认 `agent_loop`。

## 应用中心 / 智能体模板

如果不希望每次在页面里反复选择 agent、Workflow、Skill 和 MCP 工具，可以把常用组合沉淀成
Workbench 应用模板。应用中心不是写死在前端，Workbench 只调用后端 Apps 接口读取模板；模板配置保存在：

```text
config/apps/*.json
```

也兼容集合文件：

```text
config/apps/templates.json
```

当前推荐一个应用一个 JSON 文件，例如：

```text
config/apps/algorithm-dataset-curation.json
config/apps/algorithm-engineer-full-cycle.json
config/apps/algorithm-engineer-workbench.json
config/apps/algorithm-evaluation-deployment.json
config/apps/algorithm-research-benchmark.json
config/apps/algorithm-training-orchestration.json
config/apps/artifact-suite.json
config/apps/behavior-safety-detector.json
config/apps/data-auto-annotation.json
config/apps/excel-template-builder.json
config/apps/general-jetlinks-assistant.json
config/apps/iot-architecture-diagram.json
config/apps/mindmap-task-breakdown.json
config/apps/project-report-ppt.json
config/apps/technical-doc-writer.json
```

当前内置了通用助手、IoT 架构图、项目汇报 PPT、Excel 模板、技术文档、XMind 任务拆解、行为识别
安全助手、行为复判、数据自动标注、算法工程全流程和产物生成全家桶等模板。每个模板可声明：

```json
{
  "name": "iot-architecture-diagram",
  "title": "IoT 架构图专家",
  "category": "generation",
  "agent_name": "default",
  "workflow": "artifact_workflow",
  "selected_skills": ["drawio-generation"],
  "selected_mcp_tools": [],
  "prompt_examples": ["生成一份 JetLinks IoT 平台架构 Draw.io 图"],
  "tags": ["Draw.io", "架构图"],
  "runtime_options": {
    "skill_parameters": {
      "drawio-generation": {
        "style": "layered"
      }
    }
  }
}
```

字段含义：

| 字段 | 作用 |
| --- | --- |
| `name` | 模板唯一名，也是 `GET /api/apps/templates/{template_name}` 和 ACP `appTemplateName` 使用的标识。文件名通常与 `name` 一致。 |
| `title` / `description` | 应用中心卡片展示名称和说明。 |
| `category` | 应用中心筛选分类，目前前端按 `general`、`generation`、`vision` 等分类展示。 |
| `icon` | 卡片图标语义名，前端会映射成简短图标。未知值会回退到通用图标。 |
| `agent_name` | 点击“使用此应用”时切换到的 agent，例如 `default`。必须能在 `config/agents/*.json` 中找到。 |
| `workflow` | 默认工作流，例如 `artifact_workflow`、`evidence_first_detection`；为 `null` 时走 agent 默认链路。非空时必须对应 `config/workflows/*.json` 中的本地 Workflow 实体。 |
| `selected_skills` | 本应用默认启用的 Skill 名称列表，必须对应 `config/skills/*.json` 中的本地技能实体。 |
| `selected_mcp_tools` | 本应用默认启用的 MCP 工具名称列表，名称来自 `config/mcp/tools.json`。 |
| `prompt_examples` | 示例提示词。Workbench 点击“使用此应用”或“填入示例”时会填入第一个示例。 |
| `tags` | 卡片标签，只用于展示和快速识别能力。 |
| `models` | 应用模板绑定的模型列表，例如 `gpt-5.5` 的 `base_url`、能力标签、采样参数和 `api_key`。服务端运行时使用，对前端响应会脱敏；需要加密时可改用 `api_key_enc`。 |
| `runtime_options` | 更细的默认运行参数，例如 `skill_parameters`、额外 workflow 参数等。模型连接参数优先放在 `models`。请求显式传入的字段优先级更高。 |

接口：

```text
GET /api/apps/templates
GET /api/apps/templates/{template_name}
```

后端加载逻辑：

- `app/api/apps.py` 注册 `/api/apps/templates` 和 `/api/apps/templates/{template_name}`。
- `AppTemplateRegistry` 默认读取仓库根目录下的 `config/apps`。
- `config/apps/<name>.json` 会按单个模板读取。
- `config/apps/templates.json` 会按 `{ "templates": [...] }` 集合读取，主要用于兼容旧配置。
- 列表接口会按 `category`、`title`、`name` 排序。
- 无效 JSON、字段校验失败或类型错误的模板会被跳过；本地测试里会校验内置模板引用的 agent、workflow、Skill、MCP 工具是否存在。

Workbench 左侧新增 **应用中心** 页面，也可以从输入框工具栏点 **应用** 快捷进入。点击“使用此应用”后，
页面会自动切换到对应 agent，写入本轮 `runtime_options.workflow`、`selected_skills`、
`selected_mcp_tools`，并填入第一个示例提示词；之后直接发送即可使用已经配置好的智能体。模板目前是
轻量 JSON 配置，不会修改 agent 源配置，适合把常用场景预置成可点击入口。

运行时优先级：

1. Workbench 点击“使用此应用”：前端把模板里的 `agent_name`、`workflow`、`selected_skills`、
   `selected_mcp_tools` 和 `runtime_options` 带入本轮请求。
2. ACP WebSocket `session/new` 可以传 `appTemplateName`，服务端会把模板转换成 session 默认
   `runtimeOptions`。
3. 请求里显式传入的 `runtimeOptions` 优先级高于模板默认值；其中 `skill_parameters` 会按 Skill 名合并，
   请求字段覆盖模板字段。
4. 模板只是会话入口配置，不会修改 `config/agents/*.json`、Skill manifest 或 MCP 工具源配置。

新增一个应用模板时，按这个流程做：

1. 在 `config/apps/<name>.json` 新建模板，`name` 不要包含 `/` 或 `\`。
2. 确认 `agent_name`、`workflow`、`selected_skills`、`selected_mcp_tools` 引用已存在。
3. 运行 `GET /api/apps/templates` 或刷新 Workbench 应用中心确认卡片出现。
4. 如需给定时任务复用，在应用中心点“建 Cron”，页面会把模板转换成 Cron 草稿。

### 应用中心验收

应用中心模板上线后，建议用 live smoke matrix 做一次端到端验收，确认模板配置、Skill 调用、产物生成、
上传文件读取和右侧产物列表都能闭环。先启动服务，例如：

```bash
APP_PORT=18012 ./restart.sh
```

准备一张真实测试图片后运行：

```bash
python tools/app_smoke_matrix.py \
  --base-url http://127.0.0.1:18012 \
  --output /tmp/jetlinks-app-smoke-matrix.json \
  --report-markdown /tmp/jetlinks-app-smoke-matrix.md \
  --retries 1 \
  --min-templates 16 \
  --expect-config-apps
```

脚本会读取 `GET /api/apps/templates` 返回的所有模板，并按模板里的 `agent_name` 逐个调用
`/api/agents/{agent_name}/runs`。
只有强依赖图片的模板（当前主要是 `data-auto-annotation`）会先通过 `/api/uploads/{thread_id}` 上传测试图片；
行为识别和行为复判按 evidence-first 逻辑先验证“无视觉证据时不输出视觉确认结论”，不会在 smoke 中强行附加图片。
其中 `behavior-detection` 无视觉证据时允许只返回 `requires_input=true` 和补充证据提示，不强制生成
`behavior-detection.md/json`。
默认测试图片会自动写到 `/tmp/jetlinks-app-smoke-image.jpg`；如果需要使用真实业务图片，可以显式传
`--image /path/to/image.jpg`，显式指定的图片不存在时脚本会直接失败。
每个应用会记录：

- `status_completed`：本轮 run 是否完成。
- `artifacts_present`：声明了 workflow、Skill 或 MCP 工具的应用是否至少生成了一个产物。
- `expected_artifacts_present`：常见 Skill 的关键产物是否出现，例如 `annotations.coco.json`、
  `*.drawio`、`*.pptx`、`*.xlsx`、`*.xmind` 等。
- `artifact_contents_valid`：关键文件格式是否可读，例如 COCO JSON 结构、JSON 解析、PNG 文件头、
  Draw.io XML、PPTX/XLSX/XMind/DOCX zip 容器、Markdown/TXT 非空。
- `artifact_names`：生成的文件名，便于确认 COCO、PPT、Draw.io、XMind、Markdown 等产物是否出现。
- `missing_expected_artifact_patterns`：缺失的关键产物模式，非空时该模板会被判定失败。
- `artifact_content_errors`：内容级校验失败原因，非空时该模板会被判定失败。
- `continuation_check`：对需要验证连续处理的模板，会在首轮成功后用同一个 `thread_id` 再发一轮请求，
  检查智能体能否读取已有 outputs 并继续生成新产物，且新产物也必须通过内容级校验。当前覆盖
  `data-auto-annotation` 的 `annotations.coco.json -> coco-summary.md`，以及 Markdown 文档类应用的
  `result.md -> continuation-summary.md`。
- `tool_rounds` / `tool_call_count` / `mode`：用于确认复杂任务是否走了多轮工具调用。
- `attached_smoke_image`：用于确认只有需要图片的应用才上传测试图片。

只验证某几个模板时可以用：

```bash
python tools/app_smoke_matrix.py \
  --base-url http://127.0.0.1:18012 \
  --only data-auto-annotation,behavior-safety-detector
```

当前验收标准是：脚本最终输出的 `summary.failed` 为空；需要产物的模板必须能在 artifacts 接口看到产物；
有已知关键产物的 Skill 还必须匹配 `expected_artifact_patterns`；已生成产物必须通过内容级校验；
带连续处理检查的模板还必须通过 `continuation_check`。如果只想验证首轮应用运行，可以临时加
`--skip-continuation-check`。
`--retries` 只会重试超时、连接断开等临时错误，不会掩盖缺产物或内容校验失败；如果希望把
“重试后才成功”也视为不稳定失败，可以加 `--fail-on-retry`，报告会保留首次失败的 thread、错误和耗时；
`--report-markdown`
会额外输出一份人工可读报告，适合发版前或线上排障时留档；`--min-templates` 用来防止应用中心模板
意外为空或缺失时被误判为通过；`--expect-config-apps` 会要求远端 `/api/apps/templates` 至少包含本仓库
`config/apps/*.json` 中声明的全部应用名，防止数量没变但核心应用被换错或漏掉。
如果 Runtime 开启了 `RUNTIME_API_TOKEN`，可以传 `--token <token>`，也可以设置环境变量
`RUNTIME_API_TOKEN`；报告不会输出 token。`quality-gate.sh` 和 GitHub live smoke 会优先通过环境变量
传递 token，避免把 token 放进命令行参数。
如果某个模板失败，先打开输出 JSON 查看该模板的 `thread_id`，再检查对应目录：

```text
.runtime/threads/<thread_id>/uploads
.runtime/threads/<thread_id>/outputs
```

发版或部署前可以直接跑一键质量门禁：

```bash
./quality-gate.sh
```

它会先运行关键 pytest。默认 `RUN_SMOKE=auto`，只跑本地测试和静态/单元级检查；如果要把应用中心
live smoke 或 Workbench Playwright 页面流程也纳入门禁，显式打开对应开关：

```bash
RUN_SMOKE=true ./quality-gate.sh
RUN_UI_SMOKE=true RUN_SMOKE=true ./quality-gate.sh
BASE_URL=http://127.0.0.1:18012 RUN_SMOKE=true SMOKE_RETRIES=1 ./quality-gate.sh
SMOKE_MIN_TEMPLATES=16 ./quality-gate.sh
SMOKE_EXPECT_CONFIG_APPS=true ./quality-gate.sh
SMOKE_FAIL_ON_RETRY=true ./quality-gate.sh
SMOKE_ONLY=data-auto-annotation,technical-doc-writer ./quality-gate.sh
SMOKE_TOKEN=your-runtime-token ./quality-gate.sh
RUN_SMOKE=false ./quality-gate.sh
```

当 `RUN_SMOKE=true` 或 `RUN_UI_SMOKE=true` 且没有显式传 `BASE_URL` 时，`quality-gate.sh` 会自动在随机空闲端口
启动当前工作区代码的临时 Runtime，等待 `/health` 正常后再跑 smoke，结束时自动停止临时进程。这样可以避免
本机 `18012` 上残留的旧进程污染验证结果。只有显式传 `BASE_URL=...` 时，门禁才会连接已有服务或线上环境。
如果需要关闭自动启动，可以设置：

```bash
AUTO_START_RUNTIME=false BASE_URL=http://127.0.0.1:18012 RUN_SMOKE=true ./quality-gate.sh
```

`quality-gate.sh` 默认 `SMOKE_FAIL_ON_RETRY=true`。live smoke 中任何模板如果先超时/断连、再靠重试成功，
门禁仍会失败，用来提前暴露线上应用链路不稳定；临时排查时可以设置 `SMOKE_FAIL_ON_RETRY=false`。
设置 `SMOKE_ONLY` 时，门禁会自动把 `SMOKE_MIN_TEMPLATES` 调整为 only 列表数量；如果你显式设置
`SMOKE_MIN_TEMPLATES`，则以显式值为准。
如果要把浏览器页面也纳入发版前验证，可以设置 `RUN_UI_SMOKE=true`。该检查会用 Playwright 打开
Workbench，完整走一遍“应用中心选择 `data-auto-annotation` -> 查看 JSON -> 上传图片 -> 深度执行 ->
生成 `annotations.coco.json` -> 同一会话继续生成 `coco-summary.md`”，并断言请求体确实带有上传路径、
`selected_skills`、`autonomous` 和 `max_tool_rounds=12`。Runtime 可以由门禁自动临时启动，也可以通过
`BASE_URL` 指向已有环境。

仓库也提供了 GitHub Actions 工作流 `.github/workflows/quality-gate.yml`。PR 和主分支 push 会自动执行
`RUN_SMOKE=false ./quality-gate.sh`，覆盖关键 pytest、应用模板静态检查和 smoke matrix 单元测试。
live smoke 依赖实际运行中的 Runtime、模型和外部服务，仍建议在部署机或发版环境用 `RUN_SMOKE=true`
执行。

如果需要从 GitHub 手动验证某个线上环境，可以触发 `.github/workflows/live-smoke.yml`，输入
`base_url`、`retries`、`timeout`、`fail_on_retry`，也可以打开 `run_ui_smoke` 让 workflow 额外跑
Workbench Playwright 页面 smoke。`data-auto-annotation` 依赖外部 SAM3 服务，workflow 支持通过
`sam3_predict_url` 手动输入，或通过仓库 secret `SAM3_PREDICT_URL` 注入。该 workflow 会对指定 Runtime
执行完整应用中心 smoke，并上传：

```text
live-smoke-result.json
live-smoke-report.md
```

`live-smoke.yml` 也会每天定时执行一次。定时任务读取仓库 secret `LIVE_SMOKE_BASE_URL` 作为目标
Runtime 地址；如果没有配置该 secret，workflow 会明确跳过 live smoke，不会误报失败。live smoke
报告会同时写入 GitHub Actions Step Summary，并作为 artifact 上传。如果目标 Runtime 需要 token，
配置仓库 secret `LIVE_SMOKE_TOKEN`；如果自动标注需要固定 SAM3 代理，配置仓库 secret
`SAM3_PREDICT_URL`。workflow 配置了 30 分钟超时和 concurrency，避免巡检重叠执行；上传的 smoke
报告默认保留 14 天。

`data-auto-annotation` 依赖外部 SAM3 HTTP 服务。默认脚本会读取环境变量覆盖预测地址：

```bash
SAM3_PREDICT_URL="https://192-168-33-25-8093.proxy.jetlinks.cn/sam3/predict"
```

也兼容 `SAM3_URL`。如果这两个变量都没有设置，会回退到脚本内置默认地址。线上或 VPN/内网穿透地址变化时，
优先改环境变量，不需要修改 skill JSON 或重新发版。live smoke 中如果 `data-auto-annotation` 失败，
先看 `data-auto-annotation-stderr.txt` 里的 `sam3_connect_timeout`、`sam3_read_timeout`、
`sam3_http_error` 等错误类型，再确认 `SAM3_PREDICT_URL` 是否可从 Runtime 机器访问。

同一个会话的后续任务会优先看到当前 thread 下的 `uploads` 和 `outputs`。例如自动标注生成
`annotations.coco.json` 后，继续在同一会话里要求“基于已有 COCO 生成摘要”时，智能体应通过
`present_files` / `local_read_file` / `artifact_read` 读取当前会话产物，而不是再次要求用户上传。

### 深度执行与连续产物

Workbench 输入区的“深度执行”开关会把本轮请求切到 `autonomous` 模式，并传入：

```json
{
  "mode": "autonomous",
  "config_options": {
    "max_tool_rounds": 12
  }
}
```

这个模式适合需要多次调用工具或 Skill 的复杂任务，例如“上传图片 -> 自动标注 -> 校验 COCO ->
生成报告 -> 继续修改报告”。它不是无限后台任务，而是受控的单次 run 多轮工具循环：

- `plan`：只规划，最多 1 轮工具循环。
- `safe`：只保留低风险工具，最多 2 轮。
- `edit`：默认编辑模式，使用 agent 默认轮数。
- `autonomous`：允许更宽的工具集合；未覆盖轮数时最多提升到 16，显式 `max_tool_rounds` 可在 1 到 32
  之间覆盖基础轮数。
- `yolo`：按 `autonomous` 暴露工具并自动批准 ACP 权限请求；只有缺少图片、数据集、模型文件、参数等
  运行必需内容时才会返回 `input_required` 等待用户补充。

复杂任务如果依赖上传文件或上一步产物，必须保持同一个 `thread_id`。线程目录里的文件会被注入到用户消息上下文，
并可通过本地文件工具读取：

```text
/mnt/user-data/uploads/<file>
/mnt/user-data/outputs/<file>
```

如果视觉类 Skill 返回 `input_required`，表示当前 thread 没有可用图片或数据集；用户补充上传后，
继续用同一个会话发送下一轮即可，不需要创建新 thread。

Workbench 右侧“执行”面板会展示 agent loop 的细粒度运行事件，包含：

- `llm.request.started` / `llm.request.completed`：当前第几轮模型请求、可用工具数、模型是否选择工具。
- `tool.calls.started` / `tool.started` / `tool.completed` / `tool.failed`：当前正在执行哪个工具、耗时、是否产生产物或失败码。
- `context.compacted`：长上下文压缩后继续执行。

这些事件用于排查“页面看起来卡住”的情况：如果只停在 `run.started`，通常说明后端还没有进入模型请求；
如果停在某个 `tool.started`，优先检查对应 Skill、MCP 工具或外部服务；如果已经有 `artifact.created`
但页面没有显示产物，则检查 artifacts API 或前端刷新逻辑。

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

`config/skills/*.json` 是平台侧本地技能实体目录，也是应用模板、Java 配置和 Workbench 技能列表的主来源。
每个应用模板的 `selected_skills` 都必须能在这里找到同名实体文件。

`plugins/skills/<skill-id>` 是技能实现插件实体目录，负责提供真正可执行的技能包。正式技能在插件内也应有对应同名实现实体：

```text
plugins/skills/<skill-id>/plugin.json
plugins/skills/<skill-id>/skills/<skill-id>/manifest.json
```

技能包可以同时包含 `SKILL.md`、`runner.py`、`spec_builder.py`、`requirements.txt`、`sandbox.yml`、
`scripts/` 等执行资产。运行时会按同名 skill 把 `config/skills/<skill>.json` 的平台实体定义绑定到插件实现；
插件中存在但没有本地实体文件的 skill 不会进入正式技能列表。上传新插件时，运行时会自动把插件 manifest
materialize 到 `config/skills/<skill>.json`，再暴露给 Workbench 和 Java 侧。
历史套件目录可以保留为共享实现或兼容包，但正式技能应有自己的 `plugins/skills/<skill-id>` 顶层目录。

当前正式内置 Skill 包括：

```text
drawio-generation
pptx-generation
excel-generation
xmind-generation
markdown-rendering
deliverables-export
behavior-detection
data-auto-annotation
algorithm-engineer
algorithm-research-scout
cpu-training-runner
dataset-curator
model-candidate-selector
remote-gpu-ops
gpu-training-orchestrator
detector-evaluator
deployment-candidate-reviewer
experiment-ledger
```

### Workflow

`config/workflows/*.json` 是平台侧本地 Workflow 实体目录，也是应用模板、Java 配置和 Workbench
Workflow 列表的主来源。每个应用模板的 `workflow` 字段如果非空，都必须能在这里找到同名实体文件。

`plugins/workflows/**` 是 Workflow 实现插件目录，负责提供 `handler` 指向的 Python 实现。运行时会按同名
workflow 把 `config/workflows/<workflow>.json` 的实体定义绑定到插件实现；插件中存在但没有本地实体文件的
workflow 不会进入正式 Workflow 列表。上传新 Workflow 插件时，运行时会自动把插件 manifest materialize 到
`config/workflows/<workflow>.json`，再暴露给 Workbench 和 Java 侧。

当前正式内置 Workflow 包括：

```text
artifact_workflow
evidence_first_detection
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
delegate_task       -> 把聚焦子任务委派给一次无工具子模型调用
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

### 轻量 Delegation 工具

参考 Hermes `delegate_task` 的父子 agent 思路后，v2 当前先接入了低风险版本：

```text
delegate_task(goal, context?, agent_name?, model_name?, temperature?, max_tokens?, request_timeout_seconds?)
```

它会加载 `agent_name` 指定的 agent 配置；未指定时使用当前 agent。子任务只执行一次普通模型调用，不暴露工具、
不写共享记忆、不递归委派，返回内容会作为工具结果交回父 agent。适合做独立审查、拆分分析和第二视角总结。
这不是完整 Hermes 多层 subagent 系统；后续如果要增强，可继续加入子 agent 工具 allowlist、并发调度、
超时中断、运行观测和结果尾部摘录。

### 运行时选择工具策略

`runtime_options.selected_mcp_tools` 用于“本轮临时把某些已注册工具暴露给 agent loop”。它不是任意工具
执行入口，只影响本轮发给模型的 OpenAI-compatible `tools` 列表：

- 只会暴露已经注册且 `enabled=true` 的工具；未知工具会被忽略。
- 当前允许临时选择 `manual`、`local`、`delegate`、`memory`、`skill`、`mcp_streamable_http` 来源的工具。
- ACP `session/new.params.mcpServers` 传入的 `http`/`sse` server 会在本地 agent loop 中动态执行
  `initialize`、`tools/list`，并以 `{server}__{tool}` 名称暴露；调用时转发到远端 `tools/call`。
- ACP `stdio` server 也可动态挂载，但必须显式设置 `RUNTIME_MCP_STDIO_ENABLED=true`；可用
  `RUNTIME_MCP_STDIO_ALLOWED_COMMANDS` 限制允许拉起的命令。
- 运行时 MCP `tools/list` 结果默认缓存 60 秒，可通过 `RUNTIME_MCP_TOOLS_CACHE_SECONDS` 调整。
- 可通过 `RUNTIME_MCP_ALLOWED_HOSTS` 配置允许的 MCP host 白名单，通过 `RUNTIME_MCP_BLOCKED_HOSTS`
  补充阻断列表；默认阻断云 metadata 端点。发现失败会产生 `mcp.discovery.failed` 运行事件。
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

- Workbench：`/workbench`
- 静态 Workbench 文件：`/static/workbench.html`
- 接口文档：`/static/api-docs.html`
- HTTP：`POST /api/agents/{agent}/runs`
- SSE：`POST /api/agents/{agent}/runs/stream`
- Apps：`GET /api/apps/templates`
- ACP WebSocket：`/api/acp/ws`
- ACP stdio：`python -m app.cli acp-stdio --agent default`
- MCP HTTP：`POST /mcp`

ACP WebSocket 支持通过 `runtimeOptions.workflow` 显式启用 workflow。新入口应使用
`agentName=default`。
ACP stdio 用于编辑器或本地 agent 客户端的 stdio 集成场景，同样默认进入 `agent_loop`。

ACP prompt 会把官方 `TextContentBlock`、`ImageContentBlock`、`ResourceLink` 和嵌入式
`ResourceContentBlock` 映射到内部 `Message` 与 `Attachment`。图片、音频和 blob resource 会以
base64 attachment 传给 runtime；resource link 会保留虚拟路径；嵌入式文本 resource 会同时追加到
用户消息正文，避免下游只读文本时丢上下文。

当 agent loop 需要用户补充输入时，本轮 `session/prompt` 会结束。为了对齐官方 ACP SDK，
顶层 `stopReason` 仍返回官方允许的 `end_turn`，JetLinks 扩展原因会放在
`_meta.jetlinks.stopReason = "input_required"`，并在 `result.metadata` 中带出结构化需求。WebSocket
连接保持打开，前端应把当前轮次视为“等待用户补充材料”，用户上传或填写后再发起下一次
`session/prompt`。插件和 workflow 不需要自己管理会话状态；它们可以显式返回
`metadata.requires_input=true` 和 `metadata.required_inputs`，协议层也会对常见“缺少上传文件”
回复做保守推断。

```json
{
  "stopReason": "end_turn",
  "threadId": "web-001",
  "agentName": "default",
  "_meta": {
    "jetlinks": {
      "stopReason": "input_required",
      "requiresInput": true
    }
  },
  "result": {
    "agent": "default",
    "thread_id": "web-001",
    "status": "completed",
    "reply": "请上传图片后继续。",
    "metadata": {
      "requires_input": true,
      "required_inputs": [
        {
          "type": "image",
          "accept": "image/*",
          "required": true,
          "reason": "The agent requires an uploaded image."
        }
      ]
    }
  }
}
```

`required_inputs[].type` 用于告诉客户端应该展示什么输入控件，当前约定如下：

| type | 用途 | 默认 accept |
| --- | --- | --- |
| `file` | 通用文件或无法细分的附件 | `*/*` |
| `image` | 图片、截图、照片 | `image/*` |
| `video` | 视频证据或视频数据 | `video/*` |
| `audio` | 音频、语音文件 | `audio/*` |
| `dataset` | 数据集、COCO JSON、CSV、Parquet、压缩包 | `.zip,.tar,.tar.gz,.csv,.json,.jsonl,.parquet,.yaml,.yml` |
| `model` | 模型权重、checkpoint、ONNX、safetensors 等 | `.onnx,.pt,.pth,.bin,.safetensors,.gguf,.pkl,.joblib` |
| `model_config` | 模型配置文件 | `.json,.yaml,.yml,.toml` |
| `text` | 文本参数 | `text/plain` |
| `json` | JSON 参数 | `application/json` |
| `number` | 数值参数 | `number` |
| `boolean` | 布尔参数 | `boolean` |
| `secret` | 密钥、token、密码类输入 | `password` |

推荐插件或 workflow 显式返回 `required_inputs`，例如：

```json
{
  "metadata": {
    "requires_input": true,
    "required_inputs": [
      {
        "type": "dataset",
        "accept": ".zip,.json,.jsonl",
        "reason": "COCO auto annotation requires a source image dataset."
      },
      {
        "type": "model_config",
        "reason": "SAM3 model configuration is required for this profile."
      }
    ]
  }
}
```

ACP WebSocket 的 `session/new` 可以传入 session 默认运行参数。调用方可以直接传
`runtimeOptions`，也可以传 `appTemplateName` 引用 `config/apps/*.json`，由服务端自动带出模板里的
`agent_name`、`workflow`、`selected_skills`、`selected_mcp_tools` 和 `runtime_options`。后续
`session/prompt` 会继承这些默认值，本轮 prompt 再传 `runtimeOptions` 时优先覆盖：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "session/new",
  "params": {
    "appTemplateName": "iot-architecture-diagram",
    "threadId": "web-iot-001",
    "cwd": "/",
    "runtimeOptions": {
      "modelName": "Qwen3.6-35B-A3B",
      "temperature": 0.2,
      "topP": 0.8,
      "maxTokens": 4096,
      "requestTimeoutSeconds": 120
    }
  }
}
```

ACP WebSocket 和 ACP stdio 都支持 `session/list`、`session/close`、`session/set_model` 等常用
session 扩展方法，并提供 `session/cancel`。取消是 best-effort：协议层会取消当前 prompt task，
并向 external ACP backend 转发 cancel；已经进入同步线程、沙箱或远端 provider 的底层操作可能不会瞬时停止，
但协议响应会返回 `stopReason=cancelled`，客户端不会再被正在执行的 prompt 阻塞。

ACP WebSocket 默认启用 prompt keepalive。`session/prompt` 运行期间，服务端会先通过
`session/update` 发送 `agent_thought_chunk`，提示“正在处理，请等待...”和“已收到请求，正在处理，请等待...”；
随后默认每 5 秒发送一次 `acp.prompt.keepalive` / `acp.prompt.wait_message` 更新，前端可用这些非最终
`result` 的消息展示“处理中”状态。可通过 `ACP_PROMPT_KEEPALIVE_ENABLED=0` 关闭，或用
`ACP_PROMPT_KEEPALIVE_SECONDS` 调整间隔。

### ACP 支持矩阵

当前实现是面向 JetLinks Workbench、服务端 Agent Runtime 和外部 ACP stdio backend 的 ACP 兼容实现。
核心会话链路、流式更新、文件、终端、权限请求、模型切换、应用模板和成果工作区都已经接入；IDE 侧
diff/plan 富 UI 目前通过 `session/update._meta.jetlinksRuntimeEvent` 传递原始运行事件，前端可按需渲染。

| 能力 | WebSocket | stdio Agent Server | external ACP backend client 回调 | 说明 |
| --- | --- | --- | --- | --- |
| JSON-RPC 2.0 request/result/error | 支持 | 由 ACP SDK 支持 | 由 ACP SDK 支持 | WebSocket 错误码遵循 JSON-RPC。 |
| `initialize` | 支持 | 支持 | 支持 | WebSocket 在 `_meta.jetlinks.methods` 暴露 JetLinks 当前方法列表。 |
| `authenticate` | 支持 | 支持 | - | 当前为无交互成功返回；生产鉴权仍由 HTTP/WebSocket token 控制。 |
| `session/new` | 支持 | 支持 | 支持 | WebSocket 支持 ACP 标准 `mcpServers`；JetLinks 扩展参数放在 `params._meta`，例如 `_meta.appTemplateName` 和 `_meta.runtimeOptions`。 |
| `session/load` / `session/resume` | 支持 | 支持 `resume` | 由后端能力决定 | 用于恢复或更新当前运行 session；历史记忆由 `.runtime/threads/<thread_id>/memory` 负责。 |
| `session/prompt` | 支持 | 支持 | 支持 | 进入本地 agent loop 或 external ACP backend。 |
| `session/update` | 支持 | 支持 | 支持 | 原始事件在 `_meta.jetlinksRuntimeEvent`。 |
| `session/cancel` | 支持 | - | 支持转发 | best-effort 取消当前 prompt。 |
| `session/list` / `session/close` / `session/fork` | 支持 | 支持 | - | JetLinks 会话管理扩展。 |
| `session/set_model` | 支持 | JetLinks 扩展 | - | 服务端模型托管场景使用。 |
| `session/set_mode` | 支持 | 支持 | - | `plan` 禁用工具只规划；`safe` 只保留低风险/只读工具；`edit` 为默认编辑模式但禁用本地 shell；`autonomous` 放宽工具并提高工具轮数上限；`yolo` 在 autonomous 基础上自动批准权限请求。 |
| `session/set_config_option` | 支持 | 支持 | - | 可更新 `modelName`、`temperature`、`selectedSkills`、`workflow`、`sandboxProfile` 等 session 运行配置。 |
| `session/request_permission` | 支持 | - | 支持 | WebSocket 会先发 `session/update: permission_request` 给前端；无交互 fallback 默认拒绝。可传 `approved=true` 或 `selectedOptionId` 返回批准/选择。`mode=yolo` 时自动批准。external backend 默认拒绝，避免静默授权；yolo session 代理 external backend 时自动允许。 |
| `fs/read_text_file` | 支持 | - | 支持 | WebSocket 只允许读取 session `cwd` 内文件或 `/mnt/user-data` 虚拟路径。 |
| `fs/write_text_file` | 支持 | - | 支持 | WebSocket 只允许写 session `cwd` 内文件或 `/mnt/user-data/workspace|outputs`；`uploads` 只读。 |
| `terminal/create` / `terminal/output` / `terminal/wait_for_exit` / `terminal/kill` / `terminal/release` | 支持 | - | 支持 | WebSocket 终端默认 cwd 为当前 thread workspace；输出有大小上限。 |
| text / image / audio content block | 支持 | 支持 | 支持 | image/audio 会转为内部 Attachment。 |
| resource link / embedded resource | 支持 | 支持 | 支持 | resource link 保留路径；embedded text 会追加到用户消息正文。 |
| `input_required` | 支持 | WebSocket 支持 | 由后端返回决定 | 顶层 `stopReason=end_turn` 保持官方兼容；JetLinks 扩展为 `_meta.jetlinks.stopReason=input_required`，并返回 `metadata.required_inputs`。 |
| artifact workspace | 支持 | 支持 | 支持 | 线程目录为 `.runtime/threads/<thread_id>/outputs` 等。 |
| plan/diff 稳定更新 | 支持 | 支持 | 支持 | 除原始 `_meta.jetlinksRuntimeEvent` 外，每条 update 还会尽量附带 `_meta.jetlinksPlan` 与 `_meta.jetlinksDiff`，前端可直接渲染计划和文件/产物变化。 |

ACP mode 影响 agent loop：

- `plan`：只做方案规划，不向模型暴露工具。
- `safe`：保留只读或低风险工具，过滤 skill、本地写文件和本地 shell，工具轮数最多 2 轮。
- `edit`：默认模式，允许编辑类工具，但过滤本地 shell。
- `autonomous`：允许更宽的工具集合；未覆盖轮数时最多提升到原配置的 2 倍且不超过 16，显式
  `configOptions.max_tool_rounds` 可在 1 到 32 之间覆盖基础轮数。
- `yolo`：工具暴露和轮数策略同 `autonomous`，并且 `session/request_permission` 在没有显式
  `approved` / `selectedOptionId` 时由服务端自动批准。它不会跳过 `input_required`，缺文件、缺图片、
  缺 `data.yaml`、缺模型配置这类内容缺口仍会终止本轮并要求用户补齐。

ACP config options 当前支持：

- `modelName` / `modelId`：切换托管模型或兼容旧模型名。
- `temperature`：覆盖本 session 后续 prompt 的采样温度。
- `selectedSkills`：覆盖本 session 默认 skill 列表。
- `workflow`：覆盖本 session 默认 workflow。
- `sandboxProfile`：记录本 session 的沙箱 profile 偏好，供 skill/sandbox 策略消费。

权限请求闭环：

1. 服务端收到 `session/request_permission` 且没有 `approved` / `selectedOptionId` 时，先向该 session 发送
   `session/update`，其中 `update.sessionUpdate = "permission_request"`。
2. 前端展示确认弹窗或选择控件。
3. 用户确认后，前端再次调用 `session/request_permission`，带 `approved=true` 或 `selectedOptionId`。
4. 如果没有前端确认，服务端默认返回 `cancelled`，避免无感授权；当 session `mode=yolo` 时，服务端会
   自动选择 allow 类选项或直接返回 approved。

WebSocket 文件路径规则：

- `/mnt/user-data/workspace/...` 映射到当前 thread 的 `workspace/`。
- `/mnt/user-data/uploads/...` 映射到当前 thread 的 `uploads/`，只读。
- `/mnt/user-data/outputs/...` 映射到当前 thread 的 `outputs/`。
- 本机绝对路径必须位于 session `cwd` 目录内；如果 `cwd=/`，会拒绝本机绝对路径，避免暴露整机文件系统。
- 相对路径按当前 thread `workspace/` 解析。

JetLinks 额外提供一个非官方 ACP 扩展方法，用于清理当前 session 绑定 thread 的运行文件：
WebSocket 使用 `jetlinks/session/delete_files`，stdio 按 ACP SDK 扩展约定使用
`_jetlinks/session/delete_files`。参数为 `sessionId` 必填，`scopes` 可选，取值限制为
`workspace`、`uploads`、`outputs`，默认清理三个 scope；`dryRun=true` 只返回将删除的虚拟路径、
文件数量和字节数，不落盘删除。该方法只删除 `/mnt/user-data/{workspace,uploads,outputs}` 下的内容，
保留 session 和 thread 容器目录本身，不等价于官方 `session/close`。

agent backend 分两类：

```text
local:
  backend.type = local
  使用 JetLinks 自己的 ToolCallingAgentLoop 作为主智能体闭环。

external:
  backend.type = acp_stdio
  代理 Codex 或其他 ACP stdio agent。
```

代理 external ACP backend 时，client 回调提供受限文件和终端能力：`fs/read_text_file` 可读取
`/mnt/user-data/workspace`、`/mnt/user-data/uploads`、`/mnt/user-data/outputs` 下的虚拟路径，
`fs/write_text_file` 只允许写入 workspace，并阻止路径穿越。`session/request_permission` 默认返回
cancelled，终端回调会在当前 thread workspace 内启动本地 subprocess 并捕获输出；这适合可信本地
ACP agent 集成，不应当当成远程不可信代码的安全沙箱。

## Sandbox 策略

Sandbox 执行是选择性的。普通对话、LLM spec planning、行为检测文本判断默认在本地 runtime 中完成。
只有 Skill manifest 显式映射到 sandbox profile，且当前 provider/executor 开启时，才会进入隔离执行路径。

Skill 实体 manifest 位置：

```text
config/skills/*.json
```

每个 manifest 描述平台侧契约：

- `input_schema`：Skill 接收的结构化输入
- `output_schema`：预期输出、产物或 JSON 结果
- `quality_template`：给 planner/verifier 的质量提示
- `sandbox.enabled`、`sandbox.profile`、`sandbox.adapter_command`、fallback 策略

Skill 实现插件位置：

```text
plugins/skills/<skill-id>/plugin.json
plugins/skills/<skill-id>/skills/<skill-id>/manifest.json
```

插件负责提供真实执行资产，不作为正式技能清单的主来源。正式技能是否对 App/Java/Workbench 可见，以
`config/skills/*.json` 是否存在为准；正式技能也应在 `plugins/skills/<skill-id>` 中具备同名技能包。

Workflow 实体 manifest 位置：

```text
config/workflows/*.json
```

每个 manifest 描述平台侧 Workflow 契约：

- `name`：Workflow 唯一名，供 `runtime_options.workflow` 和应用模板引用
- `display_name` / `description`：Workbench 和 Java 侧展示信息
- `enabled`：是否注册到运行时
- `handler`：绑定到插件实现的 Python handler，例如 `artifact.py:ArtifactWorkflow`
- `trigger`：当前固定为显式运行参数触发

Workflow 实现插件位置：

```text
plugins/workflows/**
```

插件只负责执行实现，不作为正式 Workflow 清单的主来源。正式 Workflow 是否对 App/Java/Workbench 可见，以
`config/workflows/*.json` 是否存在为准。

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

本机开发默认推荐轻量 local：

```json
{
  "provider": "local",
  "executor_enabled": false,
  "fallback_to_local": true
}
```

如果希望不启动 OpenSandbox Server / Docker，但又把 Skill adapter 放到独立进程里执行，可以复制
`config/sandbox/runtime.local.example.json` 为 `config/sandbox/runtime.local.json`，启用 `local_subprocess`：

```json
{
  "provider": "local_subprocess",
  "executor_enabled": true,
  "sandboxed_skills": ["drawio-generation"],
  "fallback_to_local": true
}
```

`local_subprocess` 会在当前 thread 下创建本地执行目录：

```text
.runtime/threads/<thread_id>/sandbox-run/
  workspace/request.json
  outputs/
  inputs/
  skills/
```

它不依赖 OpenSandbox Server 或 Docker，适合可信 Skill 的本机调试；它不是安全沙箱，不能用于执行不可信代码。
服务器部署或强隔离场景再启用 `opensandbox` provider。

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
- 确定性上下文压缩：保留头尾消息，把中间历史转换为 reference-only 交接摘要
- 轻量 `delegate_task`：无工具子模型调用，适合独立分析和第二视角总结

后续可继续参考 Hermes 分批补齐：

1. 多 provider adapter 和统一 `NormalizedResponse`
2. Memory 增强和 Session Resume
3. 更完整的 Context 引擎（LLM 摘要、token 预算、压缩失败退避、主动任务保护）
4. Browser/Web tools
5. Cron/Gateway 和多平台消息接入
6. 完整多层子智能体 delegation（子 agent 工具集、并发、暂停、中断、观测）

优先建议继续保持“可选、插件化、agent JSON 可控”的方式接入，避免把通用助手的高权限能力直接写进主流程。

## 验证

当前分支提交前使用以下命令验证：

```bash
python -m ruff check app tests
python -m mypy app
python -m pytest -q
```
