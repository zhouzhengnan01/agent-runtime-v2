# AI Handoff Notes

## 当前可视化智能体进度

- 可视化工作流已配置到 `config/apps/ai-vis-page.json`，字段为 `"workflow": "visualization_bigscreen_workflow"`。
- 当前主要技能已切到 `plugins/skills/ai-vis-page`。`1780654477379uykyzod7`、`1780988275767z1lxtrd1` 作为平台侧旧 ID 兼容别名，运行时会映射到 `ai-vis-page`。
- 工作流实现文件是 `plugins/workflows/visualization-bigscreen/visualization_bigscreen.py`。
- 当前链路已跑通：大模型生成内容，Python 工作流负责上传背景图、保存 ECharts 资源、上传高级组件 zip、回填 fileId。
- 地图新逻辑已加：地图、地理、点位、经纬度、区域/空间分布统一使用 `pseudo` 组件。
- `pseudo` 必须按 `references/components/pseudo.json` 模板，点位数据写入 `dataSourceProps.defaultValue`，字段固定为 `name / longitude / dimension / value`，其中 `dimension` 是纬度。
- 非地图图表走 `resourceComponentEcharts/custom-chart`，复杂高级组件走 `advancedComponents/custom-component`，`tabs` 已禁用。
- 工作流会对 AI 返回的 `pseudo` 做兜底标准化，把 `lat/lng/points/markers` 尽量转成平台 pseudo 模板结构。
- 工作流已改成 stage-aware context 主路径：首次 LLM 请求不再注入完整 `SKILL.md + references`，而是按 `initialization/region` 和区域语义选择最小规则上下文。重试只是复用同一阶段上下文，不再把 stage-aware 当 504 后兜底。
- `plugins/skills/ai-vis-page/SKILL.md` 已瘦身为流程契约，只描述初始化、区域生成、Python 副作用和组件选择意图；具体平台 JSON 规则保留在 references，由 `visualization_bigscreen.py` 按阶段读取。
- 当前上下文策略：`initialization` 只加载蓝图/SVG 相关规则；`region` 阶段固定加载标题 text 模板，并根据区域语义加载 pseudo、table、video、custom-chart/ECharts 或高级组件规则。

## 关键目录

- `config/apps/`：智能体配置。
- `config/workflows/visualization_bigscreen_workflow.json`：工作流配置。
- `plugins/skills/ai-vis-page/`：当前可视化技能。
- `plugins/skills/ai-vis-page/references/components/pseudo.json`：平台 pseudo 地图组件模板。
- `plugins/workflows/visualization-bigscreen/visualization_bigscreen.py`：可视化工作流核心代码。

## 启动后端服务

建议用 PowerShell 提权启动，因为 `.runtime` 挂载目录在普通权限下可能出现 `[WinError 5] 拒绝访问`。

```powershell
$old = (netstat -ano | Select-String ':8000' | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -First 1)
if ($old) {
  Stop-Process -Id ([int]$old) -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 2
}

$python = (Get-Command python).Source
$p = Start-Process -WindowStyle Hidden -FilePath $python `
  -ArgumentList @('-B','-m','uvicorn','app.main:app','--host','0.0.0.0','--port','8000') `
  -WorkingDirectory (Get-Location) `
  -PassThru

Start-Sleep -Seconds 8
Write-Output "started pid=$($p.Id)"
netstat -ano | findstr 8000
Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 15
```

健康检查成功时应返回：

```json
{"status":"ok","service":"jetlinks-agent-runtime-v2"}
```

## 排查要点

- 如果线上报“当前工作流没有注册”，优先确认 `plugins/workflows/visualization-bigscreen` 是否已同步、工作流配置是否存在、应用 JSON 是否指定 workflow。
- 如果 LLM 报 500 且包含 `Connection reset by peer`，通常是 LLM 网关或模型上游断连，不是 Python 生成 JSON 本身失败；看 providerId、requestId 和是否出现 `llm.request.retry`。
- 如果线上日志仍出现 `skill_md_chars` 和 `total_reference_chars: 120000`，说明线上 workflow 还没有同步当前主路径 stage-aware context 版本，或服务没有重启成功。
- 如果地图组件结构不对，优先检查线上是否同步了最新版 `ai-vis-page` 技能和 `visualization_bigscreen.py` 工作流。
- 如果启动后访问 `.runtime` 报 `[WinError 5]`，用提权 PowerShell 重启服务。

## 2026-06-09 线上/本地差异排查进度

- 最新本地逻辑中，第一次 LLM 请求前的 `skill.context.loaded` 事件就应包含 `stage_aware: true`、`primary_context: true`，且不应出现 `skill_md_chars`、`total_reference_chars: 120000`。只有上游 500/502/503/504 后才会额外出现 `strategy: "stage_aware_context_retry"`。
- 因此线上如果仍出现 `compact_visualization_context` 或完整 `skill_md_chars` 注入，实际运行链路仍不是当前本地最新版；需要重点确认服务是否重启、是否加载了另一个同名 workflow 插件目录、workflow 配置是否指向正确 handler。
- 除 `plugins/workflows/visualization-bigscreen/visualization_bigscreen.py` 外，当前排查认为还有几类文件会影响线上链路：
  - `app/core/agent/runtime.py`：当前分支非流式 workflow 路径直接执行 `workflow.run_with_events(...)`，不是丢到 `asyncio.to_thread(...)`。这不会直接造成 LLM 网关 504，但会阻塞 Python 事件循环，放大本地/线上卡住和请求堆积问题。
  - `app/core/apps/registry.py`、`app/core/tools/registry.py`、`app/core/resources.py`：这些和配置/插件上传目录、默认目录覆盖逻辑有关。线上如果依赖 `config/upload`、`plugins/upload` 或镜像默认目录，加载不到 app/skill/tool 会导致链路不一致。
  - `config/workflows/visualization_bigscreen_workflow.json`、`plugins/workflows/visualization-bigscreen/plugin.json`、workflow manifest：这些决定 workflow 是否注册以及 handler 指向哪个 Python 文件。只同步 Python 文件不一定够。
  - `app/core/llm/openai_compatible.py`：主要影响 LLM 日志、图片 URL、上游 500/502/503/504 重试和总耗时，不是这次仍走旧 `compact_visualization_context` 的第一嫌疑。
- 最新线上日志里出现过 `selected_skills: ["1780988275767z1lxtrd1"]`，但事件是 `skill.context.loaded found=false reason=skill_not_found`。当前本地已将 `1780988275767z1lxtrd1` 映射到 `ai-vis-page`，线上仍报 not_found 时优先确认 `app/core/skills/aliases.py` 是否同步和服务是否重启。
- 如果使用最新分支重新本地测试，关键验证点：
  - 初始化阶段日志不应再出现 `skill_md_chars`、`total_reference_chars: 120000`。
  - 初始化阶段第一次 `skill.context.loaded` 就应出现 `stage_aware: true`、`primary_context: true`、`stage: "initialization"`。
  - 上游 504 后如果进入工作流重试，应出现 `strategy: "stage_aware_context_retry"`。
  - 如果仍报工具不存在，重点看 Java MCP 暴露的真实工具名是否等于默认 `visual-bigscreen_UploadFile`，或是否需要配置 `VISUAL_BIGSCREEN_UPLOAD_MCP_TOOL`。

## 2026-06-10 本地联调错误结论

- `WinError 10061 由于目标计算机积极拒绝，无法连接` 发生在 LLM 请求阶段，日志表现为 `llm upstream retry ... reason=ConnectError`，请求地址是 `http://192.168.32.7:9200/.../chat/completions`。这说明当次请求时 Java/LLM 网关 9200 没有可用连接，和 skill 注入大小不是同一个问题。
- 本地当前配置里 LLM `base_url` 和 MCP server 都走 Java 9200，因此发起测试前必须保证 Java 9200 已启动且模型代理路径可用。Python 可先启动，但只要请求发生时 Java 9200 不可达，工作流就会失败；如果 Java 是进程管理方，推荐顺序是 Java/LLM 网关 ready 后再启动或重启 Python。
- `WinError 5 拒绝访问：*.tmp -> conversation.jsonl / run-*.json` 是 Windows 文件替换问题。已对 `SessionConversationStore` 和 `RunEventStore` 改成唯一 tmp 文件 + 多次重试 + 直接写入兜底，避免失败结果保存时二次报错覆盖原始异常。

## 2026-06-10 演示兜底通道

- 已在 `plugins/workflows/visualization-bigscreen/visualization_bigscreen.py` 增加可视化工作流直连 LLM override：配置了 `visualBigscreenDirectLlmBaseUrl` / `visualBigscreenDirectLlmChatUrl`、`visualBigscreenDirectLlmApiKey`、`visualBigscreenDirectLlmModel` 后，workflow 会绕开 Java LLM 网关，直接用 OpenAI-compatible `/chat/completions` 调模型。
- 同一逻辑也支持 `directLlm*`、`openai*`、`codex*` 形式的 `runtime_options.config_options` key，兼容嵌套 `ccSwitch.openai.baseUrl/apiKey/model`，以及环境变量 `VISUAL_BIGSCREEN_DIRECT_LLM_BASE_URL`、`VISUAL_BIGSCREEN_DIRECT_LLM_CHAT_URL`、`VISUAL_BIGSCREEN_DIRECT_LLM_API_KEY`、`VISUAL_BIGSCREEN_DIRECT_LLM_MODEL`、`VISUAL_BIGSCREEN_DIRECT_LLM_TIMEOUT_SECONDS`。
- 如果给的是完整 `.../chat/completions` 地址，workflow 会自动裁成 OpenAI-compatible `base_url`，避免二次拼接 `/chat/completions`。
- 已在 `app/core/tools/providers/visualization/bigscreen/provider.py` 增强上传兜底：只要配置了 `visualBigscreenUploadUrl` / `fileUploadUrl` 或环境变量 `VISUAL_BIGSCREEN_UPLOAD_URL` / `JETLINKS_FILE_UPLOAD_URL`，即使 runtime 里还有 MCP server，也优先走 REST multipart 上传。
- 永久 token 支持两种方式：完整鉴权头用 `visualBigscreenUploadAuthorization` 或 `VISUAL_BIGSCREEN_UPLOAD_AUTHORIZATION`；只有 token 值时用 `visualBigscreenUploadToken` 或 `VISUAL_BIGSCREEN_UPLOAD_TOKEN`，默认拼成 `Authorization: Bearer <token>`。如果平台要求其他 header，可配 `visualBigscreenUploadTokenHeader` 或 `VISUAL_BIGSCREEN_UPLOAD_TOKEN_HEADER`。
- multipart 文件字段默认是 `file`，可用 `visualBigscreenUploadField` 或 `VISUAL_BIGSCREEN_UPLOAD_FIELD` 改成平台接口需要的字段名。
- 资源保存仍默认走原 `visualization_bigscreen_save_resource` MCP。若后续也要绕开 MCP，可额外配置 `visualBigscreenResourceSaveUrl` 或 `VISUAL_BIGSCREEN_RESOURCE_SAVE_URL`，当前实现会按 `{"data": [resource...]}` POST JSON。
