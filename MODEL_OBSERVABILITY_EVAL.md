# 模型观测与效果评估（基于 jetlinks-agent）

本文描述在当前 `jetlinks-agent` 代码基础上，如何做 **模型可观测（Observability）** 与 **效果评估（Evaluation）** 的整体思路与落地步骤。

适用入口（两种都覆盖）：
- HTTP JSON：`POST /api/v1/agents/{agentId}/chat/json`（页面 `/review.html` 的数据来源）
- WebSocket（JAIP）：`/api/v1/jaip/session/{agentId}`（同样也挂在 `/api/v1/websocket/session/{agentId}`；自动模式见 `websocket_auto.py`）

---

## 1. 目标与边界

### 目标
- **可定位**：当结果异常（字段为空、误报/漏报、视频抓帧失败、schema 不匹配等）能快速定位到“输入/模型/配置/外部依赖”的哪一环。
- **可量化**：用稳定指标衡量“可靠性、质量、性能、成本”随时间变化，并能对比不同模型/提示词/参数的差异。
- **可回放**：对线上真实请求进行离线回放（replay），复现并回归验证。
- **可灰度**：支持 A/B 或分流策略上线新模型/新提示词，观察指标后再全量。

### 边界（务实）
- **观测**解决“发生了什么/为什么”，**评估**解决“好不好/有没有退化”。
- 结构化字段（JSON）可以做强约束评估；自由文本字段建议用“规则 + 抽样人工复核/LLM Judge”的组合。

---

## 2. 现有能力（这套代码里已经有的）

### 2.1 调用记录落盘（review records）
- HTTP `POST /api/v1/agents/{agentId}/chat/json` 已在服务端落盘请求与响应（含 `task`、`meta.media_frames` 等）。
- 查询接口：
  - 列表：`GET /api/v1/review-records?pageIndex=0&pageSize=50`
  - 详情：`GET /api/v1/review-records/{record_id}`
- 存储目录（默认）：`storage/review_records/<record_id>/record.json`
- 页面：`/review.html` 可查看记录、预览图片/视频、告警等。

这套落盘机制是做“观测+评估”的最关键底座：所有指标都可从 `record.json` 反算/聚合。

> WebSocket（JAIP）目前以“实时流式返回”为主。要让 WS 也纳入统一观测/评估，推荐在 WS 的“请求开始/结束”处同样写入 `review_records`（见本文第 5 节落地路径）。

### 2.2 Agent 配置来源（DB）
- Agent 的 `model/system_prompt/temperature/max_tokens...` 在 DB 的 `agents.config` 字段（JSON）里。
- DB 连接信息来自 `.env`（如 `DB_HOST/DB_USER/DB_PASSWORD/DB_NAME` 等）。

### 2.3 模型路由（环境变量）
当前项目使用 OpenAI 兼容接口形式调用模型：
- 文本模型：`LLM_BASE_URL/LLM_API_KEY/LLM_MODEL`
- 视觉模型：`VLM_BASE_URL/VLM_API_KEY/VLM_MODEL`

不论你使用云服务还是本地自建，只要提供 **OpenAI 兼容** 的 `base_url` 与 `api_key`，均可工作。

### 2.4 相关代码位置（便于落地）
- HTTP `agent_json`：`app/api/v1/agents_json.py`（解析、调用、落盘、构建 `meta.media_frames`）
- JSON 处理器：`app/core/agents/agent_json.py`（模型选择/多模态兜底、调用与解析）
- Review records：`app/services/review_record_service.py`、`app/api/v1/review_records.py`
- 复判页面：`website/review.html`
- WebSocket（JAIP）：`app/api/v1/websocket.py`、`app/core/jaip/handler.py`（流式过程与最终输出）
- WebSocket 自动模式：`app/api/v1/websocket_auto.py`

---

## 3. 模型观测（Observability）设计

### 3.1 建议观测维度（每次调用都记录）
建议把每条调用记录分成 4 类信息：

1) **输入**（可复现）
- `channel`（`http`/`ws`）、`agent_id`、`task`（任务名）、`message`
- 关联 ID：
  - HTTP：`review_record_id`（服务端生成）/（可选）`X-Request-Id`
  - WS：`session_id` + JSON-RPC `id` +（可选）客户端 `traceId`
- `json_schema`（或 schema hash）
- `files`：图片/视频 URL（对 data URI 可截断/脱敏）
- `roi`：可选裁剪区域（`context.roi`），仅把裁剪后的图像/帧送入模型

2) **执行过程**
- 选用的模型：最终 `model`（包含是否发生“模型切换/兜底”）
- 关键参数：`temperature/max_tokens`
- 多媒体：抓帧/下载是否成功、耗时、失败原因（例如视频拉取失败）
-（WS 特有）流式过程指标：`time_to_first_chunk_ms`、`chunks_count`、`stream_duration_ms`

3) **输出**
- `raw_response`（原始模型输出）
- `parsed_json`（服务端解析后的 JSON）
- schema 验证结果（是否缺字段/类型不符/额外字段）

4) **性能/稳定性**
- `http_status`、`cost_ms`
- 错误类型与堆栈（若有）
-（可选）token usage / 计费（如果上游返回 usage）

> 当前 `review_records` 已覆盖其中多数项；后续可逐步补齐“schema 校验结果、token usage、模型切换原因”等字段。

### 3.2 指标（Metrics）建议
围绕 4 大类：

- **可靠性**
  - 请求成功率（`success=true` 比例）
  - JSON 解析成功率（能否 parse）
  - schema 通过率（必填字段齐全、类型匹配、无多余字段）
  - 空字段率（例如 `结论/建议/风险等级` 全为空字符串）
  - 外部依赖失败率（视频拉取/抓帧失败、文件下载失败）

- **质量（按任务定制）**
  - 结构化字段准确率/F1（如 `illegal`、`count`）
  - 规则一致性（是否按业务规则输出：例如 illegal=true 的严格判据）
  - 误报/漏报（需要人工标注或规则标注作为 gold）

- **性能**
  - P50/P95/P99 延迟（`cost_ms`）
  -（WS）TTFT（time-to-first-token/first-chunk）分位数
  - 多媒体耗时拆分（下载、抓帧、模型推理）

- **成本（可选）**
  - tokens / 次、tokens / 日
  - 费用估算（按供应商计费规则）

---

## 4. 效果评估（Evaluation）流程

### 4.1 数据集（Dataset）建设
建议从 `storage/review_records` 抽取并“固化”为评测集：
- 按 `task`/场景分桶（非法停车、事故复判、值守告警…）
- 覆盖：典型样本、边界样本、困难样本、失败样本（视频拉取失败/噪声大）
- 对每条样本写入 **期望输出（gold）**：
  - 结构化字段（boolean/int/enums）尽量人工标注
  - 文本字段给 rubric（评分标准）或关键点列表

数据集需要版本化（例如 `datasets/v1/`），以便回归对比。

### 4.2 离线回放（Replay）评测
核心目标：同一份输入，重复跑不同版本（模型/提示词/参数），自动产出对比指标。

建议产物：
- `run_<timestamp>/predictions.jsonl`（每条样本的输出）
- `run_<timestamp>/metrics.json`（指标汇总）
- `run_<timestamp>/report.html`（可视化报告，便于人工抽查）

离线评测优先用“结构化评测”：
- schema 校验（强约束）
- 关键字段对比（accuracy/F1/MAE）
- 失败原因统计（下载失败、抓帧失败、输出空等）

回放入口建议（二选一）：
- **优先回放 HTTP `agent_json`**：接口稳定、一次性返回完整结果，最适合批量评测。
- **需要覆盖 WS 场景时回放 WS**：用 WS 客户端脚本模拟 `session.initialize` → `session.message`，收集 `ai.responseComplete` 的最终内容再做同样的评测。

### 4.3 在线评估（灰度 / A-B）
当你要替换模型或提示词，建议：
- 给请求打 `experiment_id`（或在服务端按比例分流）
- A/B 同时运行一段时间
- 对比关键指标：成功率、schema 通过率、空字段率、误报/漏报（抽样复核）

---

## 5. 结合本项目的落地路径（最小可用 → 可规模化）

### Phase 1：把“观测数据”做完整（1～2 天）
- 在 `record.json` 里补充：
  - `model_selected`（最终模型）
  - `model_switch_reason`（例如：visual_input_but_text_model）
  - `schema_validation`（缺字段/类型不符/额外字段）
- 确保 `task` 在请求里透传并落盘（你已支持 `context.task`）。
- **让 WS 也落盘**（关键）：在 JAIP 的请求开始创建记录、在 `ai.responseComplete` 时写回最终 `raw_response/parsed_json/meta`，并把 `channel=ws` 与 `session_id/jsonrpc_id/traceId` 一起存下。

### Phase 2：做“离线评测脚本”（1～2 天）
- 从 `storage/review_records` 导出数据集（jsonl）
- 回放调用（调用本地接口或直接复用内部函数）
- 生成指标 + HTML 报告（至少包含：失败样本列表、字段空列表、schema 失败列表）

### Phase 3：做“线上灰度与看板”（按需）
- 引入 `experiment_id` / 版本号（记录到 `review_records`）
- Prometheus/Grafana 或简单的定时聚合任务输出日报（JSON/HTML）

---

## 6. 常见问题与排查建议（结合你遇到的案例）

### 6.1 “返回字段全空，但 error=null”
典型原因组合：
- `message` 要求字段与 `json_schema` 不一致（模型按 schema 生成空字符串也能通过 strict）
- 传了图片/视频但走了文本模型（视觉输入被忽略）
- 文件下载/抓帧失败（模型无法看到内容，只能空填或瞎猜）

建议观测项：
- schema 与 message 的一致性检查（可做静态规则）
- 记录最终使用的 `model`
- 记录文件加载状态（成功/失败 + reason）

---

## 7. 后续可选增强
- 增加“自动判分器”（rule-based + 抽样人工复核/LLM Judge）
- 在 UI 上展示“schema 失败/空字段/下载失败”的标签与筛选
- 统一日志 trace_id（把一次请求串起：API → 多媒体 → LLM）

### 7.1 模型调用稳定性：并发失败的补偿/重试策略（规划）
你提到的现象是典型的“并发 + 上游不稳定”问题：同一时间多个任务调用模型时，容易遇到 `429 限流`、`网关超时`、`5xx`、偶发连接断开/重置，导致任务失败。

**目标**：失败可补（可自动补 + 可手动补）、避免重复扣费、避免重试风暴、可观测可追踪。

建议策略（先写入文档，后续按需实现）：

1) **作业化（Job）/先落盘再执行**
   - 复用现有 `review_records`：收到请求先创建 `record.json`（保存 `request_payload`），并写入 `status=pending/running/succeeded/failed`、`attempt`、`next_retry_at` 等字段。
   - 同步模式：接口仍然等待模型返回；若失败也保留 `record_id`，方便后续补偿重跑。
   - 异步模式（推荐给视频/高耗时任务）：接口立即返回 `202 + record_id`，由后台 worker 执行；前端轮询 `/api/v1/review-records/<record_id>` 查看进度与结果。

2) **幂等（Idempotency）与去重**
   - 客户端传 `idempotency_key`（或 header `Idempotency-Key`），服务端用 `agent_id + task + message + files(url/sha) + roi + schema_hash` 生成/校验去重键。
   - 同一个幂等键在“未完成/已完成”状态下重复提交，直接返回同一 `record_id`（避免重复执行与重复扣费）。

3) **错误分类决定是否补偿**
   - **可重试**（transient）：429/5xx/网络超时/连接重置/上游返回临时错误/偶发 JSON 解析失败。
   - **不可重试**（permanent）：schema 不合法、必填参数缺失、文件 URL 404/鉴权失败、文件损坏、视频无法解码等。
   - 记录 `error_type`（例如 `RateLimited/Upstream5xx/Timeout/BadRequest/MediaDecodeError`）以便统计与策略调参。

4) **重试退避（Backoff）与抖动（Jitter）**
   - 建议：指数退避 + 抖动，例如 `1s → 3s → 10s → 30s → 60s`（上限 3～5 次）。
   - 遇到 `429` 优先遵循上游 `Retry-After`；无则退避加倍。
   - 设置全局“重试预算”（例如每分钟最多补偿 N 次），避免雪崩。

5) **并发控制与排队**
   - 以 `provider/api_key/model` 为维度设置并发上限（Semaphore/队列），宁可排队也不要把 429 打满。
   - 对视频任务可拆阶段：下载/抓帧与模型调用分别限流，并分别记录耗时与失败原因。

6) **死信队列（DLQ）+ 手动补偿入口**
   - 超过最大重试次数后标记 `status=dead`（或 `failed_permanent`），进入死信列表。
   - 提供“手动重试”入口（例如 `POST /api/v1/review-records/<record_id>/retry`），并在落盘记录中累积 `attempt_history`（每次尝试的时间、错误、耗时）。

> 有了上述机制，任何一次“并发导致的失败”都不会丢：要么自动补齐成功，要么明确落盘为可追踪的失败并支持后续一键补偿。
