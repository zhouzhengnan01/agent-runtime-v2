# VideoPatrolTemplate 适配说明（短版）

> 本文为 **压缩短版**，用于快速对接与排错；以 `video_patrol_template.py` 当前实现为准。

---

## 1) Chat 任务（聊天界面）

### 1.1 original_params 结构
```json
{
  "sessionId": "...",
  "content": "...",
  "messageType": "text",
  "context": {
    "system_prompt": "..."
  }
}
```
- **message**：来自 `content`
- **system_prompt**：来自 `context.system_prompt`

### 1.2 content 命令约定
`content` 为自然语言，但需包含：**动作命令 + URL +（可选）轮询间隔 +（可选）停止时间**  
动作命令仅支持：
- 开始类：启动分析/开始分析/启动巡检/开始巡检
- 停止类：停止分析/停止巡检/终止分析/结束分析/结束巡检
- 状态类：查看状态/查看进度/当前状态/当前进度/任务状态

### 1.3 URL -> 底层模式选择
- `http(s)://...`：离线视频分析 → **OFFLINE**
  - HTTP 视频会先下载到：`storage/http_tmp/<sessionId>/...`，再传本地路径给底层
  - **忽略轮询间隔**
  - system_prompt 放入：`VlmConfig.offline_system_prompt`
- `rtsp(s)://...` / `rtmp(s)://...`：实时巡检 → **SECURITY_SINGLE（聊天侧单路）**
  - system_prompt 放入：`RTSPBatchConfig.polling_list[0].rtsp_system_prompt`
  - 轮询间隔（若提供）生效

### 1.4 输出与流式
- Chat 侧：`VlmConfig.vlm_streaming=True`（流式输出）
- 模板可多次 `yield` 字符串块，前端流式展示（启动提示/状态/停止/错误等）

---

## 2) Machine 任务（机器视觉，JSON-RPC）

### 2.1 JSON-RPC 输入
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "ComputerVisionTask",
  "params": {
    "state": "start",
    "source": [
      {
        "id": "cam-1",
        "rtsp": "rtsp://...",
        "rtmp": "rtmp://...",
        "prompt": "..."
      }
    ],
    "interval": 13,
    "alwaysReturn": false
  }
}
```

### 2.2 支持模式
Machine 端 **不支持 OFFLINE**，仅支持：
- 单路：`SECURITY_SINGLE`
- 多路：`SECURITY_POLLING`

规则：
- `len(source) == 1` → SECURITY_SINGLE
- `len(source) >= 2` → SECURITY_POLLING
- 多路时 `interval < 10` 会自动提升到 `10`

### 2.3 关键：prompt 追加约束（强制结构化）
每路 `source.prompt` 会被模板追加常量：

- `prompt = (src.prompt 或 "") + VLM_PATROL_REPORT_WITH_OBJECTS_JSON_CONSTRAINTS`

该常量定义在 `video_patrol_template.py` 内，用于要求 VLM 输出“**巡检报告 + objects 坐标**”的结构化 JSON（下游依赖 `full_text` 做画框/策略）。

> 注意：Machine 侧 `VlmConfig.vlm_streaming=False`（非流式）。

---

## 3) Machine 返回（模板输出 & 上层包装）

### 3.1 模板层（video_patrol_template.py）实际 yield 的 payload
模板会在收到底层事件 `type == "vlm_stream_done"` 时产出：

```json
{
  "params": {
    "sourceId": "cam-1",
    "evidence_image_urls": ["http://.../storage/...jpg"],
    "evidence_image_box_urls": ["http://.../storage/...jpg"],
    "full_text": "..." 
  }
}
```

- `sourceId`：来自底层 `stream_rtsp_id`
- `evidence_image_urls`：证据帧 URL（模板负责把 `/storage/...` 拼成可访问的 http(s)）
- `full_text`：底层 VLM 原始输出（字符串/JSON 字符串/数组等；模板会对字符串做 `strip` 并透传）

### 3.2 上层（JAIP handler）包装后的 JSON-RPC（供业务方理解）
上层会把上述 payload 放进 `result` 并带回请求 `id/jsonrpc`（示意）：

```json
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": {
    "sourceId": "cam-1",
    "evidence_image_urls": ["http://..."],
    "evidence_image_box_urls": ["http://..."],
    "full_text": "..."
  }
}
```

---

## 4) alwaysReturn 与画框逻辑（Machine）

### 4.1 判定依据：full_text 是否包含对象
模板会从 `full_text` 解析对象框：
- 兼容：
  - 旧格式：`full_text` 直接是 `[{label, confidence, box}, ...]`
  - 新格式：事件数组，每条事件含 `objects:[{label, confidence, box}]`
- 会把所有 `objects` 拉平用于判断“空/非空”和画框输入

### 4.2 行为规则
- **若对象为空**
  - `alwaysReturn=false`：**不返回任何结果**（继续等待下一轮底层消息）
  - `alwaysReturn=true`：返回结果，且  
    `evidence_image_box_urls = evidence_image_urls`（不画框）
- **若对象非空**
  - 使用 `export_evidence_images_with_boxes(...)` 画框
  - 画框结果写入：`storage/evidence_images_box/<rpc_id>/...`
  - 返回 `evidence_image_box_urls` 为画框图 URL

---

## 5) stop / 清理 / 并发（chat + machine 共用）

### 5.1 并发控制
最大并发由 `.env` 的 `VLM_MAX_THREAD` 控制（聊天 + 机器视觉共用）。  
超过并发：
- Chat：yield 提示文本
- Machine：返回 `{"error":{"message":"当前任务繁忙，建议指数退避后再次请求。"}}`

### 5.2 stop 行为
- Chat：识别到“停止分析/巡检” → `analyzer.force_stop()` → 清理 sessionId 目录
- Machine：`params.state == "stop"` → 按 `rpc_id` 找任务 → `force_stop()` → 清理 rpc_id 目录

### 5.3 清理目录
- Chat（按 sessionId）：
  - `storage/evidence_images/<sessionId>`
  - `storage/out/<sessionId>`
  - `storage/http_tmp/<sessionId>`
- Machine（按 rpc_id）：
  - `storage/evidence_images/<rpc_id>`
  - `storage/evidence_images_box/<rpc_id>`
  - `storage/out/<rpc_id>`

---

## 6) 最小对接检查清单（机器视觉）

1. `id`（rpc_id）必须有：用于 task_id 目录隔离与清理  
2. `params.source[]` 至少 1 个，且每个 source 必须有 `rtsp` 或 `rtmp`  
3. `prompt` 会被追加 `VLM_PATROL_REPORT_WITH_OBJECTS_JSON_CONSTRAINTS`（确保 full_text 可解析）  
4. 业务侧若希望“无目标也回包”，必须 `alwaysReturn=true`  
