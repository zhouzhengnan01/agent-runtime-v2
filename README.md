# JetLinks Agent (Local)

## Run

```bash
# from repo root
./deploy/jetlinks-agent/start_server.sh --reload
```

- Health: `GET http://localhost:8005/health`
- Web UI: `http://localhost:8005/` / `http://localhost:8005/review.html`

## Review Records (落盘 + 回放)

When calling `POST /api/v1/agents/{agentId}/chat/json`, the server will:

1. Write a call record to disk
2. If the request contains a video in `context.files` (first `media_type=video`), record a **30s mp4 clip** via `ffmpeg`
3. Keep only the latest **1000** records globally (oldest will be deleted)

### ROI（可选裁剪区域）

在 `context` 中可传 `roi`，仅把裁剪后的区域送入 VLM（图像/视频帧都生效）。

支持格式（rect）：

```json
{
  "context": {
    "roi": {"rect": [120, 80, 1400, 900]}
  }
}
```

```json
{
  "context": {
    "roi": {"rect": [0.1, 0.1, 0.9, 0.9], "normalized": true}
  }
}
```

## .env 配置说明（中文）

### 数据库
- `DB_TYPE`: 数据库类型（postgresql / mysql）
- `DB_HOST`: 数据库地址
- `DB_PORT`: 数据库端口
- `DB_USER`: 数据库用户名
- `DB_PASSWORD`: 数据库密码
- `DB_NAME`: 数据库名称

### 模型服务
- `OPENAI_BASE_URL`: OpenAI 兼容接口地址
- `OPENAI_API_KEY`: OpenAI 兼容接口 Key
- `LLM_BASE_URL`: LLM 服务地址
- `LLM_API_KEY`: LLM 服务 Key
- `VLM_BASE_URL`: 视觉模型服务地址
- `VLM_API_KEY`: 视觉模型 Key
- `EMBEDDING_BASE_URL`: 向量服务地址
- `EMBEDDING_API_KEY`: 向量服务 Key
- `LLM_MODEL`: 默认对话模型
- `VLM_MODEL`: 默认视觉模型

### 复判 -> 知识库同步
- `REVIEW_KB_ENABLED`: 是否启用复判结果同步
- `REVIEW_KB_BASE_URL`: 知识库服务地址（jetlinks-knowledge）
- `REVIEW_KB_COLLECTION_ID`: 目标集合 ID（默认 video_search）
- `REVIEW_KB_COLLECTION_NAME`: 目标集合名称（自动创建/更新时使用）
- `REVIEW_KB_COLLECTION_TYPE`: 集合类型（text / multimodal）
- `REVIEW_KB_INCLUDE_FAILED`: 是否同步失败复判
- `REVIEW_KB_MIN_HIT`: 最小 hit 阈值（1=仅命中）
- `REVIEW_KB_MAX_PAYLOAD_CHARS`: 单条同步 payload 长度上限

### 工具（按需启用）
- `SERPAPI_KEY`: Google/SerpAPI 搜索 key
- `BAIDU_API_KEY`: 百度搜索 key
- `GOOGLE_SEARCH_ENGINE`: 搜索引擎（google / bing / serpapi）
- `GOOGLE_SEARCH_NUM_RESULTS`: 结果条数
- `GOOGLE_SEARCH_LANGUAGE`: 语言
- `GOOGLE_SEARCH_SAFE_SEARCH`: 安全等级
- `BAIDU_SEARCH_ENGINE`: 百度引擎标识
- `BAIDU_SEARCH_NUM_RESULTS`: 结果条数
- `BAIDU_SEARCH_LANGUAGE`: 语言
- `BAIDU_SEARCH_SAFE_SEARCH`: 安全等级
- `BAIDU_DEFAULT_TIMEOUT`: 请求超时
- `BAIDU_MAX_RESULTS`: 最大总结果
- `USE_SEARCH_PROXY`: 是否启用搜索代理
- `PROXY_HOST`: 代理地址
- `PROXY_PORT`: 代理端口
- `VIDEO_GENERATOR_API_BASE`: 视频生成服务地址
- `VIDEO_GENERATOR_API_KEY`: 视频生成 key
- `VIDEO_GENERATOR_TIMEOUT`: 视频生成超时
- `VIDEO_GENERATOR_MAX_DURATION`: 最大时长（秒）
- `VIDEO_GENERATOR_MAX_KEYFRAMES`: 最大关键帧数
- `VIDEO_GENERATOR_DEFAULT_ASPECT_RATIO`: 默认比例
- `VIDEO_GENERATOR_DEFAULT_IMAGE_SIZE`: 默认尺寸
- `VIDEO_GENERATOR_ENABLE_MULTI_KEYFRAME`: 是否启用多关键帧
- `VIDEO_GENERATOR_DEFAULT_MOTION_STRENGTH`: 运动强度
- `SORA_VIDEO_DURATION`: Sora 视频时长
- `SORA_VIDEO_ASPECT_RATIO`: Sora 视频比例
- `GEMINI_API_KEY`: Gemini 画图 Key
- `GEMINI_IMAGE_SIZE`: Gemini 图片尺寸
- `GEMINI_ASPECT_RATIO`: Gemini 图片比例

```json
{
  "context": {
    "roi": [120, 80, 1400, 900]
  }
}
```

说明：
- 坐标原点在左上角，`rect` 为 `[x1, y1, x2, y2]`
- `normalized=true` 表示 0~1 归一化坐标；不传时会自动判断

### Storage Layout

```
storage/review_records/
  <record_id>/
    record.json
    video.mp4           # optional, recorded via ffmpeg
```

Recorded media is served as static files:

- `GET /storage/review_records/<record_id>/video.mp4`

### Query APIs

- List (latest first): `GET /api/v1/review-records?pageIndex=0&pageSize=50&agentId=<agentId>`
- Detail: `GET /api/v1/review-records/<record_id>`
- `POST /api/v1/agents/{agentId}/chat/json` response includes a top-level `hit` field (0/1, generic match flag).

### review.html

Open `http://localhost:8005/review.html` → **调用记录** → switch to **服务器** mode to view persisted records and play recorded `mp4`.

### Prerequisite

`ffmpeg` must be available in `PATH` (otherwise `video.status=failed` in `record.json`).

### Configuration (.env)

配置文件已移到 `deploy/jetlinks-agent/.env`，可通过 `ENV_FILE` 指定路径（或在部署目录运行）。

```bash
REVIEW_RECORDS_STORAGE_PATH=storage/review_records
# Keep only latest N records (0 = keep all / disable retention cleanup)
REVIEW_RECORDS_MAX_RECORDS=1000
REVIEW_RECORDS_CLIP_SECONDS=30
REVIEW_RECORDS_FFMPEG_BIN=ffmpeg
REVIEW_RECORDS_FFMPEG_TIMEOUT=60
```

### Local ASR（可选）

当业务侧启用 `asr_backend=local` 时，会调用本地模型服务的 `/v1/audio/transcriptions`。

```bash
ASR_API_BASE=http://127.0.0.1:8007/v1
ASR_API_KEY=local
ASR_MODEL=damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
ASR_HTTP_TIMEOUT_SEC=120
```

### Monthly Cleanup (按月清理磁盘)

清理脚本已从源码移出。建议用系统任务定期清理 `storage/review_records` 旧记录。

### Other Disk Cleanup (其他可清理目录)

Video inspection will generate temporary files under `storage/`:

- `storage/out/`（临时切片/中间产物）
- `storage/http_tmp/`（HTTP 离线下载的临时文件）
- `storage/evidence_images/`、`storage/evidence_images_box/`（证据帧）

可用系统任务按需清理以上目录。

Session history storage is under `storage/sessions/` (optional). You can cleanup old files by your own scheduled job.
