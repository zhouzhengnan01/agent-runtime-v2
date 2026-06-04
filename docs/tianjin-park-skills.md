# 天津园区模型原生技能

Runtime 内置 `park-operations` 分类，用于把天津园区运营场景暴露为 Agent Loop 可调用应用和技能。默认不依赖 `/Users/.../tianjin/backend`，技能会使用当前 Runtime 大模型、当前会话文件和本地模板生成可交付成果。

## 默认执行模式

天津园区技能默认 `mode=local`：

- 优先使用当前应用/会话选择的大模型生成 Markdown 成果。
- 大模型不可用时，使用本地确定性模板兜底，仍然写入 `outputs/` 和 `manifest.json`。
- 技能会读取当前 thread 的 `uploads/`、`workspace/`、`outputs/`、`memory/conversation.md` 摘要作为上下文。
- 发布、启动、停止等带副作用的动作仍需要显式 `confirm=true`。

## 可选后端模式

如果确实需要复用天津 FastAPI 后端，可以显式传：

```json
{
  "mode": "backend",
  "api_base_url": "http://127.0.0.1:18080/api/v1"
}
```

也可以用环境变量：

```bash
TIANJIN_SKILL_MODE=backend
TIANJIN_API_BASE_URL=http://127.0.0.1:18080/api/v1
```

后端模式只是兼容集成路径，不是默认路径。

## 应用模板

所有模板都在 `config/apps/`，分类为 `park-operations`：

- `tianjin-park-assistant`：综合园区助手。
- `tianjin-data-analyst`：智能问数。
- `tianjin-business-docs`：商务文档、报价、投标。
- `tianjin-content-ops`：公众号、小红书内容运营。
- `tianjin-public-opinion`：舆情监控与报告。
- `tianjin-meeting-efficiency`：会议、日程、效率培训。
- `tianjin-rpa-ops`：RPA 与系统巡检。

## 技能

每个技能都会在当前 thread 的 `outputs/` 下写入 Markdown 报告和 JSON 结构化结果：

- `tianjin-chatbi-analyst`
- `tianjin-document-generator`
- `tianjin-content-operator`
- `tianjin-sentiment-monitor`
- `tianjin-market-research`
- `tianjin-meeting-minutes`
- `tianjin-calendar-reminder`
- `tianjin-rpa-operator`
- `tianjin-system-monitor`
- `tianjin-efficiency-training`

## 快速验证

这个验证不需要启动天津后端：

```bash
.venv312/bin/python - <<'PY'
from app.core.artifacts import ArtifactStore
from app.core.skills import SkillRunner

store = ArtifactStore()
paths = store.prepare_thread("tianjin-local-smoke")
result = SkillRunner(store).run(
    "tianjin-meeting-minutes",
    {
        "action": "analyze",
        "title": "天津园区例会",
        "transcriptLines": [
            {"speakerName": "张工", "time": "09:30", "text": "张工负责完成园区大屏数据接入。"}
        ],
    },
    paths,
)
print(result.data)
print([artifact.name for artifact in result.outputs])
PY
```
