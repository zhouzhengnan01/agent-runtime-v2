---
name: gpu-training-orchestrator
description: 训练 YOLO（detect/segment）的编排 skill。读取 input.json，自动下载模型权重、本地 COCO 数据划分与转换、conda 环境训练、日志流式输出与结果落盘。
allowed-tools: Read, Write, Shell
---

# gpu-training-orchestrator

## 适用场景

当用户明确要“训练 YOLO 模型”时使用本 skill，尤其是：
- 模型权重需要自动下载（如 `yolov8n.pt`、`yolo11n.pt`）
- 数据集在本地路径
- 除文本目标外参数通过 `input.json` 提供
- 需要在聊天窗口看到训练日志

## 输入约束（input.json）

必填字段：
- `runtime.conda_env_name`
- `dataset.root_dir`（本地路径）
- `dataset.coco_json`（本地 COCO 标注文件）
- `dataset.split.train/val/test`（三者和为 `1.0`）
- `training.task`（`detect` 或 `segment`）
- `training.model`（如 `yolov8n.pt` / `yolo11n.pt`）
- `output.project_dir`
- `output.run_name`

常用可选字段：
- `runtime.enforce_conda_env`（默认 `true`）
- `training.epochs/imgsz/batch/device/workers/patience`
- `dataset.class_names`
- `dataset.copy_images`

## 脚本

- `scripts/run_yolo_training.py`
  - 读取 `input.json`
  - 检查 conda 环境一致性
  - 本地 COCO 转 YOLO 标签
  - 按用户比例切分 train/val/test
  - 自动下载并训练 YOLO
  - 在 test split 上评估
  - 输出 `run_summary.json`

- `scripts/run_in_conda.ps1`
  - 在指定 conda 环境运行训练
  - 前台模式：实时日志直接显示在聊天输出
  - 后台模式：写日志文件并支持 `-Tail` 轮询

## 使用方式

### 前台（推荐：日志直接在聊天输出）

```powershell
powershell -ExecutionPolicy Bypass -File .cursor/skills/gpu-training-orchestrator/scripts/run_in_conda.ps1 -InputJsonPath .cursor/skills/gpu-training-orchestrator/input.json
```

### 后台（长任务）

启动：

```powershell
powershell -ExecutionPolicy Bypass -File .cursor/skills/gpu-training-orchestrator/scripts/run_in_conda.ps1 -InputJsonPath .cursor/skills/gpu-training-orchestrator/input.json -LogFile D:/runs/yolo/myrun/training.log
```

查看最近日志：

```powershell
powershell -ExecutionPolicy Bypass -File .cursor/skills/gpu-training-orchestrator/scripts/run_in_conda.ps1 -LogFile D:/runs/yolo/myrun/training.log -Tail 30
```

若输出含 `[TAIL_DONE]`，表示训练进程已结束。

## 产物

- `<project_dir>/<run_name>/prepared_dataset/`
- `<project_dir>/<run_name>/dataset.yaml`
- `<project_dir>/<run_name>/runs/`
- `<project_dir>/<run_name>/run_summary.json`
