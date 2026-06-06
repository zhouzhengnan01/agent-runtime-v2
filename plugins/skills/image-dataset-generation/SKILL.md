---
name: image-dataset-generation
description: >
  通过本地 image-gen HTTP API（例如 Flux），根据参考图片和文本提示词生成合成图片。
  当用户提供参考图片路径和提示词，并希望创建新图片、执行风格迁移或生成训练数据集图片时使用。
  触发场景包括：基于参考图生成图片、本地 image-gen API 调用、结合提示词生成合成图片。
---

# 图片数据集生成

通过调用本地 image-gen HTTP API（例如 Flux），根据本地参考图片和文本提示词生成图片。

## 工作流程

1. 从聊天输入中解析用户提供的参数。
2. 将 `input.json` 写入该 skill 的工作目录。
3. 校验参数（必填字段、路径是否存在）。
4. 执行 `scripts/run_generation.py`，由该脚本调用 image-gen API。
5. 向用户报告结果路径和摘要信息。

## 参数

### 必填
- **input_image**：参考图片的本地绝对路径。
- **prompt**：期望生成效果的文本描述（支持中文和英文）。
- **output_dir**：保存生成图片的本地绝对路径目录。

### 可选（带默认值）
- **api_url**：`http://218.67.242.10:58801/v1/flux2/generate`
- **token**：API 的 Bearer token（无需包含 `Bearer ` 前缀）。
- **model**：默认 `flux2`。
- **width/height/steps**：默认 `640/640/8`。
- **timeout**：请求超时时间，单位为秒（默认 `120`）。

## input.json 结构

运行脚本前先写入该文件。脚本会通过 `--input` 读取它。

```json
{
  "model": {
    "api_url": "http://218.67.242.10:58801/v1/flux2/generate",
    "token": "abc@123",
    "timeout": 120
  },
  "task": {
    "input_image": "D:/path/to/ref.jpg",
    "prompt": "Turn the cat into an astronaut on the moon",
    "output_dir": "D:/path/to/outputs"
  }
}
```

## 执行方式

```bash
python <skill-dir>/scripts/run_generation.py --input <path-to-input.json>
```

该脚本会在内部调用参考实现项目根目录下的 `image-genereate.py`。如果你的 `image-genereate.py` 位于其他位置，请调整 `run_generation.py` 中的 `IMAGE_GENERATE_SCRIPT`。

## 约束

- `input_image` 必须是**本地绝对路径**。
- `output_dir` 必须是**本地绝对路径**。
- `prompt` 不能为空。
- 不要修改 `image-genereate.py` 的核心逻辑；应通过 CLI 参数驱动行为。
- 如果失败（文件缺失、HTTP 错误、返回内容不是图片），应以错误状态退出，并给出排查建议。

## 输出摘要

执行完成后，报告：

- api_url
- input_image
- prompt
- output_dir
- result_image_path
- elapsed_seconds
- status: success / failed
- error_message（仅失败时）

## 故障排查

- **找不到输入图片**：确认 `task.input_image` 是绝对路径，且没有拼写错误。
- **HTTP 非 200 状态码**：检查 `model.api_url` 和 `model.token`。查看 `output_dir` 中的 `error_response_*.raw`。
- **返回内容不是图片格式**：查看 `output_dir` 中的 `unknown_response_*.raw`，排查服务端错误。
- **未指定输出目录**：用户必须提供 `task.output_dir` 作为必填参数。
