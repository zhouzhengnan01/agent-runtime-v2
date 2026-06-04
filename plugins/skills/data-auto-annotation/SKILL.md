---
name: data-auto-annotation
description: Use SAM3 to auto-annotate a single image, export standard COCO JSON, and verify the generated dataset. Use when the user uploads an image and asks for automatic labeling or COCO output.
compatibility: Designed for Python 3 with network access to a SAM3 HTTP endpoint.
metadata:
  author: jetlinks
  version: "1.0"
  language: zh-CN
---

tools:
  - data_auto_annotation

tags:
  - sam3
  - coco
  - annotation
  - image
  - dataset

# 何时使用

- 用户上传了一张图片，希望自动标注并导出 COCO
- 用户要把图片转成目标检测数据集的预标注结果
- 用户要生成标准 COCO `json` 文件

# 工具输入

- `image_path`：本地图片路径；运行时应传入附件落盘后的真实路径
- `labels`：需要标注的类别，例如 `person`
- `format`：输出格式，当前固定为 `coco`

# 推荐工作流

1. 读取聊天附件中的图片路径，优先使用 `attachments[0].path`。
2. 如果运行时只提供了 `data_base64`，先写入临时文件，再得到本地路径。
3. 将本地路径作为 `image_path` 传给 `data_auto_annotation` 工具。
4. 工具内部调用 `scripts/sam3-predict.py` 生成 COCO JSON。
5. 将 COCO 文件路径或 COCO 内容返回给上层 runtime。

# 入口脚本

- `scripts/sam3-predict.py`

# 使用方式

- 直接调用脚本：
  - `python plugins/skills/data-auto-annotation/scripts/sam3-predict.py --input-dir <image_dir> --text-prompts person --output coco.json`
- 通过运行时工具调用时，参数应映射为：
  - `image_path`（目录）-> `--input-dir`
  - `image_path`（单图/混合）-> `--input-json`
  - `labels` -> `--text-prompts`
  - `format=coco` -> 固定输出 COCO

# 输出要求

输出必须是标准 COCO 结构，至少包含：

- `images`
- `annotations`
- `categories`
- `licenses`
- `info`

并确保：

- `images` 中保存图片的 `id`、`file_name`、`width`、`height`
- `annotations` 中保存每个目标的 `image_id`、`category_id`、`bbox`、`area`、`iscrowd`
- `bbox` 使用 `[x, y, width, height]`

# 依赖

- Python 3
- `requests`
- `Pillow`

安装命令：

```bash
pip install -r plugins/skills/data-auto-annotation/requirements.txt
```

# 注意事项

- 不要让模型自己伪造 `<tool_code>`；必须走 runtime 的标准 `tool_calls`
- 不要依赖用户上传时看到的文件名，必须使用附件真实路径
- 目前该 skill 只负责自动标注与 COCO 导出，不负责图像可视化

# 备注

这个技能的目标是把单张图片快速转换成可用的 COCO 标注数据，并让 runtime 能把它当成真正可执行的工具来调用。

## Runtime Input Mapping (Updated)

- `image_path`: can be a single image file path, or a folder path containing images.
- `labels`: class labels, mapped to `--text-prompts`.
- `format`: keep `coco`.

CLI mapping:

- `image_path` (folder) -> `--input-dir`
- `image_path` (single file or mixed paths) -> `--input-json`
- `labels` -> `--text-prompts`
- `format=coco` -> COCO JSON output

Example CLI:

```bash
python scripts/sam3-predict.py --input-dir D:/dataset/images --text-prompts person car --output D:/dataset/coco.json
```
