# 视频生成工具 README

这是一个基于 Gemini AI 图像生成和 Sora 视频生成的完整视频创作工具，支持文生图、图生图和图生视频功能。

## 📁 项目结构

```
video_generator/
├── gemini_image_generator.py    # Gemini AI 图片生成模块
├── sora.py                     # Sora 视频生成模块
├── video_creation_tool.py      # 统一视频创作工具
└── README.md                   # 本文档
```

## 🚀 核心功能

### 1. **Gemini 图片生成** (`gemini_image_generator.py`)
- **文生图 (Text-to-Image)**: 基于文本描述生成高质量图片
- **图生图 (Image-to-Image)**: 基于参考图片进行风格转换和内容修改
- **自动下载**: 生成的图片自动保存到本地
- **状态监控**: 实时监控生成进度和状态

**核心方法**:
- `text_to_image()`: 文生图功能
- `image_to_image()`: 图生图���能
- `generate_image()`: 统一的图片生成接口
- `download_image()`: 图片下载功能

### 2. **Sora 视频生成** (`sora.py`)
- **图生视频**: 将静态图片转换为动态视频
- **进度监控**: 轮询查询视频生成状态
- **自动下载**: 视频生成完成后自动下载
- **错误处理**: 完善的异常处理和重试机制

**核心方法**:
- `generate_video()`: 提交视频生成请求
- `check_task_status()`: 查询任务状态
- `generate_and_wait()`: 生成并等待完成的完整流程
- `download_video()`: 视频下载功能

### 3. **统一视频创作工具** (`video_creation_tool.py`)
- **多种工作模式**: 支持不同的创作流程
- **多关键帧生成**: 创建复杂动画的关键帧序列
- **智能提示词**: 自动分析转换意图，生成渐进式内容
- **完整工作流**: 从输入到最终视频的端到端处理

**核心功能**:
- `generate_video_from_content()`: 标准视频生成流程
- `generate_multi_keyframe_video()`: 多关键帧动画生成
- `generate_image_only()`: 仅图片生成模式
- `generate_keyframes()`: 智能关键帧生成

## 🛠️ 安装与配置

### 依赖要求
```bash
pip install requests
```

### API 配置
在使用前，需要确保 API 密钥正确配置（代码中已包含授权密钥）

## 📖 使用指南

### 基础用法

#### 1. 文本到视频
```bash
python video_creation_tool.py --prompt "未来科技城市夜景" --duration 15
```

#### 2. 图片到视频
```bash
python video_creation_tool.py \
  --prompt "让汽车变成机器人" \
  --images "https://example.com/car.jpg" \
  --duration 10
```

#### 3. 仅图片生成
```bash
python video_creation_tool.py --prompt "赛博朋克城市" --image-only
```

#### 4. 多关键帧动画
```bash
python video_creation_tool.py \
  --prompt "让汽车逐步变成机器人" \
  --multi-keyframe \
  --keyframes 5 \
  --segment-duration 3
```

### 参数说明

#### 通用参数
- `--prompt`: 文本描述或修改要求 (必需)
- `--images`: 图片URL列表 (可选)
- `--image-size`: 图片分辨率 (1k/2k/4k)，默认2k
- `--aspect-ratio`: 宽高比 (16:9/9:16/1:1/4:3)，默认16:9

#### 视频参数
- `--duration`: 视频时长(秒)，默认15

#### 功能选项
- `--image-only`: 仅生成图片，不生成视频
- `--save-steps`: 保存中间步骤文件
- `--multi-keyframe`: 启用多关键帧模式
- `--keyframes`: 关键帧数量，默认5
- `--segment-duration`: 每个视频片段时长，默认3

## 🎨 工作流程详解

### 1. 标准视频生成流程
```
文本/图片输入 → Gemini生成图片 → Sora生成视频 → 自动下载
```

### 2. 多关键帧动画流程
```
输入分析 → 生成渐进式关键帧 → 每帧生成视频片段 → 合成完整视频
```

### 3. 过渡视频生成
```
原始图片 → Gemini图生图 → Sora生成过渡动画 → 平滑转换效果
```

## 🔧 技术特性

### 智能提示词生成
- 自动识别转换意图（"变成"、"转换"等关键词）
- 生成渐进式描述（0% → 50% → 100%）
- 保持内容连续性和一致性

### 链式图片生成
- 第一帧：文生图模式
- 后续帧：基于前一张的图生图
- 确保视觉连续性

### 状态监控与错误处理
- 实时API状态查询
- 自动重试机制
- 详细的错误信息和调试输出

## 📂 输出文件组织

```
video_generator/
├── downloads/
│   ├── images/         # 生成的图片
│   └── videos/         # 生成的视频
├── workspace/
│   ├── text2img/       # 文生图结果
│   ├── img2img/        # 图生图结果
│   └── final_videos/   # 最终视频
```

## 🎯 使用示例

### 示例1：科幻场景视频
```bash
python video_creation_tool.py \
  --prompt "未来科技城市，飞行器穿梭，霓虹闪烁" \
  --duration 20 \
  --save-steps
```

### 示例2：产品转换动画
```bash
python video_creation_tool.py \
  --prompt "汽车变形为机器人" \
  --images "https://example.com/car.jpg" \
  --multi-keyframe \
  --keyframes 3 \
  --segment-duration 5
```

### 示例3：艺术风格转换
```bash
python video_creation_tool.py \
  --prompt "转换为梵高风格" \
  --images "https://example.com/portrait.jpg" \
  --image-only
```

## ⚠️ 注意事项

### API使用规范
- 遵守相关法律法规和使用条款
- 注意API调用频率限制
- 生成时间与内容复杂度相关

### 内容建议
- 使用清晰、具体的描述
- 避免违规内容生成
- 尊重版权和知识产权

### 性能优化
- 大尺寸生成可能需要较长时间
- 建议先用小尺寸测试效果
- 合理设置视频时长和关键帧数量

## 🐛 故障排除

### 常见问题
1. **API调用失败**: 检查网络连接和API密钥
2. **生成超时**: 降低图片尺寸或视频时长
3. **下载失败**: 检查本地存储空间
4. **内容违规**: 修改提示词内容

### 错误处理
- 自动重试机制
- 详细错误日志
- 友好的状态提示

## 🔄 更新日志

### v1.0.0 (当前版本)
- ✅ 基础文生图/图生图功能
- ✅ 图生视频功能
- ✅ 多关键帧动画生成
- ✅ 智能提示词生成
- ✅ 链式图片生成
- ✅ 完整的错误处理和状态监控

## 📞 技术支持

如遇到问题或有改进建议，请查看相关文档或提交反馈。

---

**注意**: 本工具仅供学习和研究使用，请遵守相关API的使用条款和版权协议。