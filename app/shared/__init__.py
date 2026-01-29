"""
JetLinks Agent 共享组件模块

包含可在多个模块间复用的组件：
- jetlinks_video/: JetLinksAI原始代码（可整体更新）
- simple_streaming_api: 简化的流式视频分析接口

使用示例:
    from app.shared.simple_streaming_api import start_offline_analysis
    task_id = start_offline_analysis("video.mp4")
"""

__all__ = []