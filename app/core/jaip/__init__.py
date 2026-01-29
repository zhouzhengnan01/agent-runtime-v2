"""
JetLink AI Interactive Protocol (JAIP) v2.0 实现

JAIP是JetLinks系统的统一AI交互协议，定义了前端、后端和AI服务之间的通信规范。

基于 LangChain 框架实现，使用 WebSocket JSON-RPC 2.0 协议。
"""

from app.core.jaip.handler import JAIPHandler

__all__ = [
    "JAIPHandler"
]

__version__ = "3.0.0"  # 移除 qwen-agent handler 和 protocol 数据模型