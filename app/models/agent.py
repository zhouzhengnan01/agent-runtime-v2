"""
智能体数据模型
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON
from sqlalchemy.sql import func
from app.db.session import Base
import json
from typing import List, Dict, Any, Optional


class Agent(Base):
    """智能体表"""
    __tablename__ = "agents"

    id = Column(String(50), primary_key=True, index=True, comment="主键ID")
    name = Column(String(200), nullable=False, comment="智能体名称")
    description = Column(Text, comment="智能体描述")
    type = Column(String(50), nullable=False, comment="智能体类型/模板")

    # 配置信息 (JSON格式)
    config = Column(JSON, comment="智能体配置，包含model、tools、system_prompt等")

    # 状态和管理
    status = Column(String(20), default="active", comment="状态：active/inactive")
    created_by = Column(String(100), comment="创建者")

    # 输出格式配置
    output_format = Column(String(20), default="text", comment="输出格式：text/json")
    json_schema = Column(JSON, comment="JSON输出格式定义")

    # 界面配置
    opening_statement = Column(Text, comment="开场白")
    suggested_questions = Column(Boolean, default=True, comment="是否启用问题建议")

    # 时间戳
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间")

    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "config": self.config,
            "status": self.status,
            "created_by": self.created_by,
            "output_format": self.output_format,
            "json_schema": self.json_schema,
            "opening_statement": self.opening_statement,
            "suggested_questions": self.suggested_questions,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    # 配置字段的便捷访问器 (property decorators)
    @property
    def system_prompt(self) -> str:
        """获取系统提示"""
        if not self.config:
            return ""
        return self.config.get("system_prompt", "")

    @property
    def model(self) -> str:
        """获取模型名称"""
        if not self.config:
            return "qwen-turbo"
        return self.config.get("model", "qwen-turbo")

    @property
    def temperature(self) -> float:
        """获取温度参数"""
        if not self.config:
            return 0.3
        return self.config.get("temperature", 0.3)

    @property
    def max_tokens(self) -> int:
        """获取最大token数"""
        if not self.config:
            return 2048
        return self.config.get("max_tokens", 2048)

    @property
    def tools(self) -> List[str]:
        """获取工具列表"""
        if not self.config:
            return []
        return self.config.get("tools", [])

    @property
    def user_prompt(self) -> str:
        """获取用户提示"""
        if not self.config:
            return ""
        return self.config.get("user_prompt", "")

    @property
    def template(self) -> str:
        """获取模板类型（兼容性属性）"""
        return self.type

    # 配置管理方法
    def update_config(self, key: str, value: Any):
        """更新配置项"""
        if not self.config:
            self.config = {}
        self.config[key] = value

    def get_config(self, key: str, default: Any = None):
        """获取配置项"""
        if not self.config:
            return default
        return self.config.get(key, default)