"""
智能体模板数据模型
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON
from sqlalchemy.sql import func
from app.db.session import Base
from typing import List, Dict, Any


class AgentTemplate(Base):
    """智能体模板表"""
    __tablename__ = "agent_templates"

    id = Column(String(50), primary_key=True, index=True, comment="主键ID")
    template_id = Column(String(100), unique=True, nullable=True, index=True, comment="模板唯一标识")
    name = Column(String(255), nullable=False, comment="模板名称")
    description = Column(Text, comment="模板描述")
    type = Column(String(50), nullable=True, comment="模板类型")
    class_path = Column(String(500), nullable=False, comment="模板类路径")

    # 配置信息 (JSON格式)
    config = Column(JSON, comment="模板配置")
    default_tools = Column(JSON, comment="默认工具列表")
    default_config = Column(JSON, comment="默认配置")
    capabilities = Column(JSON, comment="能力列表")

    # 状态和管理
    is_active = Column(Boolean, default=True, comment="是否启用")
    framework = Column(String(50), default="qwen-agent", comment="框架类型")

    # 时间戳
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间")

    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "class_path": self.class_path,
            "config": self.config,
            "default_tools": self.default_tools,
            "default_config": self.default_config,
            "capabilities": self.capabilities,
            "is_active": self.is_active,
            "framework": self.framework,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }