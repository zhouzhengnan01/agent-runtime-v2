"""
工具数据模型
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON
from sqlalchemy.sql import func
from app.db.session import Base


class Tool(Base):
    """工具表"""
    __tablename__ = "tools"

    id = Column(Integer, primary_key=True, index=True, comment="主键ID")
    tool_id = Column(String(100), unique=True, nullable=False, index=True, comment="工具唯一标识")
    name = Column(String(200), nullable=False, comment="工具名称")
    description = Column(Text, comment="工具描述")

    # 输入输出定义 (JetLinks格式)
    inputs = Column(JSON, comment="输入参数定义，JetLinks格式")
    output = Column(JSON, comment="输出定义")

    # 执行配置
    is_async = Column(Boolean, default=False, comment="是否异步执行")
    execution_mode = Column(String(20), default="sync", comment="执行模式：sync(同步)/auto(自动后台执行)")
    estimated_time = Column(Integer, default=3, comment="预估执行时间（秒）")
    timeout = Column(Integer, default=30, comment="执行超时时间（秒）")

    # 管理字段
    source = Column(String(50), default="internal", comment="工具来源：internal/external")
    is_active = Column(Boolean, default=True, comment="是否启用")

    # 扩展字段
    expand = Column(JSON, comment="扩展字段，存储JSON格式的额外配置信息。目前包含: document - 用户对工具使用示例的描述")

    #直接输出字段
    direct_output = Column(Boolean,default=False,comment="是否直接输出")
    # 时间戳
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间")

    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "output": self.output,
            "is_async": self.is_async,
            "execution_mode": self.execution_mode,
            "estimated_time": self.estimated_time,
            "timeout": self.timeout,
            "source": self.source,
            "is_active": self.is_active,
            "expand": self.expand,
            "direct_output":self.direct_output,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    def get_document(self):
        """获取工具使用示例文档"""
        if not self.expand:
            return None
        return self.expand.get("document")

    def set_document(self, document: str):
        """设置工具使用示例文档"""
        if not self.expand:
            self.expand = {}
        self.expand["document"] = document

    def get_expand_field(self, key: str):
        """获取expand中的指定字段"""
        if not self.expand:
            return None
        return self.expand.get(key)

    def set_expand_field(self, key: str, value):
        """设置expand中的指定字段"""
        if not self.expand:
            self.expand = {}
        self.expand[key] = value

    # def get_direct_output(self):
    #     """获取直接输出配置

    #     Returns:
    #         bool: 是否直接输出工具结果，而不让模型总结
    #         True: 直接输出工具原始结果
    #         False: 让模型总结工具结果（默认）
    #     """
    #     return self.get_expand_field("direct_output") or False

    def get_direct_output(self):
        """获取直接输出配置

        Returns:
            bool: 是否直接输出工具结果，而不让模型总结
            True: 直接输出工具原始结果
            False: 让模型总结工具结果（默认）
        """
        # 优先使用数据库字段，如果没有则使用 expand 字段
        if self.direct_output is not None:
            return self.direct_output
        
        # 兼容性：从 expand 字段获取
        return self.get_expand_field("direct_output") or False

    def set_direct_output(self, direct_output: bool):
        """设置直接输出配置

        Args:
            direct_output: 是否直接输出工具结果
        """
        self.set_expand_field("direct_output", direct_output)
    

    # def get_direct_output(self):
    #     """获取直接输出配置
        
    #     Returns:
    #         bool: 是否直接输出工具结果。
    #     """
    #     # 直接返回类的 direct_output 属性
    #     # print("直接输出",self.direct_output)
    #     return self.direct_output

    # def set_direct_output(self, direct_output: bool):
    #     """设置直接输出配置
        
    #     Args:
    #         direct_output: 是否直接输出工具结果
    #     """
    #     # 直接设置类的 direct_output 属性
    #     self.direct_output = direct_output