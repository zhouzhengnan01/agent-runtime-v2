"""
工具相关的数据模式定义
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class ToolCallRequest(BaseModel):
    """工具调用请求"""
    tool_name: str = Field(..., description="工具名称")
    parameters: Dict[str, Any] = Field(..., description="调用参数")


class ToolInfo(BaseModel):
    """工具信息"""
    id: str = Field(..., description="工具ID")
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    inputs: List[Dict[str, Any]] = Field(..., description="输入参数定义")
    output: Dict[str, Any] = Field(..., description="输出定义")
    async_: bool = Field(False, description="是否异步执行")

    class Config:
        fields = {"async_": "async"}


class ToolConfigRequest(BaseModel):
    """工具配置更新请求"""
    inputs: Optional[List[Dict[str, Any]]] = Field(None, description="输入参数定义，JetLinks格式")
    output: Optional[Dict[str, Any]] = Field(None, description="输出定义")
    execution_mode: Optional[str] = Field(None, description="执行模式：sync(同步)/auto(自动后台执行)")
    expand: Optional[Dict[str, Any]] = Field(None, description="扩展字段，存储JSON格式的额外配置信息")
    document: Optional[str] = Field(None, description="工具使用示例文档（便捷字段，会存储到expand.document）")


class ToolConfigResponse(BaseModel):
    """工具配置响应"""
    tool_id: str = Field(..., description="工具ID")
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    inputs: List[Dict[str, Any]] = Field(..., description="输入参数定义，JetLinks格式")
    output: Dict[str, Any] = Field(..., description="输出定义")
    execution_mode: str = Field(..., description="执行模式：sync/auto")
    expand: Optional[Dict[str, Any]] = Field(None, description="扩展字段")
    document: Optional[str] = Field(None, description="工具使用示例文档（从expand.document中提取）")