"""
智能体相关的数据模式定义
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class AgentCreate(BaseModel):
    """创建智能体"""
    id: Optional[str] = Field(default=None, description="智能体ID，不传则自动生成")
    template_id: Optional[str] = Field(default=None, description="模板ID，不传则从config.type获取")
    name: str = Field(..., description="智能体名称")
    created_by: Optional[str] = Field(default=None, description="创建者ID")
    opening_statement: Optional[str] = Field(default=None, description="开场白")
    suggested_questions: Optional[bool] = Field(default=True, description="是否启用用户问题建议")
    output_format: Optional[str] = Field(default="markdown", description="内容输出格式：markdown/json")
    json_schema: Optional[Dict] = Field(default=None, description="JSON输出格式定义，仅当output_format='json'时生效")
    config: Optional[Dict] = Field(default={}, description="自定义配置")


class AgentChat(BaseModel):
    """智能体对话/执行"""
    agent_id: str = Field(..., description="智能体ID")
    message: str = Field(..., description="消息/指令")
    context: Optional[Dict] = Field(default={}, description="上下文信息")


class AgentExecute(BaseModel):
    """执行智能体任务"""
    agent_id: str = Field(..., description="智能体ID")
    task_type: str = Field(..., description="任务类型")
    params: Dict = Field(default={}, description="任务参数")


class VideoInspectionParams(BaseModel):
    """视频巡检参数"""
    targets: List[Dict] = Field(..., description="巡检目标列表")
    rules: List[Dict] = Field(default=[], description="巡检规则")
    schedule: Optional[Dict] = Field(default=None, description="调度配置")


class AgentQuery(BaseModel):
    """查询智能体列表"""
    pageIndex: int = Field(default=0, description="页码，从0开始")
    pageSize: int = Field(default=25, description="每页数量")
    terms: Optional[List[Dict]] = Field(default=None, description="查询条件")


class AgentResponse(BaseModel):
    """智能体响应"""
    agent_id: str
    type: str  # inspection_result, chat_response, report, etc.
    result: Any
    timestamp: datetime
    message: Optional[str] = None