"""
分页相关的数据模式定义
"""
from typing import List, Dict, Optional, Any, Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar('T')


class SortItem(BaseModel):
    """排序项"""
    name: str = Field(..., description="字段名")
    order: str = Field("desc", description="排序方向: asc/desc")
    value: Optional[Any] = Field(None, description="排序值")


class TermItem(BaseModel):
    """查询条件项"""
    column: str = Field(..., description="字段名")
    value: Any = Field(..., description="值")
    termType: str = Field("eq", description="查询类型: eq/like/gt/gte/lt/lte/in/not")


class PageRequest(BaseModel):
    """分页请求"""
    pageIndex: int = Field(0, description="页索引")
    pageSize: int = Field(12, description="页大小")
    sorts: List[SortItem] = Field(default=[], description="排序条件")
    terms: List[TermItem] = Field(default=[], description="查询条件")


class PageResponse(BaseModel, Generic[T]):
    """分页响应"""
    pageIndex: int = Field(..., description="页索引")
    pageSize: int = Field(..., description="页大小")
    total: int = Field(..., description="总数")
    data: List[T] = Field(..., description="数据列表")


class StandardResponse(BaseModel):
    """标准响应"""
    message: str = Field("success", description="消息")
    result: Any = Field(None, description="结果数据")
    status: int = Field(200, description="状态码")
    timestamp: int = Field(..., description="时间戳")