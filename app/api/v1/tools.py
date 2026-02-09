"""
工具相关API
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.tool import Tool
from app.schemas.tool import ToolCallRequest, ToolInfo, ToolConfigRequest, ToolConfigResponse

router = APIRouter()


def convert_value_type_to_jetlinks_format(inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将简化的 valueType 格式转换为 JetLinks 标准格式

    简化格式: {"id": "name", "valueType": "string", ...}
    JetLinks格式: {"id": "name", "valueType": {"type": "string"}, ...}
    """
    import logging
    logger = logging.getLogger(__name__)

    if not inputs:
        return []

    converted = []
    for input_param in inputs:
        param_copy = input_param.copy()

        # 转换 valueType
        if "valueType" in param_copy:
            value_type = param_copy["valueType"]
            logger.debug(f"转换前: valueType = {value_type}, type = {type(value_type)}")

            # 如果已经是对象格式,不需要转换
            if isinstance(value_type, dict):
                logger.debug(f"已经是对象格式,不转换")
                converted.append(param_copy)
                continue

            # 如果是字符串,转换为对象格式
            if isinstance(value_type, str):
                param_copy["valueType"] = {
                    "type": value_type
                }
                logger.debug(f"转换后: valueType = {param_copy['valueType']}")

                # 如果有 enum 字段,添加 elements 数组（保持原类型）
                if "enum" in param_copy:
                    # 保持原始类型，只添加 elements 数组
                    param_copy["valueType"]["elements"] = [
                        {
                            "value": v,
                            "text": str(v)
                        }
                        for v in param_copy["enum"]
                    ]
                    param_copy["valueType"]["multi"] = False  # 单选模式
                    # 删除顶层的 enum
                    del param_copy["enum"]

                # 如果有 elementType (数组类型),添加到 valueType
                if "elementType" in param_copy:
                    param_copy["valueType"]["elementType"] = param_copy["elementType"]
                    del param_copy["elementType"]

        converted.append(param_copy)

    return converted


@router.get("/list")
async def get_tools_list(include_document: bool = False, db: Session = Depends(get_db)):
    """获取所有可用工具列表"""
    # 从数据库查询所有启用的工具
    tools = db.query(Tool).filter(Tool.is_active == True).all()

    result = []
    for tool in tools:
        # 转换 inputs 为 JetLinks 格式
        converted_inputs = convert_value_type_to_jetlinks_format(tool.inputs or [])

        # 转换 output 为 JetLinks 格式
        output = tool.output or {"type": "object"}

        tool_data = {
            "id": tool.tool_id,
            "name": tool.name,
            "description": tool.description or "",
            "inputs": converted_inputs,
            "output": output,
            "async": tool.is_async,
            "source": tool.source,
            "_converted": True  # 标记：表明经过转换
        }

        # 可选：包含文档信息
        if include_document:
            document = tool.get_document()
            if document:
                tool_data["document"] = document

        result.append(tool_data)

    return result


@router.get("/{tool_id}")
async def get_tool_info(tool_id: str, db: Session = Depends(get_db)):
    """获取指定工具的详细信息"""
    # 从数据库查询
    tool = db.query(Tool).filter(
        Tool.tool_id == tool_id,
        Tool.is_active == True
    ).first()

    if not tool:
        raise HTTPException(status_code=404, detail=f"工具 {tool_id} 不存在")

    # 转换 inputs 为 JetLinks 格式
    converted_inputs = convert_value_type_to_jetlinks_format(tool.inputs or [])

    return {
        "id": tool.tool_id,
        "name": tool.name,
        "description": tool.description or "",
        "inputs": converted_inputs,
        "output": tool.output or {"type": "object"},
        "async": tool.is_async
    }


@router.get("/{tool_id}/config")
async def get_tool_config(tool_id: str, db: Session = Depends(get_db)):
    """获取工具配置"""
    tool = db.query(Tool).filter(Tool.tool_id == tool_id).first()

    if not tool:
        raise HTTPException(status_code=404, detail=f"工具 {tool_id} 不存在")

    # 转换 inputs 为 JetLinks 格式
    converted_inputs = convert_value_type_to_jetlinks_format(tool.inputs or [])

    return ToolConfigResponse(
        tool_id=tool.tool_id,
        name=tool.name,
        description=tool.description or "",
        inputs=converted_inputs,
        output=tool.output or {"type": "object"},
        execution_mode=getattr(tool, 'execution_mode', 'sync'),
        expand=getattr(tool, 'expand', None),
        document=tool.get_document()  # 从expand.document中提取
    )


@router.put("/{tool_id}/config")
async def update_tool_config(tool_id: str, config: ToolConfigRequest, db: Session = Depends(get_db)):
    """更新工具配置"""
    tool = db.query(Tool).filter(Tool.tool_id == tool_id).first()

    if not tool:
        raise HTTPException(status_code=404, detail=f"工具 {tool_id} 不存在")

    # 更新配置
    if config.inputs is not None:
        tool.inputs = config.inputs

    if config.output is not None:
        tool.output = config.output

    if config.execution_mode is not None:
        if hasattr(tool, 'execution_mode'):
            tool.execution_mode = config.execution_mode
        # 如果是 auto 模式，设置为后台执行
        if config.execution_mode == 'auto':
            tool.is_async = True
        else:
            tool.is_async = False

    # 处理expand字段
    if config.expand is not None:
        tool.expand = config.expand

    # 处理document字段（便捷方式）
    if config.document is not None:
        tool.set_document(config.document)

    db.commit()
    db.refresh(tool)

    return {"message": "工具配置更新成功", "tool_id": tool_id}

