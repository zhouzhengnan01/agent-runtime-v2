"""
Schema 查询 API - 查询智能体模板支持的 JSON 返回结构
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# ========== 请求/响应模型 ==========

class SchemaQueryRequest(BaseModel):
    """Schema 查询请求"""
    template_type: str  # 智能体模板类型（对应agents表的type字段）
    schema_name: Optional[str] = None
    type: Optional[str] = None  # schema的类型（simple/detail）

    class Config:
        json_schema_extra = {
            "example": {
                "template_type": "video_inspection",
                "schema_name": "videoInspection",
                "type": "simple"
            }
        }


class TemplateListRequest(BaseModel):
    """模板列表查询请求"""
    pass  # 暂时不需要参数


@router.get("/template/{template_type}/schemas")
async def get_template_schemas(
    template_type: str,
    schema_name: Optional[str] = Query(None, description="指定schema名称，如果不提供则返回所有"),
    type: Optional[str] = Query(None, description="指定类型(simple/detail)，如果不提供则返回所有类型")
):
    """
    查询智能体模板支持的 JSON 返回结构

    参数：
      - template_type: 模板类型（对应agents表的type字段），如 "video_inspection"
      - schema_name: 可选，指定schema名称，如 "videoInspection"
      - type: 可选，指定类型 "simple" 或 "detail"

    示例：
      GET /api/v1/schemas/template/video_inspection/schemas
      GET /api/v1/schemas/template/video_inspection/schemas?schema_name=videoInspection
      GET /api/v1/schemas/template/video_inspection/schemas?schema_name=videoInspection&type=simple

    返回格式：
    {
      "template": "video_inspection",
      "templateDisplayName": "视频巡检模板",
      "schemas": [...] 或 "schema": {...}
    }
    """
    try:
        # 根据模板类型创建临时智能体实例
        agent_instance = _create_agent_from_template(template_type)

        # 检查是否支持 schema
        if not hasattr(agent_instance, 'get_supported_schemas'):
            raise HTTPException(
                status_code=400,
                detail=f"模板 '{template_type}' 不支持 Schema-Based 响应"
            )

        # 获取所有支持的 schemas
        all_schemas = agent_instance.get_supported_schemas()

        if not all_schemas:
            return {
                "template": template_type,
                "templateDisplayName": _get_template_display_name(template_type),
                "schemas": []
            }

        # 如果指定了 schema_name，只返回该 schema
        if schema_name:
            if schema_name not in all_schemas:
                raise HTTPException(
                    status_code=404,
                    detail=f"Schema '{schema_name}' 在模板 '{template_type}' 中不存在"
                )

            schema_def = all_schemas[schema_name]

            # 如果还指定了 type，只返回该类型
            if type:
                if type not in schema_def.get("types", {}):
                    raise HTTPException(
                        status_code=404,
                        detail=f"类型 '{type}' 在 schema '{schema_name}' 中不存在"
                    )

                # 返回单个类型
                type_def = schema_def["types"][type]
                return {
                    "template": template_type,
                    "templateDisplayName": _get_template_display_name(template_type),
                    "schema": {
                        "name": schema_name,
                        "displayName": schema_def.get("name"),
                        "description": schema_def.get("description"),
                        "type": type,
                        "typeDescription": type_def.get("description"),
                        "structure": type_def.get("structure"),
                        "example": type_def.get("example")
                    }
                }

            # 返回该 schema 的所有类型
            formatted_schema = _format_schema_with_types(schema_name, schema_def)
            return {
                "template": template_type,
                "templateDisplayName": _get_template_display_name(template_type),
                "schema": formatted_schema
            }

        # 返回所有 schemas
        formatted_schemas = []
        for name, schema_def in all_schemas.items():
            formatted_schemas.append(_format_schema_with_types(name, schema_def))

        return {
            "template": template_type,
            "templateDisplayName": _get_template_display_name(template_type),
            "schemas": formatted_schemas
        }

    except HTTPException:
        raise
    except ImportError as e:
        logger.error(f"模板加载失败: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"模板 '{template_type}' 不存在或无法加载"
        )
    except Exception as e:
        logger.error(f"获取模板 schema 失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"获取模板 schema 失败: {str(e)}"
        )


@router.post("/query")
async def query_schemas(request: SchemaQueryRequest = Body(...)):
    """
    POST 方式查询智能体模板支持的 JSON 返回结构

    请求体：
    {
      "template_name": "video_inspection",
      "schema_name": "videoInspection",  // 可选
      "type": "simple"  // 可选
    }

    示例1 - 查询所有 schemas：
    POST /api/v1/schemas/query
    {
      "template_name": "video_inspection"
    }

    示例2 - 查询特定 schema：
    POST /api/v1/schemas/query
    {
      "template_name": "video_inspection",
      "schema_name": "videoInspection"
    }

    示例3 - 查询特定类型：
    POST /api/v1/schemas/query
    {
      "template_name": "video_inspection",
      "schema_name": "videoInspection",
      "type": "simple"
    }
    """
    return await get_template_schemas(
        template_type=request.template_type,
        schema_name=request.schema_name,
        type=request.type
    )


@router.post("/templates/list")
async def list_available_templates(request: TemplateListRequest = Body(...)):
    """
    列出所有支持 Schema-Based Response 的模板

    请求体：
    {}

    返回示例：
    {
      "templates": [
        {
          "name": "video_inspection",
          "displayName": "视频巡检模板",
          "description": "具备完整认知能力的视频安全巡检智能体",
          "supportsSchemas": true,
          "availableSchemas": ["videoInspection", "correctnessCheck", "safetyDetection"]
        }
      ]
    }
    """
    templates = []

    # 模板映射（与 _create_agent_from_template 保持一致）
    template_configs = {
        "video_inspection": {
            "displayName": "视频巡检模板",
            "description": "具备完整认知能力的视频安全巡检智能体",
            "module": "app.core.agents.template_agent.video_inspection_template.VideoInspectionAgent"
        },
        "tool_calling": {
            "displayName": "工具调用模板",
            "description": "基于LangChain的工具调用智能体，支持异步执行和工具确认",
            "module": "app.core.agents.template_agent.tool_calling_template.ToolCallingAgent"
        },
        # 可以添加更多模板
    }

    for template_type, config in template_configs.items():
        try:
            # 尝试创建实例检查是否支持 schema
            agent_instance = _create_agent_from_template(template_type)

            template_info = {
                "name": template_type,
                "displayName": config["displayName"],
                "description": config["description"],
                "supportsSchemas": hasattr(agent_instance, 'get_supported_schemas'),
                "availableSchemas": []
            }

            # 如果支持 schema，获取列表
            if template_info["supportsSchemas"]:
                schemas = agent_instance.get_supported_schemas()
                template_info["availableSchemas"] = list(schemas.keys())

            templates.append(template_info)

        except Exception as e:
            logger.warning(f"加载模板 {template_type} 失败: {e}")
            continue

    return {
        "templates": templates,
        "total": len(templates)
    }


def _create_agent_from_template(template_type: str):
    """根据模板名称创建临时智能体实例"""
    # 模板映射
    template_map = {
        "video_inspection": "app.core.agents.template_agent.video_inspection_template.VideoInspectionAgent",
        "tool_calling": "app.core.agents.template_agent.tool_calling_template.ToolCallingAgent",
        # 可以添加更多模板
    }

    if template_type not in template_map:
        raise ImportError(f"Unknown template: {template_type}")

    # 动态导入
    module_path, class_name = template_map[template_type].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    agent_class = getattr(module, class_name)

    # 创建实例（使用默认配置）
    # LangChain 模板需要传入 config 参数
    try:
        # 尝试传入空配置（适用于 LangChain 模板）
        return agent_class(config={})
    except TypeError:
        # 如果不接受 config 参数，尝试无参数调用（适用于旧模板）
        return agent_class()


def _get_template_display_name(template_type: str) -> str:
    """获取模板显示名称"""
    display_names = {
        "video_inspection": "视频巡检模板",
        "tool_calling": "工具调用模板",
        "general": "通用对话模板",
        "safety_monitor": "安全监控模板"
    }
    return display_names.get(template_type, template_type)


def _format_schema_with_types(schema_name: str, schema_def: Dict) -> Dict:
    """格式化 schema 定义，将 types 转换为数组格式"""
    types_list = []

    for type_name, type_def in schema_def.get("types", {}).items():
        types_list.append({
            "type": type_name,
            "description": type_def.get("description"),
            "structure": type_def.get("structure"),
            "example": type_def.get("example")
        })

    return {
        "name": schema_name,
        "displayName": schema_def.get("name"),
        "description": schema_def.get("description"),
        "types": types_list
    }