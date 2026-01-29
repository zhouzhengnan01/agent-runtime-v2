"""
智能体管理API
"""
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from pydantic import BaseModel
import logging
import time

from app.db.session import get_db
from app.services.agent_service import AgentService
from app.schemas.pagination import PageRequest, StandardResponse
from app.schemas.agent import AgentCreate, AgentChat

router = APIRouter()


def get_agent_service(db: Session = Depends(get_db)) -> AgentService:
    """获取智能体服务实例"""
    return AgentService(db)


@router.post("/agent-templates/list", response_model=StandardResponse)
async def list_agent_templates(
    page_index: Optional[int] = None,
    page_size: Optional[int] = None,
    service: AgentService = Depends(get_agent_service)
):
    """
    查询智能体模板列表
    支持空传参（返回所有）或分页查询

    请求示例：
    - 空body: POST /api/v1/agent/agent-templates/list {}
    - 分页查询: POST /api/v1/agent/agent-templates/list {"pageIndex": 0, "pageSize": 10}
    """
    try:
        # 如果没有分页参数，返回全部数据
        if page_index is None or page_size is None:
            result = await service.get_all_agent_templates()
        else:
            result = await service.get_agent_templates(
                page_index=page_index,
                page_size=page_size
            )
        
        return StandardResponse(
            message="success",
            result=result,
            status=200,
            timestamp=int(time.time() * 1000)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/create", response_model=StandardResponse)
async def create_agent(
    data: AgentCreate,
    service: AgentService = Depends(get_agent_service)
):
    """
    根据模板创建或更新智能体实例
    
    - 如果指定的ID不存在，创建新的智能体
    - 如果指定的ID已存在，更新现有智能体
    """
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        # 记录接收到的数据
        logger.info(f"[API] Received create request with id: {data.id}")
        logger.info(f"[API] Full data: {data.model_dump()}")
        
        # 如果没有template_id，从config中获取type作为template_id
        template_id = data.template_id
        if not template_id and data.config and 'type' in data.config:
            template_id = data.config['type']
        
        # 根据template_id创建对应类型的智能体
        logger.info(f"[API] Calling service.create_agent_from_template with agent_id: {data.id}")
        agent = await service.create_agent_from_template(
            agent_id=data.id,  # 传递用户指定的ID
            template_id=template_id,
            name=data.name,
            created_by=data.created_by,  # 传递创建者ID
            opening_statement=data.opening_statement,
            suggested_questions=data.suggested_questions,
            output_format=data.output_format,
            config=data.config
        )
        logger.info(f"[API] Agent created with id: {agent.get('id')}")
        
        return StandardResponse(
            message="智能体创建成功",
            result=agent,
            status=200,
            timestamp=int(time.time() * 1000)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/list", response_model=StandardResponse)
async def list_agents(
    request: PageRequest,
    service: AgentService = Depends(get_agent_service)
):
    """查询智能体列表"""
    try:
        result = await service.get_agents(
            page_index=request.pageIndex,
            page_size=request.pageSize
        )
        
        return StandardResponse(
            message="success",
            result=result,
            status=200,
            timestamp=int(time.time() * 1000)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=StandardResponse)
async def chat_with_agent(
    request: AgentChat,
    service: AgentService = Depends(get_agent_service)
):
    """与智能体对话（执行任务）"""
    try:
        # 根据agent_id获取智能体
        agent = await service.get_agent_by_id(request.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # 根据智能体类型执行对应的逻辑
        result = await service.execute_agent_task(
            agent_id=request.agent_id,
            message=request.message,
            context=request.context
        )
        
        return StandardResponse(
            message="任务执行成功",
            result=result,
            status=200,
            timestamp=int(time.time() * 1000)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute", response_model=StandardResponse)
async def execute_agent_task(
    request: Dict,
    service: AgentService = Depends(get_agent_service)
):
    """执行智能体任务（通用接口）"""
    try:
        agent_id = request.get("agent_id")
        task_type = request.get("task_type")
        params = request.get("params", {})
        
        # 执行任务
        result = await service.execute_specific_task(
            agent_id=agent_id,
            task_type=task_type,
            params=params
        )
        
        return StandardResponse(
            message="任务执行完成",
            result=result,
            status=200,
            timestamp=int(time.time() * 1000)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates/list")
async def get_templates_list(
    request: Optional[Dict] = None,
    service: AgentService = Depends(get_agent_service)
):
    """获取智能体模板列表 - 支持分页 (从数据库读取最新数据)"""
    try:
        logger = logging.getLogger(__name__)
        logger.info("🔍 调用模板列表接口 (从数据库读取)")

        # 从数据库获取最新的模板信息，不使用缓存
        from app.db.session import SessionLocal
        from app.models.agent_template import AgentTemplate

        db = SessionLocal()
        try:
            # 查询所有启用的模板
            db_templates = db.query(AgentTemplate).filter(
                AgentTemplate.is_active == True
            ).all()

            logger.info(f"📋 从数据库查询到 {len(db_templates)} 个模板")

            # 转换为API格式
            templates = []
            for template in db_templates:
                template_info = {
                    "id": template.template_id or template.id,
                    "type": template.type or template.template_id,
                    "name": template.name,
                    "description": template.description,
                    "icon": "🎬" if "patrol" in template.template_id else "🛠️" if "tool_calling" in template.template_id else "🎥",
                    "category": "patrol" if "patrol" in template.template_id else "tools" if "tool_calling" in template.template_id else "inspection",
                    "framework": template.framework,
                    "capabilities": template.capabilities or [],
                    "tools": template.default_tools or [],
                    "status": "active" if template.is_active else "inactive",
                    "createTime": template.created_at.isoformat() if template.created_at else "2024-01-01T00:00:00Z",
                    "implemented": True
                }
                templates.append(template_info)

            logger.info(f"✅ 成功转换 {len(templates)} 个模板信息")

        finally:
            db.close()

        # 如果没有传参或传了空对象，返回所有数据
        if not request or ("pageIndex" not in request and "pageSize" not in request):
            # 没有分页参数，返回所有数据
            return {
                "status": 200,
                "message": "success",
                "result": templates,
                "timestamp": int(time.time() * 1000)
            }

        # 有分页参数，进行分页处理
        page_index = request.get("pageIndex", 0)
        page_size = request.get("pageSize", 10)

        # 计算分页
        total = len(templates)
        start = page_index * page_size
        end = start + page_size
        paged_templates = templates[start:end]
        
        # 返回分页格式
        return {
            "status": 200,
            "message": "success",
            "result": {
                "pageIndex": page_index,
                "pageSize": page_size,
                "total": total,
                "data": paged_templates
            },
            "timestamp": int(time.time() * 1000)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}", response_model=StandardResponse)
async def get_agent(
    agent_id: str,
    service: AgentService = Depends(get_agent_service)
):
    """获取智能体详情"""
    try:
        agent = await service.get_agent_by_id(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # 转换为字典格式
        agent_dict = {
            "id": agent.id,
            "agent_id": agent.id,
            "name": agent.name,
            "description": agent.description if hasattr(agent, 'description') else "",
            "type": agent.type if hasattr(agent, 'type') else 'general',
            "status": agent.status if hasattr(agent, 'status') else 'active',
            "opening_statement": agent.opening_statement if hasattr(agent, 'opening_statement') else None,
            "suggested_questions": agent.suggested_questions if hasattr(agent, 'suggested_questions') else True,
            "output_format": agent.output_format if hasattr(agent, 'output_format') else "text",
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
            "updated_at": agent.updated_at.isoformat() if agent.updated_at else None
        }
        
        # 从 config 中提取信息
        if agent.config:
            import json
            config = json.loads(agent.config) if isinstance(agent.config, str) else agent.config
            agent_dict["template"] = config.get('template', 'general')
            agent_dict["model"] = config.get('model', 'qwen-max')
            agent_dict["tools"] = config.get('tools', [])
            agent_dict["system_prompt"] = config.get('system_prompt', '')
            agent_dict["user_prompt"] = config.get('user_prompt', '')
            agent_dict["custom_prompt"] = config.get('custom_prompt', '')  # 兼容多种字段名
        
        return StandardResponse(
            message="success",
            result=agent_dict,
            status=200,
            timestamp=int(time.time() * 1000)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{agent_id}", response_model=StandardResponse)
async def delete_agent(
    agent_id: str,
    service: AgentService = Depends(get_agent_service)
):
    """删除智能体实例"""
    try:
        # 删除智能体
        success = service.delete_agent(agent_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return StandardResponse(
            message="删除成功",
            result={"agent_id": agent_id, "deleted": True},
            status=200,
            timestamp=int(time.time() * 1000)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
