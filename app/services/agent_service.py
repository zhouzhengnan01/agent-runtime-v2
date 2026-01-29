"""
智能体服务层
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app.models.agent import Agent
from app.models.tool import Tool
from app.exceptions import DatabaseUnavailableError
import logging
import json
import uuid

logger = logging.getLogger(__name__)


class AgentService:
    """智能体服务"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """获取智能体"""
        try:
            # 适配现有数据库结构，使用 id 字段
            agent = self.db.query(Agent).filter(Agent.id == agent_id).first()
            if agent:
                # 将 id 映射到 agent_id 以保持兼容性
                agent.agent_id = agent.id
                # 注意：不需要设置 agent.template，因为 Agent 模型的 @property template 已经映射到 type 字段
                # agent.system_prompt, model, tools 等属性也由 Agent 模型的 @property 自动从 config 读取
            return agent
        except SQLAlchemyError as e:
            logger.error("获取智能体失败（DB不可用）: %s", e, exc_info=True)
            raise DatabaseUnavailableError("数据库繁忙或连接超时，请稍后重试") from e
        except Exception as e:
            logger.error("获取智能体失败: %s", e, exc_info=True)
            raise
    
    def create_agent(self, agent_data: Dict[str, Any]) -> Agent:
        """创建智能体"""
        try:
            agent = Agent(**agent_data)
            self.db.add(agent)
            self.db.commit()
            self.db.refresh(agent)
            return agent
        except Exception as e:
            self.db.rollback()
            logger.error(f"创建智能体失败: {e}")
            raise
    
    def update_agent(self, agent_id: str, update_data: Dict[str, Any]) -> Optional[Agent]:
        """更新智能体"""
        try:
            agent = self.get_agent(agent_id)
            if agent:
                for key, value in update_data.items():
                    if hasattr(agent, key):
                        setattr(agent, key, value)
                self.db.commit()
                self.db.refresh(agent)
            return agent
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新智能体失败: {e}")
            return None
    
    def delete_agent(self, agent_id: str) -> bool:
        """删除智能体"""
        try:
            agent = self.get_agent(agent_id)
            if agent:
                self.db.delete(agent)
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            logger.error(f"删除智能体失败: {e}")
            return False
    
    def list_agents(self, skip: int = 0, limit: int = 100) -> List[Agent]:
        """列出所有智能体"""
        try:
            return self.db.query(Agent).offset(skip).limit(limit).all()
        except Exception as e:
            logger.error(f"列出智能体失败: {e}")
            return []
    
    # 工具管理
    def register_tool(self, tool_data: Dict[str, Any]) -> Tool:
        """注册工具"""
        try:
            tool = Tool(**tool_data)
            self.db.add(tool)
            self.db.commit()
            self.db.refresh(tool)
            return tool
        except Exception as e:
            self.db.rollback()
            logger.error(f"注册工具失败: {e}")
            raise
    
    def get_agent_tools(self, agent_id: str) -> List[Tool]:
        """获取智能体的工具"""
        try:
            # 从智能体配置中获取工具ID列表
            agent = self.get_agent(agent_id)
            if not agent:
                return []
            
            tool_ids = agent.tools if agent.tools else []
            if not tool_ids:
                return []
            
            # 查询工具
            return self.db.query(Tool).filter(Tool.tool_id.in_(tool_ids)).all()
        except Exception as e:
            logger.error(f"获取智能体工具失败: {e}")
            return []
    
    def get_session_tools(self, session_id: str) -> List[Tool]:
        """获取会话级工具"""
        try:
            return self.db.query(Tool).filter(
                Tool.session_id == session_id,
                Tool.is_active == True
            ).all()
        except Exception as e:
            logger.error(f"获取会话工具失败: {e}")
            return []

    # 智能体模板管理
    async def get_all_agent_templates(self) -> List[Dict[str, Any]]:
        """获取所有智能体模板（直接从数据库读取）"""
        try:
            from app.models.agent_template import AgentTemplate

            # 直接从数据库查询所有启用的模板
            db_templates = self.db.query(AgentTemplate).filter(
                AgentTemplate.is_active == True
            ).all()

            logger.info(f"📋 从数据库查询到 {len(db_templates)} 个启用的模板")

            templates_list = []
            for template in db_templates:
                template_info = {
                    "id": template.template_id or template.id,
                    "name": template.name,
                    "description": template.description,
                    "type": template.type or template.template_id,
                    "framework": template.framework,
                    "class_path": template.class_path,
                    "capabilities": template.capabilities or [],
                    "default_tools": template.default_tools or [],
                    "default_config": template.default_config or {},
                    "status": "active" if template.is_active else "inactive",
                    "created_at": template.created_at.isoformat() if template.created_at else None,
                    "updated_at": template.updated_at.isoformat() if template.updated_at else None
                }
                templates_list.append(template_info)

                logger.debug(f"  ✅ 加载模板: {template.name} (ID: {template.template_id})")

            logger.info(f"✅ 成功返回 {len(templates_list)} 个模板")
            return templates_list

        except Exception as e:
            logger.error(f"❌ 从数据库获取模板失败: {e}")
            # 如果数据库查询失败，返回空列表而不是抛出异常
            return []
    
    async def get_agent_templates(self, page_index: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """分页获取智能体模板"""
        all_templates = await self.get_all_agent_templates()
        total = len(all_templates)
        start = (page_index - 1) * page_size
        end = start + page_size
        
        return {
            "total": total,
            "pageIndex": page_index,
            "pageSize": page_size,
            "data": all_templates[start:end]
        }
    
    async def create_agent_from_template(
        self,
        agent_id: str = None,  # 新增参数，允许用户指定ID
        template_id: str = None,
        name: str = None,
        created_by: str = None,  # 创建者ID
        opening_statement: str = None,
        suggested_questions: bool = True,
        output_format: str = "text",
        config: Dict[str, Any] = None
    ) -> Agent:
        """根据模板创建或更新智能体"""
        try:
            logger.info(f"[SERVICE] create_agent_from_template called with agent_id: {agent_id}")
            logger.info(f"[SERVICE] agent_id type: {type(agent_id)}, value: '{agent_id}'")
            
            # 如果用户未指定ID，则自动生成
            if not agent_id:
                agent_id = str(uuid.uuid4())
                logger.info(f"[SERVICE] Generated new agent_id: {agent_id}")
            else:
                logger.info(f"[SERVICE] Using provided agent_id: {agent_id}")
            
            # 构建配置
            full_config = {
                "template": template_id,
                "model": "qwen-max",
                "tools": [],
                "system_prompt": ""
            }
            if config:
                full_config.update(config)
            
            from datetime import datetime

            # 如果用户未指定ID，则自动生成
            if not agent_id:
                agent_id = str(uuid.uuid4())
                logger.info(f"[SERVICE] Generated new agent_id: {agent_id}")

            # 根据 agent_id 检查智能体是否已存在
            existing_agent = self.db.query(Agent).filter(Agent.id == agent_id).first()

            if existing_agent:
                # 如果存在，执行更新操作
                logger.info(f"[SERVICE] Agent {agent_id} already exists, updating...")

                existing_agent.name = name
                existing_agent.description = config.get("description", "") if config else ""
                existing_agent.type = template_id
                existing_agent.config = full_config
                existing_agent.opening_statement = opening_statement
                existing_agent.suggested_questions = suggested_questions
                existing_agent.output_format = output_format
                existing_agent.created_by = created_by
                existing_agent.updated_at = datetime.now()

                self.db.commit()
                self.db.refresh(existing_agent)
                logger.info(f"[SERVICE] Agent {agent_id} updated successfully")

                agent = existing_agent
            else:
                # 如果不存在，创建新智能体
                logger.info(f"[SERVICE] Creating new agent with id={agent_id}")

                agent_data = {
                    "id": agent_id,
                    "name": name,
                    "description": config.get("description", "") if config else "",
                    "type": template_id,
                    "config": full_config,
                    "status": "active",
                    "created_by": created_by,
                    "opening_statement": opening_statement,
                    "suggested_questions": suggested_questions,
                    "output_format": output_format,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }

                agent = Agent(**agent_data)
                self.db.add(agent)
                self.db.commit()
                self.db.refresh(agent)
                logger.info(f"[SERVICE] Agent {agent_id} created successfully")
            
            # 返回字典格式，避免序列化问题
            return {
                "id": agent.id,
                "agent_id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "type": agent.type,
                "opening_statement": agent.opening_statement,
                "suggested_questions": agent.suggested_questions,
                "output_format": agent.output_format,
                "template": template_id,
                "status": agent.status,
                "created_at": agent.created_at.isoformat() if agent.created_at else None,
                "updated_at": agent.updated_at.isoformat() if agent.updated_at else None
            }
        except Exception as e:
            self.db.rollback()
            logger.error(f"创建或更新智能体失败: {e}")
            raise
    
    def list_all_agents(self, skip: int = 0, limit: int = 100) -> List[Agent]:
        """列出所有智能体"""
        try:
            agents = self.db.query(Agent).offset(skip).limit(limit).all()
            # 为每个智能体设置兼容属性
            for agent in agents:
                agent.agent_id = agent.id
                # 注意：agent.template 由 Agent 模型的 @property 自动映射到 type 字段
            return agents
        except Exception as e:
            logger.error(f"列出智能体失败: {e}")
            return []
    
    async def get_agents(self, page_index: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """分页获取智能体列表"""
        try:
            total = self.db.query(Agent).count()
            # 修复分页逻辑：page_index从0开始
            skip = page_index * page_size if page_index >= 0 else 0
            agents = self.list_all_agents(skip=skip, limit=page_size)
            
            # 转换为字典格式
            agents_data = []
            for agent in agents:
                agent_dict = {
                    "id": agent.id,
                    "agent_id": agent.id,
                    "name": agent.name,
                    "description": agent.description,
                    "type": agent.type if hasattr(agent, 'type') else 'general',
                    "status": agent.status if hasattr(agent, 'status') else 'active',
                    "created_at": agent.created_at.isoformat() if agent.created_at else None,
                    "updated_at": agent.updated_at.isoformat() if agent.updated_at else None
                }
                
                # 从 config 中提取模板信息
                if agent.config:
                    config = json.loads(agent.config) if isinstance(agent.config, str) else agent.config
                    agent_dict["template"] = config.get('template', 'general')
                    agent_dict["model"] = config.get('model', 'qwen-max')
                
                agents_data.append(agent_dict)
            
            return {
                "total": total,
                "pageIndex": page_index,
                "pageSize": page_size,
                "data": agents_data
            }
        except Exception as e:
            logger.error(f"获取智能体列表失败: {e}")
            return {
                "total": 0,
                "pageIndex": page_index,
                "pageSize": page_size,
                "data": []
            }
    
    async def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """根据ID获取智能体"""
        return self.get_agent(agent_id)
    
    async def execute_agent_task(self, agent_id: str, message: str, context: Dict = None) -> Dict[str, Any]:
        """执行智能体任务"""
        # 这里可以调用JAIPHandler或直接使用工厂创建智能体
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"智能体 {agent_id} 不存在")
        
        # 返回简单结果
        return {
            "agent_id": agent_id,
            "message": message,
            "response": f"已接收任务: {message}",
            "status": "completed"
        }
    
    async def execute_specific_task(self, agent_id: str, task_type: str, params: Dict) -> Dict[str, Any]:
        """执行特定任务"""
        return {
            "agent_id": agent_id,
            "task_type": task_type,
            "params": params,
            "status": "completed",
            "result": f"任务 {task_type} 执行完成"
        }
