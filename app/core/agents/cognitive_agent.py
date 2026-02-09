"""
LangChain认知智能体 - 核心执行引擎

架构特点：
- 使用 IntelligentPlanner 进行智能工具选择和任务规划
- 使用 parameter_filler + tool_router 替代 LangChain AgentExecutor
- 完全异步设计，不阻塞事件循环
- 支持流式输出和立即计划发送
- 可选的Redis对话记忆

主要方法：
- _run_stream(): 核心流式执行方法
- _intelligent_tool_selection(): 使用独立计划器进行工具选择
- _execute_single_tool(): 执行单个工具
- _execute_tool_pipeline(): 执行工具流水线
"""

from typing import Dict, Any, List, Optional, AsyncIterator
import logging
import os
import json
import re
import time
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.tools.tool_router import ToolRouter
from app.core.tools.internal_tool_registry import internal_tool_registry
from app.core.memory.redis_chat_memory import redis_memory_manager
from app.services.tool_config_service import tool_config_service
from app.services.session_markdown_logger import SessionMarkdownLogger
from app.core.agents.planning import IntelligentPlanner
from app.config import settings


logger = logging.getLogger(__name__)


class ToolPipelineAbort(RuntimeError):
    """Abort the current tool pipeline with a user-facing reason."""

    def __init__(self, message: str, *, tool_name: Optional[str] = None, missing_required: Optional[List[str]] = None):
        super().__init__(message)
        self.tool_name = tool_name
        self.missing_required = list(missing_required or [])


class LangChainCognitiveAgent:
    """LangChain认知智能体 - 负责LLM调用、任务规划、工具编排和记忆管理"""

    @staticmethod
    def _display_tool_name(tool_name: Optional[str]) -> str:
        """Return tool name as-is (keep prefixes like `_ai_service_#`)."""
        if not tool_name:
            return ""
        return str(tool_name)

    def _display_tool_label(self, tool_name: Optional[str]) -> str:
        """Best-effort tool label for UI/logs: '<display_name> (tool_id)'."""
        tool_id = self._display_tool_name(tool_name)
        if not tool_id:
            return ""

        display_name = None
        description = ""
        try:
            for tool in self.function_list or []:
                if str(getattr(tool, "name", "") or "") != str(tool_name or ""):
                    continue
                display_name = getattr(tool, "display_name", None)
                description = (getattr(tool, "description", "") or "").strip()
                if not display_name and isinstance(getattr(tool, "tool_def", None), dict):
                    tool_def = getattr(tool, "tool_def") or {}
                    display_name = (
                        tool_def.get("name")
                        or tool_def.get("displayName")
                        or tool_def.get("display_name")
                        or tool_def.get("title")
                    )
                    if not description:
                        description = (tool_def.get("description") or "").strip()
                break
        except Exception:
            display_name = None

        def _looks_like_tool_id(value: str) -> bool:
            v = (value or "").strip()
            if not v:
                return False
            # e.g. visualizationService:project#bgr39i / deviceService:device#Query / knowledgeService#Search
            if "#" in v and ":" in v and " " not in v:
                return True
            if v.startswith("_ai_service_#"):
                return True
            # Keep it conservative: treat "xxx#yyy" without spaces as tool id.
            if "#" in v and " " not in v:
                return True
            return False

        def _has_chinese(value: str) -> bool:
            return bool(re.search(r"[\u4e00-\u9fff]", value or ""))

        def _extract_display_from_description(desc: str) -> Optional[str]:
            if not desc:
                return None
            # Java 侧 ToolInfo.description 常为：toolsDescription,metadata.name,metadata.description
            parts = [p.strip() for p in re.split(r"[，,]", desc) if p and p.strip()]
            if not parts:
                return None
            if len(parts) >= 2:
                return parts[1]
            return parts[0]

        if display_name:
            display_name = str(display_name).strip()

        # 1) Prefer explicit display_name when it looks like a human label (esp. Chinese)
        if display_name and display_name != tool_id and not _looks_like_tool_id(display_name):
            return display_name

        # 2) Fallback: derive a Chinese label from tool.description (common for platform-provided tools)
        derived = _extract_display_from_description(description)
        if derived:
            derived = str(derived).strip()
        if derived and derived != tool_id and _has_chinese(derived):
            return derived

        # 3) Last resort: show the tool id to avoid blank UI.
        return tool_id

    def __init__(self, config: Dict[str, Any]):
        """初始化认知智能体"""
        import os
        # 基础配置
        self.name = config.get("name", "LangChain智能体")
        self.model = config.get("model", os.getenv("LLM_MODEL", "qwen-turbo"))
        self.model = self._normalize_configured_model(self.model)
        self.system_prompt = config.get("system_prompt", "")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 2000)
        self.init_parameters = config.get("init_parameters", {})

        # 设置logger
        self.logger = logger

        # 工具列表
        self.function_list = config.get("tools", [])
        # 工具返回值
        self.tool_res = {}

        # JSON处理器特定配置
        self.auto_confirm_tools = config.get("auto_confirm_tools", False)
        self.disable_tools = config.get("disable_tools", False)

        # 🆕 初始化参数填充器实例（延迟初始化）
        self.parameter_filler = None
        logger.info("🔧 ParameterFiller 将在首次使用时初始化")

        # 认知能力
        self.enable_memory = config.get("enable_memory", False)
        self.enable_compression = config.get("enable_compression", False)
        self.enable_multimodal = config.get("enable_multimodal", False)

        # 会话和记忆
        self.session_id = config.get("session_id", None)
        self.agent_id = config.get("agent_id", None)
        self.memory = None
        self.session_md_enabled = bool(config.get("session_md_enabled", False))
        self.session_md_dir = config.get("session_md_dir", "storage/session_rd")
        self.session_md_logger = (
            SessionMarkdownLogger(self.session_md_dir) if self.session_md_enabled else None
        )

        if self.enable_memory and self.session_id:
            self.memory = redis_memory_manager.create_memory(
                session_id=self.session_id,
                agent_id=self.agent_id,
                window_size=10
            )

        # 初始化LLM和工具路由
        self._init_langchain()
        self.tool_router = ToolRouter()

        # 🚀 初始化独立计划器
        # 优先使用规划器专用配置，避免使用视觉模型
        from app.config import settings

        # 检查是否启用了规划器专用配置
        if any([settings.PLANNER_API_KEY, settings.PLANNER_BASE_URL, settings.PLANNER_MODEL]):
            # 使用规划器专用配置
            planner_config = {
                "model": settings.PLANNER_MODEL or "glm-4.6",
                "api_key": settings.PLANNER_API_KEY,
                "base_url": settings.PLANNER_BASE_URL,
                "temperature": settings.PLANNER_TEMPERATURE,
                "max_tokens": settings.PLANNER_MAX_TOKENS,
                "timeout": settings.PLANNER_TIMEOUT,
                "similarity_threshold": 0.85
            }
            planner_model = planner_config["model"]
            logger.info(f"✅ 使用规划器专用配置: {planner_model}")
        else:
            planner_model = self.model
            # 使用通用LLM配置
            planner_config = {
                "model": planner_model,
                "temperature": 0.05,  # 计划生成使用更低温度
                "max_tokens": 800,
                "timeout": 30.0,
                "similarity_threshold": 0.85
            }
            logger.info(f"✅ 使用通用LLM配置，规划器模型: {planner_model}")

        self.planner = IntelligentPlanner(planner_config)
        logger.info(f"✅ 独立计划器初始化完成: {planner_model}")

    def _postprocess_tool_arguments(self, tool: Any, arguments: Any, agent_context: Dict[str, Any]) -> Any:
        """Best-effort补齐外部工具常见必填字段（例如 componentId）。

        NOTE:
        - 外部工具的真实后端可能要求字段，但客户端传来的 JSON Schema 未声明 required；
          此处做轻量兜底，避免重复触发 “xxx 不能为空” 的外部报错。
        """
        if not isinstance(arguments, dict):
            return arguments

        tool_name = str(getattr(tool, "name", "") or "")
        tool_parameters = getattr(tool, "parameters", None)
        declared_properties = set()
        try:
            props = tool_parameters.get("properties") if isinstance(tool_parameters, dict) else None
            if isinstance(props, dict):
                declared_properties = set(props.keys())
        except Exception:
            declared_properties = set()

        init_parameters = agent_context.get("init_parameters") if isinstance(agent_context, dict) else None
        if not isinstance(init_parameters, dict):
            init_parameters = {}

        def _is_missing(value: Any) -> bool:
            if value is None:
                return True
            if isinstance(value, str):
                s = value.strip()
                if not s:
                    return True
                if s.lower() in {"auto-filled", "autofilled", "auto filled"}:
                    return True
            if isinstance(value, (list, tuple, set)) and len(value) == 0:
                return True
            if isinstance(value, dict) and len(value) == 0:
                return True
            return False

        def _extract_suffix_id(name: str) -> Optional[str]:
            if not name or "#" not in name:
                return None
            suffix = name.rsplit("#", 1)[-1].strip()
            if not suffix:
                return None
            # Heuristic: CamelCase/含大写通常是方法名（如 #GetMetadata），非资源 id
            if any(ch.isupper() for ch in suffix):
                return None
            if re.match(r"^[a-z0-9][a-z0-9_-]{2,63}$", suffix):
                return suffix
            # fallback: still return suffix when it's short and looks id-ish
            if re.match(r"^[a-z0-9_-]{3,128}$", suffix, flags=re.IGNORECASE):
                return suffix
            return None

        args = dict(arguments)

        # componentId 兜底：优先使用显式参数，其次 init_parameters，其次工具名后缀（或 alias 映射后的后缀）
        # IMPORTANT: 仅当该字段在工具 schema 中声明时才自动补齐，避免向严格校验的外部工具注入“非法字段”。
        component_keys = ("componentId", "component_id", "component id")
        allowed_component_keys = [k for k in component_keys if k in declared_properties]
        if not allowed_component_keys:
            # Also strip these legacy aliases when the tool schema doesn't declare them.
            for k in component_keys:
                args.pop(k, None)
            return args

        suffix_id = _extract_suffix_id(tool_name)
        if suffix_id is None:
            tool_aliases = init_parameters.get("_tool_aliases") if isinstance(init_parameters, dict) else None
            if isinstance(tool_aliases, dict):
                mapped = tool_aliases.get(tool_name)
                if isinstance(mapped, str) and mapped:
                    suffix_id = _extract_suffix_id(mapped)

        candidate = None
        for v in (
            args.get("componentId"),
            args.get("component_id"),
            args.get("component id"),
            init_parameters.get("componentId"),
            init_parameters.get("component_id"),
            init_parameters.get("component id"),
            suffix_id,
        ):
            if not _is_missing(v):
                candidate = str(v).strip() if isinstance(v, str) else v
                break
        if candidate is not None:
            for k in allowed_component_keys:
                if _is_missing(args.get(k)):
                    args[k] = candidate

        return args

    def _get_env_model(self, model_type: str) -> Optional[str]:
        key = "LLM_MODEL" if model_type == "llm" else "VLM_MODEL"
        legacy_key = "LLM_MODEL_NEW" if model_type == "llm" else "VLM_MODEL_NEW"
        value = os.getenv(key) or os.getenv(legacy_key)
        if value:
            return value
        try:
            from app.core.llm.model_config_manager import get_model_config_manager

            manager = get_model_config_manager()
            if not getattr(manager, "_config", None):
                manager.initialize()
            cfg = manager.get_model_config("llm" if model_type == "llm" else "vlm")
            return getattr(cfg, "model_name", None)
        except Exception:
            return None

    def _normalize_configured_model(self, model: Optional[str]) -> str:
        if not model:
            return model or ""
        lowered = str(model).strip().lower()
        is_legacy = lowered.startswith(("qwen-", "qwen_")) or "dashscope" in lowered
        if not is_legacy:
            return model
        env_model = self._get_env_model("llm")
        if env_model and env_model != model:
            logger.warning("⚠️ 检测到旧模型配置 '%s'，使用环境模型 '%s'", model, env_model)
            return env_model
        return model
    
    def _read(self):
        self.logger.debug("tool_res=%s", self.tool_res)

    def _md_append(self, title: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.session_md_logger or not self.session_id:
            return
        self.session_md_logger.append(
            session_id=self.session_id,
            title=title,
            content=content,
            meta=meta,
            agent_id=self.agent_id,
        )

    async def _fill_parameters_with_parameter_filler(
        self,
        tool: Any,
        user_message: str,
        agent_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用 ParameterFiller 填充参数并生成 JAIP 协议的 tools.confirm 格式

        Args:
            tool: 工具实例
            user_message: 用户消息
            agent_context: Agent上下文

        Returns:
            JAIP 协议的 tools.confirm 消息
        """
        import uuid

        # 获取工具信息
        tool_name = getattr(tool, 'name', 'unknown')
        display_tool_name = self._display_tool_name(tool_name)
        tool_label = self._display_tool_label(tool_name)
        tool_description = (getattr(tool, 'description', '') or '').strip()
        parameters = getattr(tool, 'parameters', {})

        # 🆕 调试：打印参数定义
        logger.info(f"🔧 [参数调试] 工具: {display_tool_name}")
        logger.info(f"🔧 [参数调试] 参数定义: {parameters}")
        if parameters:
            props = parameters.get('properties', {})
            for param_name, param_def in props.items():
                default_val = param_def.get('default', 'NO_DEFAULT')
                logger.info(f"🔧 [参数调试] 参数 '{param_name}': {param_def.get('description', 'NO_DESC')} (默认值: {default_val})")
        else:
            logger.warning(f"🔧 [参数调试] 参数定义为空！")

        # 生成执行ID
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"

        # 使用 ParameterFiller 填充参数
        try:
            filler = self._get_parameter_filler()
            if filler is None:
                logger.error(f"❌ ParameterFiller 不可用，无法填充参数: {tool_name}")
                raise RuntimeError(f"ParameterFiller 不可用，无法为工具 {tool_name} 填充参数")

            logger.info(f"🔧 ParameterFiller 开始填充参数: {tool_name}")

            # 🆕 传递init_parameters到参数填充器
            enhanced_context = agent_context.copy()
            enhanced_context['init_parameters'] = self.init_parameters
            logger.info(f"🔧 [参数调试] init_parameters: {self.init_parameters}")

            filled_params = await filler.fill_parameters(
                parameters=parameters,
                user_message=user_message,
                context=enhanced_context,
                tool_name=tool_name
            )
            filled_params = self._postprocess_tool_arguments(tool, filled_params, enhanced_context)

            # 将 parameters 转换为 inputs 格式（使用 ParameterFiller 的方法）
            # NOTE: 有些前端会用 `command.inputs` 作为“允许传参白名单”，因此这里要把自动补齐的关键参数
            # (如 componentId) 同步进 inputs，避免前端执行时丢参导致 “component id 不能为空”。
            # IMPORTANT: deep-copy to avoid mutating tool.parameters (which would affect later LLM filling).
            import copy

            parameters_for_inputs = copy.deepcopy(parameters) if isinstance(parameters, dict) else {"type": "object"}
            props = parameters_for_inputs.get("properties")
            if not isinstance(props, dict):
                props = {}
            # 先把已填充的参数写入 default，方便前端表单/执行器直接复用（有些实现不会读取 call.arguments）
            if isinstance(filled_params, dict):
                for k, v in filled_params.items():
                    if k in props and isinstance(props.get(k), dict) and "default" not in props.get(k, {}):
                        props[k] = {**props[k], "default": v}
            for key in ("componentId", "component_id", "component id"):
                if key in (filled_params or {}) and key not in props:
                    props[key] = {
                        "type": "string",
                        "description": "auto-filled",
                        "default": (filled_params or {}).get(key),
                    }
            parameters_for_inputs["properties"] = props
            if "required" not in parameters_for_inputs or not isinstance(parameters_for_inputs.get("required"), list):
                parameters_for_inputs["required"] = []

            inputs = filler._convert_parameters_to_inputs(parameters_for_inputs)
        except Exception as e:
            logger.error(f"❌ ParameterFiller 调用失败: {e}")
            raise

        # 构建 tools.confirm 格式
        confirmation = {
            'jsonrpc': '2.0',
            'method': 'tools.confirm',
            'id': execution_id,
            'params': {
                'message': (
                    f"准备调用工具：{tool_label}"
                    + (f"（{tool_description[:80]}{'...' if len(tool_description) > 80 else ''}）" if tool_description else "")
                    + "，请确认参数。"
                ),
                'call': {
                    'toolName': display_tool_name,
                    'toolDisplayName': tool_label,
                    'arguments': filled_params,
                    'executionId': execution_id
                },
                'command': {
                    'id': display_tool_name,
                    'name': tool_label,
                    'description': tool_description,
                    'inputs': inputs
                }
            }
        }

        # 如果工具是知识库工具
        if 'knowledgeService#Search' == display_tool_name:
            confirmation['params']['auto'] = True

        logger.info(f"✅ ParameterFiller 填充完成: {display_tool_name}, 参数: {filled_params}")
        return confirmation 

    def _init_langchain(self):
        """初始化LangChain LLM（用于对话生成和计划生成）"""
        try:
            from app.core.llm.langchain_factory import create_langchain_llm

            # 🚀 使用新的分离配置创建LangChain LLM（支持LLM_API_KEY等分离配置）
            config_model = self.model
            if config_model and ("qwen" in config_model.lower() or "dashscope" in config_model.lower()):
                # 如果是旧的qwen/dashscope模型，强制使用None让统一配置决定
                logger.warning(f"⚠️ 检测到旧模型配置 '{config_model}'，强制使用分离配置")
                config_model = None

            self.llm = create_langchain_llm(
                model=config_model,  # 传递处理后的模型，None则使用环境变量LLM_MODEL
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            logger.info(f"✅ LangChain认知代理 LLM初始化成功: {self.model}")

            logger.info(f"🔧 参数填充器: ParameterFiller (已简化)")

        except ImportError as e:
            logger.error(f"LangChain dependency missing: {e}")
            raise
        except Exception as e:
            logger.error(f"LangChain initialization failed: {e}")
            raise

    
    
    def _format_plan_to_markdown(self, plan: Dict[str, Any]) -> str:
        """
        将工具调用计划转换为Markdown格式

        Args:
            plan: {"steps": [...], "tools_needed": [...]}

        Returns:
            Markdown格式的计划文本
        """
        steps = plan.get("steps", [])
        tools = plan.get("tools_needed", [])

        if not steps:
            return ""

        markdown = "## 📋 执行计划\n\n"
        planner_model = getattr(getattr(self, "planner", None), "model", None)
        if planner_model:
            markdown += f"**规划器模型(Planner)**：`{planner_model}`\n\n"

        for i, (step, tool) in enumerate(zip(steps, tools), 1):
            markdown += f"**步骤 {i}**\n"
            markdown += f"{step}\n"
            markdown += f"🔧 工具：`{self._display_tool_label(tool)}`\n\n"

        unavailable = plan.get("unavailable_tools") or []
        if isinstance(unavailable, list) and unavailable:
            # Keep it compact; this is mainly for debugging “为什么只规划了两个工具”.
            labels = [self._display_tool_label(str(t)) for t in unavailable if t]
            labels = [x for x in labels if x]
            if labels:
                markdown += "⚠️ 当前会话未注册/不可用的工具（因此未纳入计划）：\n"
                for item in labels[:12]:
                    markdown += f"- `{item}`\n"
                markdown += "\n"

        return markdown

    
    # ==========================================
    # 辅助方法：上下文初始化
    # ==========================================

    async def _initialize_execution_context(
        self,
        messages: List[Any],
        **kwargs
    ) -> Dict[str, Any]:
        """初始化执行上下文

        职责：
        1. 设置 WebSocket handler
        2. 解析用户消息和系统提示
        3. 加载对话历史和工具调用历史
        4. 构建 agent_context

        Returns:
            {
                "system_prompt": str,
                "input_text": str,
                "tools": List,
                "agent_context": Dict
            }
        """
        # 设置WebSocket handler
        websocket_handler = kwargs.get('websocket_handler')
        if websocket_handler:
            self.tool_router.websocket = websocket_handler
            logger.info("✅ WebSocket handler已设置")

        # 解析消息
        system_prompt = self.system_prompt
        input_text = ""
        tools = []

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role')
                content = msg.get('content')
                if role == 'system':
                    system_prompt = content
                elif role == 'user':
                    input_text = content
                elif role == 'tools':
                    tools = content

        # 加载历史
        conversation_history = []
        tool_call_history = []

        if self.session_id:
            try:
                import asyncio
                filler = self._get_parameter_filler()
                if filler is not None:
                    tool_call_history = await asyncio.to_thread(
                        filler.load_tool_call_history,
                        session_id=self.session_id,
                        agent_id=self.agent_id
                    )
                else:
                    tool_call_history = []
            except Exception as e:
                logger.warning(f"⚠️ 加载工具历史失败: {e}")

        if self.memory:
            try:
                memory_variables = self.memory.load_memory_variables({})
                chat_history = memory_variables.get("chat_history", [])
                recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history

                for msg in recent_history:
                    role = "用户" if msg.type == "human" else "助手"
                    content = msg.content[:100]
                    conversation_history.append(f"{role}: {content}")
            except Exception as e:
                logger.warning(f"⚠️ 加载对话历史失败: {e}")

        # 构建agent上下文
        agent_context = {
            "system_prompt": system_prompt,
            "agent_prompt": self.system_prompt,
            "init_parameters": self.init_parameters,
            "conversation_history": conversation_history,
            "tool_call_history": tool_call_history,
            "tool_res": self.tool_res
        }
        usage_tracker = kwargs.get("usage_tracker")
        if usage_tracker:
            agent_context["usage_tracker"] = usage_tracker

        return {
            "system_prompt": system_prompt,
            "input_text": input_text,
            "tools": tools,
            "agent_context": agent_context
        }

    # ==========================================
    # 辅助方法：工具执行规划
    # ==========================================

    async def _intelligent_tool_selection(
        self,
        input_text: str,
        tools: List[Any],
        agent_context: Dict
    ):
        """
        🚀 重构：使用独立计划器进行智能工具选择

        统一处理：
        1. 判断是否需要工具（单个或多个）
        2. 选择工具和调用链路
        3. 生成执行计划
        4. 映射工具实例

        Returns:
            (plan, planned_tools, use_historical_plan, is_complex_task)
        """
        logger.info(f"🎯 使用独立计划器: {input_text[:50]}...")

        # 准备历史计划数据
        historical_plans = []
        try:
            similar_plans = await self._search_similar_successful_plans(
                user_input=input_text,
                top_k=3,
                similarity_threshold=0.85
            )
            if similar_plans:
                historical_plans = [{
                    "user_input": input_text,
                    "plan": similar_plans["plan"],
                    "similarity": similar_plans["similarity"]
                }]
        except Exception as e:
            logger.warning(f"⚠️ 获取历史计划失败: {e}")

        # 🚀 使用独立计划器生成计划
        planning_context = {
            "agent_context": agent_context,
            "conversation_history": agent_context.get("conversation_history", []),
            # IMPORTANT: planner has its own system prompt; pass agent prompt so it can respect
            # fixed tool orders / business constraints configured on the agent.
            "system_prompt": self.system_prompt,
        }
        if agent_context.get("usage_tracker"):
            planning_context["usage_tracker"] = agent_context.get("usage_tracker")

        try:
            from app.core.skills import get_skill_registry

            registry = get_skill_registry()
            skills = registry.list_skills()
            planning_context["skills"] = [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "tags": skill.tags,
                }
                for skill in skills
            ]
        except Exception as e:
            logger.debug("⚠️ Failed to load skills for planning: %s", e)

        plan_result = await self.planner.generate_plan(
            user_input=input_text,
            available_tools=tools,
            context=planning_context,
            historical_plans=historical_plans
        )

        # 提取结果
        task_type = plan_result.get("task_type", "conversation")
        tools_needed = plan_result.get("tools_needed", [])
        steps = plan_result.get("steps", [])
        reasoning = plan_result.get("reasoning", "")
        is_cached = plan_result.get("is_cached", False)
        strategy = plan_result.get("strategy", "unknown")

        # 构建标准计划格式
        plan = {
            "steps": steps,
            "tools_needed": tools_needed,
            "reasoning": reasoning,
            "task_type": task_type,
            "confidence": plan_result.get("confidence", 0.8)
        }
        # Surface filtered/unavailable tools without treating them as fatal missing-tools.
        # (The fatal `missing_tools` list is computed later when a tool cannot be resolved to an instance.)
        unavailable_tools: List[str] = []
        for key in ("unavailable_tools", "missing_tools"):
            value = plan_result.get(key)
            if isinstance(value, list):
                unavailable_tools.extend([str(x) for x in value if x])
        if unavailable_tools:
            seen = set()
            plan["unavailable_tools"] = [x for x in unavailable_tools if x and not (x in seen or seen.add(x))]

        # 映射工具实例 - 添加详细调试
        logger.info(f"🔍 [DEBUG] 可用工具列表 (共{len(tools)}个):")
        for i, t in enumerate(tools):
            tool_name = getattr(t, 'name', 'NO_NAME')
            tool_id = getattr(t, 'id', getattr(t, 'tool_id', 'NO_ID'))
            logger.info(f"   [{i}] name='{tool_name}', id='{tool_id}', type={type(t)}")

        logger.info(f"🔍 [DEBUG] 计划器需要的工具: {tools_needed}")

        # 尝试多种工具标识符映射
        tool_map = {}
        normalized_index = {}
        normalized_count = {}
        suffix_index = {}
        suffix_count = {}

        def _norm(value: Any) -> str:
            try:
                text = str(value or "")
            except Exception:
                text = ""
            return re.sub(r"[^0-9a-zA-Z]+", "", text).lower()

        for t in tools:
            # 使用多种可能的标识符
            name = getattr(t, 'name', '')
            tool_id = getattr(t, 'id', getattr(t, 'tool_id', ''))
            if name:
                tool_map[name] = t
                n = _norm(name)
                if n:
                    normalized_index[n] = t
                    normalized_count[n] = normalized_count.get(n, 0) + 1
                if isinstance(name, str) and "#" in name:
                    suffix = name.rsplit("#", 1)[-1].strip().lower()
                    if suffix:
                        suffix_index[suffix] = t
                        suffix_count[suffix] = suffix_count.get(suffix, 0) + 1
            if tool_id:
                tool_map[tool_id] = t
                n = _norm(tool_id)
                if n:
                    normalized_index[n] = t
                    normalized_count[n] = normalized_count.get(n, 0) + 1

        logger.info(f"🔍 [DEBUG] 工具映射表: {list(tool_map.keys())}")

        planned_tools = []
        missing_tools = list(plan.get("missing_tools", []) or [])
        tool_aliases = None
        try:
            init_params = agent_context.get("init_parameters") if isinstance(agent_context, dict) else None
            if isinstance(init_params, dict):
                tool_aliases = init_params.get("_tool_aliases")
        except Exception:
            tool_aliases = None

        for name in tools_needed:
            resolved = tool_map.get(name)
            if resolved is None and isinstance(tool_aliases, dict):
                mapped = tool_aliases.get(name)
                if mapped:
                    resolved = tool_map.get(mapped)
            if resolved is None:
                n = _norm(name)
                if n and normalized_count.get(n) == 1:
                    resolved = normalized_index.get(n)
            if resolved is None and isinstance(name, str) and "#" not in name:
                s = name.strip().lower()
                if s and suffix_count.get(s) == 1:
                    resolved = suffix_index.get(s)

            if resolved is not None:
                planned_tools.append(resolved)
                logger.info(f"✅ [DEBUG] 找到工具: {name}")
            else:
                logger.warning(f"❌ [DEBUG] 未找到工具: {name}")
                if name not in missing_tools:
                    missing_tools.append(name)

        logger.info(f"🔍 [DEBUG] 实际匹配的工具: {[getattr(t, 'name', getattr(t, 'id', 'UNKNOWN')) for t in planned_tools]}")
        is_complex_task = len(planned_tools) >= 2
        use_historical_plan = is_cached and strategy == "historical_cache"

        # 日志记录
        logger.info(f"📋 计划器结果: {strategy} -> {task_type}")
        logger.info(f"   🔧 工具数量: {len(planned_tools)}")
        logger.info(f"   🎯 任务类型: {'复杂任务' if is_complex_task else '简单任务'}")
        logger.info(f"   💾 缓存状态: {'命中' if is_cached else '未命中'}")
        logger.info(f"   ⏱️  执行时间: {plan_result.get('execution_time', 0):.2f}s")

        if missing_tools:
            plan["missing_tools"] = missing_tools
            logger.warning(f"⚠️ 计划包含未注册工具: {missing_tools}")

        return plan, planned_tools, missing_tools, use_historical_plan, is_complex_task

    # ==========================================
    # 辅助方法：执行单个工具
    # ==========================================

    async def _execute_single_tool(
        self,
        tool: Any,
        iteration: int,
        input_text: str,
        agent_context: Dict
    ) -> AsyncIterator[Any]:
        """执行单个工具

        职责：
        1. 参数填充
        2. 注册确认
        3. 判断工具类型并执行
        4. 返回工具结果

        Yields:
            - confirmation 消息（需要前端确认）
            - 工具执行结果
        """
        logger.info(f"🔄 [第{iteration + 1}轮] 执行工具: {tool.name}")

        # 参数填充
        agent_context['tool_res'] = self.tool_res

        # 使用 ParameterFiller 填充参数
        logger.info(f"🔧 使用 ParameterFiller: {tool.name}")
        confirmation = await self._fill_parameters_with_parameter_filler(
            tool=tool,
            user_message=input_text,
            agent_context=agent_context
        )

        confirmation_id = confirmation.get("id", f"tool_confirm_{int(time.time() * 1000)}")
        tool_params = confirmation.get("params", {}).get("call", {}).get("arguments", {})

        # 工具来源与参数校验日志
        is_internal = internal_tool_registry.is_internal_tool(tool.name)
        tool_type = "internal" if is_internal else "external"
        tool_category = getattr(tool, "category", None)
        tool_class = type(tool).__name__
        try:
            params_preview = json.dumps(tool_params, ensure_ascii=False)
        except Exception:
            params_preview = str(tool_params)
        if len(params_preview) > 800:
            params_preview = params_preview[:800] + "...(truncated)"
        logger.info(
            "🧩 [工具信息] tool=%s | toolType=%s | class=%s | category=%s | confirm_id=%s | args=%s",
            tool.name,
            tool_type,
            tool_class,
            tool_category,
            confirmation_id,
            params_preview,
        )

        # 必填参数校验：缺参时直接中止，避免连续多次外部工具报同一个错误
        required_fields = []
        try:
            schema = getattr(tool, "parameters", None) or {}
            required_fields = schema.get("required", []) if isinstance(schema, dict) else []
        except Exception:
            required_fields = []

        def _is_missing(value: Any) -> bool:
            if value is None:
                return True
            if isinstance(value, str) and not value.strip():
                return True
            if isinstance(value, (list, tuple, set)) and len(value) == 0:
                return True
            if isinstance(value, dict) and len(value) == 0:
                return True
            return False

        missing_required = []
        # Some external tool schemas (esp. visualizationService) don't mark required fields correctly.
        # Add a small heuristic so we fail fast instead of hanging the external executor with null params.
        try:
            schema_props = (schema or {}).get("properties") if isinstance(schema, dict) else None
            if (
                isinstance(getattr(tool, "name", None), str)
                and tool.name.lower().startswith("visualizationservice:")
                and isinstance(schema_props, dict)
            ):
                must_have = []
                if "text" in schema_props:
                    must_have.append("text")
                if "terms" in schema_props:
                    must_have.append("terms")
                if "id" in schema_props:
                    must_have.append("id")
                if "component" in schema_props:
                    must_have.append("component")
                for k in must_have:
                    if k and k not in required_fields:
                        required_fields.append(k)
        except Exception:
            pass

        if isinstance(required_fields, list) and required_fields:
            for field in required_fields:
                if not field:
                    continue
                if _is_missing(tool_params.get(field)):
                    missing_required.append(field)

        if missing_required:
            logger.warning(
                "⚠️ [参数缺失] tool=%s | toolType=%s | required=%s | missing=%s | args=%s",
                tool.name,
                tool_type,
                required_fields,
                missing_required,
                params_preview,
            )
            raise ToolPipelineAbort(
                f"工具 `{tool.name}` 缺少必填参数: {', '.join(missing_required)}。"
                "请在初始化参数（session.initialize.params.parameters）或问题描述中提供这些参数后重试。",
                tool_name=tool.name,
                missing_required=missing_required,
            )

        # 🔥 JSON处理器的自动确认机制
        if self.auto_confirm_tools:
            logger.info(f"🤖 [自动确认] 自动执行工具: {tool.name}")
            # 直接执行工具，跳过用户确认
            result = await self.tool_router.execute_tool(tool, tool_params)
            yield result
            return

        # 注册Future（仅用于WebSocket确认模式）
        import asyncio
        future = asyncio.Future()
        self.tool_router.pending_confirmations[confirmation_id] = {
            "future": future,
            "tool": tool,
            "params": tool_params,
            "created_at": time.time()
        }
        logger.info(f"📋 已注册确认: {confirmation_id} -> {tool.name}")

        # ⚠️ 注意：不在这里yield confirmation，由_handle_internal_tool / _handle_external_tool负责
        # yield confirmation

        logger.info(f"🔍 工具类型: {tool.name} -> {'内部' if is_internal else '外部'}")

        # 执行工具
        final_result = None
        if is_internal:
            async for result_or_msg in self._handle_internal_tool(
                tool, tool_params, confirmation_id
            ):
                if isinstance(result_or_msg, dict):
                    yield result_or_msg
                else:
                    final_result = result_or_msg
        else:
            async for result_or_msg in self._handle_external_tool(
                confirmation, confirmation_id
            ):
                if isinstance(result_or_msg, dict):
                    yield result_or_msg
                else:
                    final_result = result_or_msg

        yield final_result

    # ==========================================
    # 辅助方法：处理内部工具
    # ==========================================

    async def _handle_internal_tool(
        self,
        tool: Any,
        tool_params: Dict,
        confirmation_id: str
    ) -> AsyncIterator[Any]:
        """处理内部工具执行

        职责：
        1. 查询工具执行配置
        2. 根据 execution_mode 选择执行策略
            - sync: 直接执行
            - async/background: yield 暂停，等待后台执行

        Yields:
            - 慢速工具的 internal_confirmation
            - 工具执行结果
        """
        # 获取工具执行配置
        tool_id = self._display_tool_name(getattr(tool, "name", ""))
        tool_config = tool_config_service.get_execution_config(tool_id)
        exec_mode = tool_config.get("execution_mode", "sync")
        estimated_time = tool_config.get("estimated_time", 3)
        exec_timeout = tool_config.get("timeout", 30)

        logger.info(f"🔧 内部工具: {tool_id} | 模式={exec_mode} | 预估={estimated_time}秒")

        if exec_mode == "sync":
            # ⚡ 快速工具：直接同步执行
            logger.info(f"⚡ 快速内部工具，直接执行: {tool.name}")
            result = await self.tool_router.execute_tool(tool, tool_params)
            logger.info(f"✅ 内部工具完成: {tool.name}")
            yield result
        else:
            # 🐌 慢速工具：yield 暂停
            logger.info(f"🐌 慢速内部工具，yield 暂停: {tool.name} (预估{estimated_time}秒)")

            internal_confirmation = {
                "id": confirmation_id,
                "method": "tools.confirm",
                "params": {
                    "call": {
                        "toolName": tool_id,
                        "arguments": tool_params
                    },
                    "toolType": "internal",
                    "executionMode": exec_mode,
                    "estimatedTime": estimated_time,
                    "timeout": exec_timeout
                }
            }

            # yield 暂停，等待 handler 后台执行
            final_result = yield internal_confirmation
            logger.info(f"✅ 慢速内部工具恢复，收到结果: {tool.name}")
            yield final_result

    # ==========================================
    # 辅助方法：处理外部工具
    # ==========================================

    async def _handle_external_tool(
        self,
        confirmation: Dict,
        confirmation_id: str
    ) -> AsyncIterator[Any]:
        """处理外部工具执行

        职责：
        1. 构造外部工具确认消息
        2. yield 暂停，等待前端返回结果

        Yields:
            - external_confirmation
            - 工具执行结果
        """
        logger.info(f"🌐 外部工具，yield 暂停")

        # 构造外部工具确认
        external_confirmation = confirmation.copy()
        if "params" not in external_confirmation:
            external_confirmation["params"] = {}
        external_confirmation["params"]["toolType"] = "external"

        # yield 暂停，等待前端返回结果
        final_result = yield external_confirmation
        logger.info(f"✅ 外部工具恢复，收到结果: {final_result}")
        yield final_result

    # ==========================================
    # 辅助方法：执行工具流水线
    # ==========================================

    async def _execute_tool_pipeline(
        self,
        planned_tools: List[Any],
        input_text: str,
        agent_context: Dict,
        lc_messages: List[Any]
    ) -> AsyncIterator[Any]:
        """执行工具流水线

        职责：
        1. 依次执行每个工具
        2. 更新上下文（让下一个工具能看到前面的结果）
        3. 构建消息历史

        Yields:
            - 工具确认消息
            - 工具执行进度

        Returns:
            tools_executed: List[str]
        """
        from langchain_core.messages import ToolMessage

        tools_executed = []

        for iteration, tool in enumerate(planned_tools):
            try:
                # 🔧 重要修复：在每个工具执行前，确保agent_context中有最新的工具结果
                if iteration > 0:  # 不是第一个工具
                    agent_context['tool_res'] = self.tool_res
                    logger.info(f"🔄 [第{iteration + 1}轮] 更新上下文: tool_res类型={type(self.tool_res)}")
                    logger.info(f"🔄 [第{iteration + 1}轮] 上下文中的工具历史: {len(agent_context.get('tool_call_history', []))}条")

                # 执行单个工具
                final_result = None
                tool_params = {}  # 初始化工具参数
                async for item in self._execute_single_tool(
                    tool, iteration, input_text, agent_context
                ):
                    if isinstance(item, dict):
                        # yield 工具确认消息
                        yield item
                        # 从确认消息中提取参数
                        if 'params' in item and 'call' in item['params']:
                            tool_params = item['params']['call'].get('arguments', {})
                    else:
                        # 工具执行结果
                        final_result = item

                tool_display_name = self._display_tool_name(getattr(tool, "name", ""))
                tools_executed.append(tool_display_name)

                # 🎯 修复：如果final_result为None，使用self.tool_res中已有的结果
                if final_result is None and self.tool_res:
                    logger.info(f"🔧 [修复] final_result为None，使用self.tool_res: {type(self.tool_res)}")
                    final_result = self.tool_res

                # 外部工具错误：立即中止后续工具，避免重复报错（尤其是缺参类错误）
                if isinstance(final_result, dict) and final_result.get("success") is False:
                    err_msg = str(final_result.get("error") or "")
                    err_code = final_result.get("error_code")
                    logger.warning(
                        "⚠️ [工具失败] tool=%s | code=%s | error=%s",
                        tool_display_name,
                        err_code,
                        err_msg,
                    )
                    raise ToolPipelineAbort(
                        f"外部工具 `{tool_display_name}` 调用失败"
                        + (f"（code={err_code}）" if err_code is not None else "")
                        + f": {err_msg}",
                        tool_name=tool_display_name,
                    )

                # 更新self.tool_res以保持一致性（关键：这会传递给下一个工具）
                self.tool_res = final_result
                logger.info(f"📝 [第{iteration + 1}轮] 工具结果已保存: {tool_display_name} -> tool_res类型={type(final_result)}")

                # 🔧 修复：不使用ToolMessage，改为收集工具结果用于提示词
                # 因为通义千问API要求ToolMessage必须跟在tool_calls之后，我们改为直接在提示词中描述工具结果

                logger.info(f"📝 [第{iteration + 1}轮] 工具结果已保存到agent_context")

                # 更新上下文（关键：让下一个工具能看到前一个工具的结果）
                if 'tool_call_history' not in agent_context:
                    agent_context['tool_call_history'] = []

                agent_context['tool_call_history'].append({
                    'tool_name': tool_display_name,
                    'arguments': tool_params,
                    'result': final_result
                })

                logger.info(f"📝 已更新上下文: 工具历史={len(agent_context['tool_call_history'])}条")

                # 🔍 调试：打印工具调用历史
                if len(agent_context['tool_call_history']) > 0:
                    last_call = agent_context['tool_call_history'][-1]
                    logger.info(f"🔍 [调试] 最新工具调用: {last_call['tool_name']} -> 结果类型: {type(last_call['result'])}")
                    if isinstance(last_call['result'], dict):
                        logger.info(f"🔍 [调试] 结果内容摘要: {str(last_call['result'])[:200]}...")

            except ToolPipelineAbort as e:
                # 缺参/外部工具失败等应立即中止，避免反复尝试导致“同一个错误刷屏”
                logger.error(f"❌ 工具执行异常: {tool.name} - {e}")
                raise
            except Exception as e:
                logger.error(f"❌ 工具执行异常: {tool.name} - {e}", exc_info=True)
                raise

        logger.info(f"🔍 [工具流水线] 执行完成，共{len(tools_executed)}个工具: {tools_executed}")
        # 在异步生成器中不能使用return，用yield代替
        # return tools_executed  # 确保返回结果

    # ==========================================
    # 辅助方法：生成最终响应
    # ==========================================

    async def _generate_final_response(
        self,
        system_prompt: str,
        input_text: str,
        lc_messages: List[Any],
        agent_context: Dict
    ) -> AsyncIterator[str]:
        """生成最终响应

        职责：
        1. 构建完整的消息列表
        2. 智能选择模型并调用 LLM 生成回复
        3. 保存到对话记忆

        Yields:
            响应文本块

        Returns:
            完整响应文本
        """

        # 构建工具执行结果摘要
        tool_results_summary = ""
        direct_output_result = None  # 直接输出的结果
        if agent_context.get('tool_call_history'):
            tool_results_summary = "\n\n## 工具执行结果：\n"
            for i, tool_call in enumerate(agent_context['tool_call_history'], 1):
                tool_name = self._display_tool_name(tool_call.get('tool_name', 'unknown'))
                arguments = tool_call.get('arguments', {})
                result = tool_call.get('result', '')

                # 检查是否为直接输出工具
                if self._should_direct_output(tool_name, result):
                    # 如果工具结果有 message 字段且包含格式化内容，直接使用
                    if isinstance(result, dict) and result.get('status') == 'success' and result.get('message'):
                        direct_output_result = result['message']
                        logger.info(f"🎯 [直接输出] 检测到直接输出工具: {tool_name}，跳过LLM总结")
                        break  # 直接输出模式下，只处理第一个成功的工具

                result_summary = self._format_tool_result_for_prompt(result)

                tool_results_summary += f"**工具{i}**: {tool_name}\n"
                tool_results_summary += f"- 参数: {arguments}\n"
                tool_results_summary += f"- 结果: {result_summary}\n\n"

        # 如果有直接输出结果，直接返回
        if direct_output_result:
            yield direct_output_result
            return

        # 构建消息列表（只包含对话历史，不包含ToolMessage）
        conversation_messages = [msg for msg in lc_messages if not hasattr(msg, 'tool_call_id')]

        logger.info(f"📝 消息列表: 共{len(conversation_messages)}条对话消息，过滤掉了ToolMessage")

        enhanced_system_prompt = f"{system_prompt}\n{self.system_prompt}{tool_results_summary}"

        message_list = [
            SystemMessage(content=enhanced_system_prompt),
            *conversation_messages,
            HumanMessage(content=f"用户问题: {input_text}\n\n请根据以上结果，给出完整准确的回答。 要求：1、回答详尽且有条理；2、如果工具结果报错或者不足以回答，请直接说明原因并建议下一步操作。3、不要凭空捏造信息。")
        ]

        # 🆕 智能模型选择：根据消息内容动态选择模型
        intelligent_model = await self._select_intelligent_model(message_list)

        # 🆕 重新创建LLM实例以使用智能选择的模型
        from app.core.llm.langchain_factory import create_langchain_llm
        intelligent_llm = create_langchain_llm(
            model=intelligent_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        logger.info(f"🧠 [智能模型选择] 原模型: {self.model} → 智能选择: {intelligent_model}")

        # 🆕 记录最终提示词（详细日志）
        final_prompt = "\n".join([f"[{type(m).__name__}] {m.content[:200]}" for m in message_list])

        # 记录LLM请求详情（使用智能选择的模型）
        self._log_llm_request(
            messages=message_list,
            model=intelligent_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        logger.info(f"📝 最终提示词:\n{final_prompt}")
        agent_context["final_prompt"] = final_prompt

        full_response = ""
        chunk_count = 0
        usage_tracker = agent_context.get("usage_tracker")
        from langchain_community.callbacks.manager import get_openai_callback
        from app.core.llm.usage import usage_from_openai_callback

        with get_openai_callback() as cb:
            async for chunk in intelligent_llm.astream(message_list):
                if hasattr(chunk, 'content'):
                    full_response += chunk.content
                    chunk_count += 1
                    yield chunk.content

        usage = usage_from_openai_callback(cb, model=intelligent_model, provider="langchain")
        if usage_tracker and usage:
            usage_tracker.add(usage, source="final_response")

        # 记录LLM响应详情
        self._log_llm_response(
            response=full_response,
            model=self.model
        )

        logger.info(f"📊 流式响应统计: {chunk_count} 个chunks, 总长度: {len(full_response)} 字符")

        # 保存到记忆
        if self.memory and full_response:
            self.memory.save_context(
                {"input": input_text},
                {"output": full_response}
            )

    # ==========================================
    # 辅助方法：智能模型选择
    # ==========================================

    async def _select_intelligent_model(self, message_list: List[Any]) -> str:
        """
        根据消息内容智能选择模型

        Args:
            message_list: LangChain消息列表

        Returns:
            str: 智能选择的模型名称
        """
        # 合并所有消息内容
        content = ""
        for msg in message_list:
            if hasattr(msg, 'content'):
                content += " " + msg.content
            elif isinstance(msg, dict):
                content += " " + str(msg.get('content', ''))

        content_lower = content.lower()

        # 图片相关关键词和模式
        image_keywords = [
            "imageurl", "image_url", "图片", "图像", "照片", "截图",
            "png", "jpg", "jpeg", "gif", "webp", "bmp"
        ]

        image_patterns = [
            r"imageurl[:：]\s*(https?://[^\s]+)",
            r"图片[:：]\s*(https?://[^\s]+)",
            r"图像[:：]\s*(https?://[^\s]+)",
            r"!\[.*?\]\(.*?\)",  # markdown图片格式
            r"https?://[^\s]*\.(png|jpg|jpeg|gif|webp|bmp)",
            r"图片分析", "图像识别", "安全巡检", "视频巡检"
        ]

        # 检查关键词
        for keyword in image_keywords:
            if keyword.lower() in content_lower:
                logger.info(f"🖼️ 检测到图片关键词: {keyword}")
                return self._get_env_model("vlm") or "qwen-vl-max"  # 多模态模型

        # 检查模式（使用简单匹配）
        import re
        for pattern in image_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.info(f"🖼️ 检测到图片模式: {pattern}")
                return self._get_env_model("vlm") or "qwen-vl-max"  # 多模态模型

        # 没有检测到图片内容，使用文本模型
        logger.info("📝 未检测到图片内容，使用文本模型")
        return self._get_env_model("llm") or self.model or "qwen-plus"

    # ==========================================
    # 辅助方法：保存到长期记忆
    # ==========================================
    async def _save_execution_to_memory(
        self,
        input_text: str,
        plan: Optional[Dict],
        tools_executed: List[str],
        full_response: str,
        use_historical_plan: bool,
        execution_success: bool,
        task_duration: float,
        agent_context: Dict
    ):
        """保存执行记录到长期记忆

        职责：
        1. 保存任务执行经验（如果有计划）
        2. 保存对话到长期记忆
        """
        # 保存任务经验
        if plan and tools_executed:
            await self._save_task_experience(
                user_input=input_text,
                plan=plan,
                tools_used=tools_executed,
                success=execution_success,
                result_summary=full_response[:200] if full_response else "无响应",
                duration=task_duration
            )

        # 保存对话
        if full_response:
            await self._save_dialog_to_long_term(
                user_input=input_text,
                assistant_response=full_response,
                context={
                    "tools_called": tools_executed,
                    "plan_type": "historical" if use_historical_plan else "generated",
                    "final_prompt": agent_context.get("final_prompt", "")
                }
            )

    async def _run_stream(self, messages: List[Any], **kwargs) -> AsyncIterator[Any]:
        """核心流式执行方法（重构版）

        重构改进：
        1. 拆分成 8 个职责单一的辅助方法
        2. 每个方法 < 100 行，职责清晰
        3. 提高可读性和可维护性

        执行流程：
        1. 初始化上下文
        2. 生成执行计划
        3. 执行工具流水线
        4. 生成最终响应
        5. 保存到长期记忆
        """
        task_start_time = time.time()
        try:
            # 🆕 记录Agent执行开始（处理不同格式的消息）
            input_text = ""
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    input_text = last_message.content
                elif isinstance(last_message, dict):
                    input_text = last_message.get('content', '')
                else:
                    input_text = str(last_message)

            self._log_agent_execution_start(input_text, self.name)
            # ===== 步骤1：初始化上下文 =====
            context_init = await self._initialize_execution_context(messages, **kwargs)
            system_prompt = context_init["system_prompt"]
            input_text = context_init["input_text"]
            tools = context_init["tools"]
            agent_context = context_init["agent_context"]
            history_count = len(agent_context.get("tool_call_history", []))
            self._md_append("user", input_text)

            # ===== 步骤2：统一工具选择和计划生成 =====
            plan, planned_tools, missing_tools, use_historical_plan, is_complex_task = await self._intelligent_tool_selection(
                input_text, tools, agent_context
            )

            # 🔥 立即发送计划（新增 - 计划完成后立即发送，不等待工具参数填充）
            if plan and plan.get("steps") and not missing_tools:
                # 🆕 记录计划生成详情
                self._log_plan_generation(plan, input_text)

                markdown_plan = self._format_plan_to_markdown(plan)
                yield markdown_plan
                yield {
                    "type": "plan_generated",
                    "steps": plan.get("steps", []),
                    "tools": plan.get("tools_needed", []),
                    "reasoning": plan.get("reasoning", ""),
                    "unavailable_tools": plan.get("unavailable_tools", []),
                    "confidence": plan.get("confidence", 0.8),
                    "sent_immediately": True  # 标记这是立即发送的计划
                }
                logger.info(f"🚀 已立即发送计划: {len(plan['steps'])} 步 - 不等待工具参数填充")

                # 🔥 关键改进：计划发送后不等待，立即开始工具执行
                # 这样用户可以立即看到计划，而工具在后台准备参数

            # ===== 步骤3：执行工具 =====
            lc_messages = []
            if self.memory:
                try:
                    memory_vars = self.memory.load_memory_variables({})
                    lc_messages.extend(memory_vars.get("chat_history", []))
                except Exception as memory_error:
                    logger.warning(f"⚠️ 内存加载失败，继续执行: {memory_error}")
                    # 继续执行，不加载历史记忆
            lc_messages.append(HumanMessage(content=input_text))
            
            task_start_time = time.time()
            tools_executed = []
            execution_success = True
            full_response = ""

            if missing_tools:
                missing_list = ", ".join(missing_tools)
                full_response = f"未配置或不可用的工具: {missing_list}。请先注册/启用相关工具后重新初始化会话。"
                execution_success = False
                logger.warning(f"⚠️ 工具缺失，终止本次执行: {missing_tools}")
                yield full_response
            elif not planned_tools:
                # 无工具：对话模式
                logger.info("⚠️ 无工具，对话模式")
                async for chunk in self._generate_final_response(
                    system_prompt, input_text, lc_messages, agent_context
                ):
                    full_response += chunk
                    yield chunk
            else:
                # 执行工具流水线
                async for item in self._execute_tool_pipeline(
                    planned_tools, input_text, agent_context, lc_messages
                ):
                    # yield 工具确认消息或进度
                    yield item

                # 流水线执行完成，收集已执行的工具名称
                tools_executed = [t.name for t in planned_tools]
                logger.info(f"🔧 [工具流水线] 执行完成，共{len(tools_executed)}个工具: {tools_executed}")
                new_history = agent_context.get("tool_call_history", [])[history_count:]
                if new_history:
                    for entry in new_history:
                        tool_name = entry.get("tool_name", "unknown")
                        arguments = entry.get("arguments", {})
                        result_text = self._format_tool_result_for_prompt(entry.get("result", ""))
                        self._md_append("tool", result_text, {"tool": tool_name, "arguments": arguments})
                    history_count = len(agent_context.get("tool_call_history", []))

                # ===== 步骤4：生成最终响应 =====
                logger.info(f"🚀 [最终响应] 开始生成最终回复...")
                try:
                    async for chunk in self._generate_final_response(
                        system_prompt, input_text, lc_messages, agent_context
                    ):
                        full_response += chunk
                        yield chunk
                    logger.info(f"✅ [最终响应] 最终回复生成完成，长度: {len(full_response)} 字符")
                except Exception as e:
                    logger.error(f"❌ [最终响应] 生成失败: {e}", exc_info=True)
                    yield f"生成回复时出错: {str(e)}"

            # ===== 步骤5：保存到长期记忆 =====
            task_duration = time.time() - task_start_time

            # 🆕 记录Agent执行结束
            self._log_agent_execution_end(self.name, task_duration, tools_executed)
            if full_response:
                self._md_append(
                    "assistant",
                    full_response,
                    {
                        "tools_executed": tools_executed,
                        "plan_type": "historical" if use_historical_plan else "generated",
                    },
                )

            await self._save_execution_to_memory(
                input_text, plan, tools_executed, full_response,
                use_historical_plan, execution_success, task_duration, agent_context
            )

        except Exception as e:
            logger.error(f"❌ 流式执行失败: {e}", exc_info=True)
            yield f"执行出错: {str(e)}"

    async def _save_task_experience(
        self,
        user_input: str,
        plan: Dict[str, Any],
        tools_used: List[str],
        success: bool,
        result_summary: str,
        duration: float = 0.0
    ):
        """保存任务执行经验（用于复用成功计划）"""
        try:
            from app.core.memory import unified_memory

            metadata = {
                "plan": plan,
                "tools_used": tools_used,
                "success": success,
                "result_summary": result_summary[:200],
                "execution_time": round(duration, 2),
                "tool_count": len(tools_used)
            }

            await unified_memory.save_memory(
                agent_id=self.agent_id or "default",
                memory_type="task",
                content=user_input,
                metadata=metadata,
                importance=1.0 if success else 0.3
            )

            logger.info(f"✅ 任务经验已保存 | {user_input[:50]}... | 成功={success}")

        except Exception as e:
            logger.warning(f"⚠️ 保存任务经验失败: {e}")

    async def _save_dialog_to_long_term(
        self,
        user_input: str,
        assistant_response: str,
        context: Dict[str, Any] = None
    ):
        """保存对话到长期记忆（突破Redis 10轮限制）"""
        try:
            from app.core.memory import unified_memory

            metadata = {
                "user_message": user_input,
                "assistant_message": assistant_response[:500],
                "session_id": self.session_id or "unknown",
                "timestamp": datetime.now().isoformat()
            }

            if context:
                metadata["tools_called"] = context.get("tools_called", [])
                metadata["files_processed"] = context.get("files_processed", [])
                metadata["final_prompt"] = context.get("final_prompt", "")

            await unified_memory.save_memory(
                agent_id=self.agent_id or "default",
                memory_type="dialog",
                content=f"Q: {user_input}\nA: {assistant_response[:200]}",
                metadata=metadata,
                importance=0.5
            )

            logger.debug(f"💾 对话已保存 | Session: {self.session_id}")

        except Exception as e:
            logger.warning(f"⚠️ 保存对话失败: {e}")

    async def _search_similar_successful_plans(
        self,
        user_input: str,
        top_k: int = 3,
        similarity_threshold: float = 0.85
    ) -> Optional[Dict[str, Any]]:
        """搜索历史中相似的成功计划（用于直接复用）"""
        try:
            from app.core.memory import unified_memory

            results = await unified_memory.search_memories(
                agent_id=self.agent_id or "default",
                query_text=user_input,
                memory_type="task",
                limit=top_k,
                use_semantic=True
            )

            if not results:
                logger.debug("📭 未找到历史成功任务")
                return None

            best_match = results[0]
            similarity = best_match.get("similarity", 0.0)

            if similarity >= similarity_threshold:
                logger.info(f"🎯 找到高相似度任务 | 相似度={similarity:.2%}")

                metadata = best_match.get("metadata", {})
                return {
                    "plan": metadata.get("plan"),
                    "tools_used": metadata.get("tools_used", []),
                    "similarity": similarity,
                    "execution_time": metadata.get("execution_time", 0.0),
                    "result_summary": metadata.get("result_summary", "")
                }
            else:
                logger.debug(f"📉 相似度不足 | {similarity:.2%} < {similarity_threshold:.2%}")
                return None

        except Exception as e:
            logger.warning(f"⚠️ 搜索历史计划失败: {e}")
            return None

    async def _search_historical_context(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索历史对话上下文（用于跨会话引用）"""
        try:
            from app.core.memory import unified_memory

            results = await unified_memory.search_memories(
                agent_id=self.agent_id or "default",
                query_text=query,
                memory_type="dialog",
                limit=top_k,
                use_semantic=True
            )

            if results:
                logger.info(f"📚 找到 {len(results)} 条相关历史对话")

                formatted_results = []
                for result in results:
                    metadata = result.get("metadata", {})
                    formatted_results.append({
                        "user_message": metadata.get("user_message", ""),
                        "assistant_message": metadata.get("assistant_message", ""),
                        "timestamp": metadata.get("timestamp", ""),
                        "similarity": result.get("similarity", 0.0),
                        "session_id": metadata.get("session_id", "")
                    })

                return formatted_results
            else:
                logger.debug("📭 未找到相关历史对话")
                return []

        except Exception as e:
            logger.warning(f"⚠️ 搜索历史对话失败: {e}")
            return []

    async def run(self, messages: List[Any], **kwargs) -> str:
        """兼容接口：非流式执行"""
        full_response = ""
        async for chunk in self._run_stream(messages, **kwargs):
            if isinstance(chunk, str):
                full_response += chunk
        return full_response

    # ═══════════════════════════════════════════════════════════
    # 🆕 详细日志记录函数
    # ═══════════════════════════════════════════════════════════

    def _log_llm_request(self, messages: List[Any], model: str, temperature: float, max_tokens: int):
        """记录LLM请求详情（包括提示词）"""
        if not settings.ENABLE_DETAILED_LOGGING or not settings.SHOW_LLM_PROMPTS:
            return

        logger.info(f"🤖 [LLM请求] 模型: {model}")
        logger.info(f"   🌡️ 温度: {temperature} | 📝 最大令牌: {max_tokens}")

        for i, message in enumerate(messages):
            # 处理不同格式的消息
            if hasattr(message, 'role'):
                role = message.role
            elif isinstance(message, dict):
                role = message.get('role', 'unknown')
            else:
                role = 'unknown'

            if hasattr(message, 'content'):
                content = message.content
            elif isinstance(message, dict):
                content = message.get('content', str(message))
            else:
                content = str(message)

            # 截断长内容
            if len(content) > settings.MAX_LOG_MESSAGE_LENGTH:
                content = content[:settings.MAX_LOG_MESSAGE_LENGTH] + "...(truncated)"

            logger.info(f"   💬 [{role.upper()}] 消息 {i+1}:\n{content}")

    def _log_llm_response(self, response: str, model: str):
        """记录LLM响应内容"""
        if not settings.ENABLE_DETAILED_LOGGING or not settings.SHOW_LLM_RESPONSES:
            return

        # 截断长响应
        if len(response) > settings.MAX_LOG_MESSAGE_LENGTH:
            response = response[:settings.MAX_LOG_MESSAGE_LENGTH] + "...(truncated)"

        logger.info(f"✅ [LLM响应] 模型: {model}")
        logger.info(f"   📤 响应内容:\n{response}")

    def _log_agent_execution_start(self, input_text: str, agent_name: str):
        """记录Agent执行开始"""
        if not settings.ENABLE_DETAILED_LOGGING or not settings.SHOW_EXECUTION_FLOW:
            return

        logger.info(f"🚀 [Agent执行开始] {agent_name}")
        logger.info(f"   📝 用户输入: {input_text[:200]}{'...' if len(input_text) > 200 else ''}")

    def _log_agent_execution_end(self, agent_name: str, duration: float, tools_used: List[str]):
        """记录Agent执行结束"""
        if not settings.ENABLE_DETAILED_LOGGING or not settings.SHOW_EXECUTION_FLOW:
            return

        logger.info(f"🏁 [Agent执行结束] {agent_name}")
        logger.info(f"   ⏱️ 执行时长: {duration:.2f}秒")
        logger.info(f"   🛠️ 使用工具: {', '.join(tools_used) if tools_used else '无'}")

    def _log_plan_generation(self, plan: Dict[str, Any], input_text: str):
        """记录计划生成过程"""
        if not settings.ENABLE_DETAILED_LOGGING or not settings.SHOW_PLAN_STEPS:
            return

        plan_type = plan.get("type", "unknown")
        confidence = plan.get("confidence", 0)
        reasoning = plan.get("reasoning", "")
        steps = plan.get("steps", [])

        logger.info(f"📋 [计划生成] 类型: {plan_type}")
        logger.info(f"   🎯 置信度: {confidence}")
        logger.info(f"   💭 推理: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")
        logger.info(f"   📋 步骤数: {len(steps)}")

        for i, step in enumerate(steps[:5]):  # 最多显示5个步骤
            # 处理不同格式的步骤
            if isinstance(step, dict):
                step_desc = step.get("description", "")
                step_tool = step.get("tool", "")
            elif isinstance(step, str):
                step_desc = step
                step_tool = "unknown"
            else:
                step_desc = str(step)
                step_tool = "unknown"

            logger.info(f"   步骤{i+1}: {step_desc[:100]}{'...' if len(step_desc) > 100 else ''} (工具: {step_tool})")

        if len(steps) > 5:
            logger.info(f"   ... 还有 {len(steps) - 5} 个步骤")

    def _log_tool_execution_pipeline(self, tools_executed: List[str], execution_success: bool, errors: List[str] = None):
        """记录工具执行流水线"""
        if not settings.ENABLE_DETAILED_LOGGING or not settings.SHOW_EXECUTION_FLOW:
            return

        logger.info(f"🔧 [工具流水线] 执行完成")
        logger.info(f"   🛠️ 执行工具: {', '.join(tools_executed) if tools_executed else '无'}")
        logger.info(f"   ✅ 执行状态: {'成功' if execution_success else '失败'}")

        if errors:
            for error in errors[:3]:  # 最多显示3个错误
                logger.error(f"   ❌ 错误: {error}")
            if len(errors) > 3:
                logger.error(f"   ... 还有 {len(errors) - 3} 个错误")

    def _get_parameter_filler(self):
        """安全获取parameter_filler实例（延迟初始化）"""
        if self.parameter_filler is None:
            try:
                logger.debug("🔧 开始延迟初始化 ParameterFiller...")
                from app.core.tools.parameter_filler import parameter_filler
                self.parameter_filler = parameter_filler
                logger.info("✅ ParameterFiller 延迟初始化成功")
            except ImportError as e:
                logger.error(f"❌ 无法导入 ParameterFiller: {e}")
                self.parameter_filler = None
                raise RuntimeError(f"ParameterFiller 导入失败: {e}")
            except Exception as e:
                logger.error(f"❌ ParameterFiller 初始化失败: {e}")
                self.parameter_filler = None
                raise RuntimeError(f"ParameterFiller 初始化失败: {e}")

        return self.parameter_filler

    def _should_direct_output(self, tool_name: str, tool_result: Any) -> bool:
        """
        检查工具是否应该直接输出结果

        Args:
            tool_name: 工具名称
            tool_result: 工具执行结果

        Returns:
            bool: 是否应该直接输出
        """
        try:
            # 从数据库获取工具配置
            tool_config = tool_config_service.get_tool_config(self._display_tool_name(tool_name))
            if not tool_config:
                return False

            # 检查是否启用了直接输出
            return tool_config.get_direct_output()

        except Exception as e:
            logger.warning(f"⚠️ 检查工具直接输出配置失败: {tool_name}, 错误: {e}")
            return False

    def _format_tool_result_for_prompt(self, result: Any) -> str:
        """格式化工具结果，优先保留完整的链接，避免被截断"""
        try:
            if isinstance(result, dict):
                if result.get("image_url"):
                    return str(result.get("image_url"))
                if result.get("message"):
                    url = self._extract_first_url(result.get("message", ""))
                    return url or result.get("message", "")
                return json.dumps(result, ensure_ascii=False)

            result_str = str(result)
            url = self._extract_first_url(result_str)
            if url:
                return url

            # 非链接结果做安全截断，防止提示过长
            return result_str if len(result_str) <= 1000 else result_str[:1000] + "..."
        except Exception as e:
            logger.debug(f"⚠️ 格式化工具结果失败: {e}")
            safe_str = str(result)
            return safe_str if len(safe_str) <= 1000 else safe_str[:1000] + "..."

    def _extract_first_url(self, text: str) -> Optional[str]:
        """提取文本中的首个URL"""
        match = re.search(r"https?://\S+", text)
        return match.group(0) if match else None
