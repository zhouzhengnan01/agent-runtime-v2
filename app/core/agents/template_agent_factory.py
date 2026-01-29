import logging
import importlib
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from app.core.agents.template_agent.base_template import BaseTemplateAgent
from app.models.agent_template import AgentTemplate
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


class TemplateAgentFactory:
    """
    LangChain 模板智能体工厂

    功能:
    1. 根据 template 字段创建 LangChain 智能体
    2. 支持从数据库动态加载模板
    3. 纯 LangChain 实现，无 qwen-agent 依赖
    4. 支持工具确认、流式输出、认知能力

    支持的模板:
    - tool_calling: 工具调用智能体 (LangChain + 工具确认)
    - 未来: chat, rag, code_generator 等

    使用示例:
    ```python
    config = {
        "template": "tool_calling",
        "name": "我的智能体",
        "model": "qwen-max",
        "tools": [tool1, tool2],
        "handler": jaip_handler,
        "enable_memory": True,
        "enable_compression": True,
        "enable_multimodal": True,
    }

    agent = LangChainTemplateAgentFactory.create_agent(config)
    ```

    架构:
    ```
    LangChainTemplateAgentFactory (工厂)
        ↓ 创建
    BaseTemplateAgent (基类)
        ↓ 子类
    ├─ ToolCallingAgent (工具调用)
    ├─ ChatAgent (对话)
    ├─ RAGAgent (检索增强)
    └─ ... (更多模板)
    ```
    """

    # ═══════════════════════════════════════════════════════════════
    # 模板注册表 (LangChain only)
    # ═══════════════════════════════════════════════════════════════
    TEMPLATE_REGISTRY: Dict[str, type] = {
        # ✅ 内置模板兜底：即使数据库未初始化/未写入 agent_templates，也能正常创建常用智能体
        # 具体类在 _register_builtin_templates 中延迟导入
    }

    # 模板缓存
    _template_cache: Dict[str, type] = {}
    _cache_loaded: bool = False
    _builtin_loaded: bool = False

    # 兼容历史模板ID（数据库里/前端里可能仍在用旧名称）
    TEMPLATE_ALIASES: Dict[str, str] = {
        # tool calling
        "tool_calling_template": "tool_calling",
        "tool_calling_agent": "tool_calling",
        "tool-calling": "tool_calling",
        "general": "tool_calling",
        "jetlinks_patrol": "video_patrol",
        "jetlinks_patrol_template": "video_patrol",
    }

    @classmethod
    def _register_builtin_templates(cls) -> None:
        """注册内置模板（延迟导入，避免启动阶段强依赖 DB 模板配置）。"""
        if cls._builtin_loaded:
            return

        builtin: List[tuple[str, str, str]] = [
            ("tool_calling", "app.core.agents.template_agent.tool_calling_template", "ToolCallingAgent"),
            ("video_inspection", "app.core.agents.template_agent.video_inspection_template", "VideoInspectionAgent"),
            ("video_patrol", "app.core.agents.template_agent.video_patrol_template", "VideoPatrolAgent"),
        ]

        for template_id, module_path, class_name in builtin:
            try:
                module = importlib.import_module(module_path)
                agent_class = getattr(module, class_name)
                cls.TEMPLATE_REGISTRY.setdefault(template_id, agent_class)
                cls._template_cache.setdefault(template_id, agent_class)
                logger.info(f"  ✅ 已注册内置模板: {template_id} → {class_name}")
            except Exception as e:
                logger.warning(f"  ⚠️ 内置模板注册失败: {template_id} ({module_path}.{class_name}) | {e}")

        cls._builtin_loaded = True

    @classmethod
    def _load_templates_from_db(cls):
        """
        从数据库加载 LangChain 模板配置

        工作流程:
        1. 查询 agent_templates 表
        2. 筛选 framework='langchain' 的模板
        3. 动态导入模板类
        4. 注册到 TEMPLATE_REGISTRY

        注意:
        - 只加载 LangChain 模板
        - qwen-agent 模板会被忽略
        """
        if cls._cache_loaded:
            return

        # 先注册内置模板，保证 DB 为空时依然可用
        cls._register_builtin_templates()

        logger.info("📥 [LangChain工厂] 从数据库加载模板配置...")

        try:
            db: Session = SessionLocal()
            try:
                # 查询所有启用的 LangChain 模板
                templates = db.query(AgentTemplate).filter(
                    AgentTemplate.is_active == True,
                    AgentTemplate.framework == "langchain"  # 只加载 LangChain 模板
                ).all()

                if not templates:
                    # DB 未配置模板时，使用内置模板即可（避免阻断会话初始化）
                    logger.info("ℹ️  数据库中未找到 LangChain 模板配置，已使用内置模板兜底")
                    return

                logger.info(f"找到 {len(templates)} 个 LangChain 模板")

                for template in templates:
                    try:
                        class_path = template.class_path
                        module_path, class_name = class_path.rsplit('.', 1)

                        module = importlib.import_module(module_path)
                        agent_class = getattr(module, class_name)

                        # 注册到缓存和注册表（DB 配置优先级更高，可覆盖内置模板）
                        cls._template_cache[template.id] = agent_class
                        cls.TEMPLATE_REGISTRY[template.id] = agent_class

                        logger.info(f"  ✅ 已加载(DB): {template.id} → {class_name}")
                        logger.info(f"     路径: {class_path}")
                        logger.info(f"     框架: {template.framework}")

                    except Exception as e:
                        logger.error(f"  ❌ 加载模板失败(DB): {template.id}")
                        logger.error(f"     原因: {e}")
                        continue

                logger.info(f"✅ [LangChain工厂] 模板加载完成，共 {len(cls.TEMPLATE_REGISTRY)} 个")

            finally:
                db.close()

        except Exception as e:
            # DB 连接失败时，也不阻断服务：依赖内置模板继续工作
            logger.warning(f"⚠️  [LangChain工厂] 数据库加载失败，已使用内置模板兜底: {e}")
        finally:
            # 标记已尝试加载，避免每次创建都重复打 DB / 重复日志
            cls._cache_loaded = True

    @classmethod
    def create_agent(cls, config: Dict[str, Any]) -> BaseTemplateAgent:
        """
        创建 LangChain 智能体实例

        Args:
            config: 配置字典，包含:
                - template (str): 模板类型 (如 "tool_calling")
                - name (str): 智能体名称
                - model (str): 模型名称 (如 "qwen-max")
                - tools (List): 工具列表
                - system_prompt (str): 系统提示词
                - handler: LangChainJAIPHandler 引用（用于工具确认）
                - enable_memory (bool): 启用记忆 (默认 True)
                - enable_compression (bool): 启用压缩 (默认 True)
                - enable_multimodal (bool): 启用多模态 (默认 True)
                - max_context_tokens (int): 压缩阈值 (默认 6000)
                - temperature (float): 温度参数 (默认 0.7)
                - max_tokens (int): 最大token (默认 2000)

        Returns:
            BaseTemplateAgent: 智能体实例

        Raises:
            ValueError: 如果 template 不存在或配置无效

        示例:
        ```python
        config = {
            "template": "tool_calling",
            "name": "工具调用测试",
            "model": "qwen-max",
            "tools": [tool1, tool2],
            "handler": handler,
            "enable_memory": True,
        }

        agent = LangChainTemplateAgentFactory.create_agent(config)
        ```
        """
        logger.info("=" * 80)
        logger.info("[LangChain工厂] 开始创建智能体")
        logger.info("=" * 80)

        # 内置模板兜底（先注册，避免 DB 未配置时无法创建）
        cls._register_builtin_templates()

        # ───────────────────────────────────────────────────────────
        # 步骤1: 从数据库加载模板（首次调用）
        # ───────────────────────────────────────────────────────────
        if not cls._cache_loaded:
            cls._load_templates_from_db()

        # ───────────────────────────────────────────────────────────
        # 步骤2: 提取 template 字段
        # ───────────────────────────────────────────────────────────
        template = config.get("template", "tool_calling")  # 默认使用 tool_calling
        template = cls.TEMPLATE_ALIASES.get(template, template)
        logger.info(f"[LangChain工厂] 模板类型: {template}")

        # ───────────────────────────────────────────────────────────
        # 步骤3: 查找模板类
        # ───────────────────────────────────────────────────────────
        if template not in cls.TEMPLATE_REGISTRY:
            available = list(cls.TEMPLATE_REGISTRY.keys())
            logger.error(f"❌ [LangChain工厂] 模板 '{template}' 未注册")
            logger.error(f"   可用模板: {available}")
            raise ValueError(
                f"LangChain 模板 '{template}' 不存在。"
                f"可用模板: {available}。"
                f"请在 agent_templates 表中添加 framework='langchain' 的模板配置。"
            )

        template_class = cls.TEMPLATE_REGISTRY[template]
        logger.info(f"✅ [LangChain工厂] 找到模板类: {template_class.__name__}")
        logger.info(f"   模块路径: {template_class.__module__}")

        # ───────────────────────────────────────────────────────────
        # 步骤4: 准备配置参数
        # ───────────────────────────────────────────────────────────
        import os
        agent_params = {
            "name": config.get("name", "LangChain智能体"),
            "model": config.get("model", os.getenv("LLM_MODEL", "qwen-turbo")),
            "system_prompt": config.get("system_prompt", ""),
            "tools": config.get("tools", []),
            "handler": config.get("handler", None),
            "temperature": config.get("temperature", 0.3),
            "max_tokens": config.get("max_tokens", 2000),

            # 认知能力配置
            "enable_memory": config.get("enable_memory", True),
            "enable_compression": config.get("enable_compression", True),
            "enable_multimodal": config.get("enable_multimodal", True),
            "max_context_tokens": config.get("max_context_tokens", 6000),

            # 其他配置
            **{k: v for k, v in config.items() if k not in [
                "template", "name", "model", "system_prompt", "tools",
                "handler", "temperature", "max_tokens", "enable_memory",
                "enable_compression", "enable_multimodal", "max_context_tokens"
            ]}
        }

        logger.info(f"[LangChain工厂] 配置参数:")
        logger.info(f"  - name: {agent_params['name']}")
        logger.info(f"  - model: {agent_params['model']}")
        logger.info(f"  - tools: {len(agent_params['tools'])} 个")
        logger.info(f"  - handler: {'已提供' if agent_params['handler'] else '未提供'}")
        logger.info(f"  - enable_memory: {agent_params['enable_memory']}")
        logger.info(f"  - enable_compression: {agent_params['enable_compression']}")
        logger.info(f"  - enable_multimodal: {agent_params['enable_multimodal']}")

        # ───────────────────────────────────────────────────────────
        # 步骤5: 创建智能体实例
        # ───────────────────────────────────────────────────────────
        try:
            agent_instance = template_class(agent_params)
            logger.info(f"✅ [LangChain工厂] 智能体创建成功: {agent_instance.name}")
            logger.info(f"   Agent ID: {agent_instance.agent_id}")
            logger.info(f"   版本: {agent_instance.version}")
            logger.info(f"   工具数量: {len(agent_instance.function_list)}")
            logger.info("=" * 80)
            return agent_instance

        except Exception as e:
            logger.error(f"❌ [LangChain工厂] 智能体创建失败: {e}", exc_info=True)
            raise

    @classmethod
    def register_template(cls, template_id: str, template_class: type):
        """
        手动注册模板

        Args:
            template_id: 模板ID (如 "tool_calling")
            template_class: 模板类 (必须继承 BaseTemplateAgent)

        示例:
        ```python
        class MyCustomAgent(BaseTemplateAgent):
            pass

        LangChainTemplateAgentFactory.register_template("my_custom", MyCustomAgent)
        ```
        """
        if not issubclass(template_class, BaseTemplateAgent):
            raise ValueError(
                f"模板类 {template_class.__name__} 必须继承 BaseTemplateAgent"
            )

        cls.TEMPLATE_REGISTRY[template_id] = template_class
        logger.info(f"✅ [LangChain工厂] 手动注册模板: {template_id} → {template_class.__name__}")

    @classmethod
    def list_templates(cls) -> List[str]:
        """
        列出所有可用的模板

        Returns:
            List[str]: 模板ID列表

        示例:
        ```python
        templates = LangChainTemplateAgentFactory.list_templates()
        # ['tool_calling', 'chat', 'rag']
        ```
        """
        if not cls._cache_loaded:
            cls._load_templates_from_db()

        return list(cls.TEMPLATE_REGISTRY.keys())

    @classmethod
    def get_template_info(cls, template_id: str) -> Optional[Dict[str, Any]]:
        """
        获取模板信息

        Args:
            template_id: 模板ID

        Returns:
            Dict: 模板信息，包括:
                - id: 模板ID
                - name: 类名
                - module: 模块路径
                - capabilities: 能力列表（如果实现了 get_capabilities）

        示例:
        ```python
        info = LangChainTemplateAgentFactory.get_template_info('tool_calling')
        # {
        #     'id': 'tool_calling',
        #     'name': 'ToolCallingAgent',
        #     'module': 'app.core.agents.template_agent.tool_calling_template',
        #     'capabilities': ['tool_calling', 'streaming', 'memory', ...]
        # }
        ```
        """
        if not cls._cache_loaded:
            cls._load_templates_from_db()

        if template_id not in cls.TEMPLATE_REGISTRY:
            return None

        template_class = cls.TEMPLATE_REGISTRY[template_id]

        return {
            "id": template_id,
            "name": template_class.__name__,
            "module": template_class.__module__,
            "doc": template_class.__doc__,
        }
