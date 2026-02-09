"""
工具调用模板 - 基于LangChain实现

功能说明:
- 专门用于处理需要工具确认的场景
- 基于LangChain框架,完全异步,不阻塞事件循环
- 支持工具参数确认流程
- 继承BaseTemplateAgent,保持接口兼容性

模板类型: tool_calling

使用场景:
- 需要工具确认的智能体
- 需要调用外部平台工具的场景
- 需要处理复杂工具调用链的场景

作者: JetLinks Team
创建时间: 2025-10-10
"""

# ═══════════════════════════════════════════════════════════════
# 标准库导入
# ═══════════════════════════════════════════════════════════════
import logging
from typing import Dict, Any, Optional, AsyncIterator, List

# ═══════════════════════════════════════════════════════════════
# 项目内部导入
# ═══════════════════════════════════════════════════════════════
from app.core.agents.template_agent.base_template import BaseTemplateAgent
# from app.core.agents.template_agent.schemas.tool_calling_schemas import (
#     TOOL_CALLING_SCHEMAS,
#     TOOL_CALLING_PROMPTS
# )  # Schemas已删除，改为智能体独立JSON结构

logger = logging.getLogger(__name__)


class ToolCallingAgent(BaseTemplateAgent):
    """
    工具调用智能体模板

    基于LangChain实现,主要特点:
    1. 完全异步执行,不阻塞WebSocket消息发送
    2. 支持工具参数确认流程
    3. 支持流式输出
    4. 兼容BaseTemplateAgent接口

    使用方式:
    在数据库中设置 template='tool_calling' 即可使用此模板
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化工具调用智能体

        Args:
            config: 配置字典,包含:
                - name: 智能体名称
                - model: 模型名称
                - tools: 工具列表
                - system_prompt: 系统提示词
                - temperature: 温度参数
                - max_tokens: 最大token数
                - handler: LangChainJAIPHandler 引用（用于工具确认）
        """
        # logger.info("="*80)
        # logger.info("初始化 ToolCallingAgent (基于LangChain)")
        # logger.info("="*80)

        # 准备配置
        if config is None:
            config = {}

        # ✅ 调用父类初始化
        # 父类会自动创建 self.cognitive_engine
        super().__init__(
            name=config.get("name", "工具调用智能体"),
            description=config.get("description", "基于LangChain的工具调用智能体"),
            version="2.0.0",
            config=config
        )

        # ========== Schema-Based 支持 ==========
        # Schemas已删除，改为智能体独立JSON结构
        self.schema_definitions = {}  # 占位符
        self.schema_prompts = {}      # 占位符

        logger.info(f"✅ ToolCallingAgent 初始化完成: {self.name}")
        logger.info(f"  认知引擎已由基类创建")
        logger.info(f"  工具数量: {len(self.function_list)}")
        logger.info(f"  支持的 Schemas: {list(self.schema_definitions.keys())}")

    def _initialize(self):
        """
        子类初始化方法(BaseLangChainTemplateAgent要求实现)

        LangChain认知引擎已在 __init__ 中初始化，这里无需额外操作
        """
        pass

    async def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        处理用户消息(非流式)

        Args:
            message: 用户消息
            context: 上下文信息

        Returns:
            智能体响应
        """

        try:
            # 构建系统提示词
            system_prompt = self.get_system_prompt(context)

            # 构建消息列表
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]

            # 调用LangChain执行(异步)
            result = await self.cognitive_engine.run(messages)

            # 提取响应内容
            if result and isinstance(result, list) and len(result) > 0:
                last_msg = result[-1]
                if hasattr(last_msg, 'content'):
                    response = last_msg.content
                    return response

            return "抱歉,我无法处理您的请求。"

        except Exception as e:
            return f"处理失败: {str(e)}"

    async def process_message_stream(self, message: str, context: Dict[str, Any], original_params: Dict[str, Any]) -> AsyncIterator[str]:
        """
        流式处理用户消息

        Args:
            message: 用户消息
            context: 上下文信息

        Yields:
            响应片段
        """

        try:
            # 构建系统提示词
            agent_system_prompt = context.get("system_prompt", "")
            system_prompt = self.get_system_prompt(context) + "\n" + agent_system_prompt

            # 构建消息列表
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
                {"role": "tools", "content": self.function_list}  # 传递工具列表
            ]
            chunk_index = 0
            usage_tracker = context.get("usage_tracker") if isinstance(context, dict) else None
            async for chunk in self.cognitive_engine._run_stream(messages, usage_tracker=usage_tracker):
                chunk_index += 1
                if chunk is None:
                    continue

                if len(chunk) == 0:
                    continue
                yield chunk

        except Exception as e:
            yield f"处理失败: {str(e)}"


    def get_capabilities(self) -> List[str]:
        """
        获取智能体能力列表(BaseTemplateAgent要求实现)

        Returns:
            能力列表
        """
        capabilities = [
            "智能对话",
            "工具调用",
            "异步执行",
            "流式输出",
            "工具参数确认"
        ]

        # 添加已注册的工具
        if self.function_list:
            capabilities.append(f"支持 {len(self.function_list)} 个工具")

        return capabilities

    def get_default_prompt(self) -> str:
        """
        获取智能体的默认系统提示词(BaseTemplateAgent要求实现)

        Returns:
            默认系统提示词
        """
        return """你是一个智能助手，可以通过调用工具来帮助用户完成任务。

                在调用工具前，我会先向用户确认工具参数，用户可以选择：
                1. 确认执行 - 使用当前参数执行工具
                2. 修改参数 - 修改部分参数后再执行
                3. 拒绝执行 - 取消本次工具调用

                请根据用户需求，合理选择和使用工具。"""

    def get_supported_schemas(self) -> Dict[str, Any]:
        """获取支持的所有 schema 定义"""
        return self.schema_definitions

    def get_schema_definition(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """获取指定 schema 的定义"""
        return self.schema_definitions.get(schema_name)

    def get_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        构建系统提示词

        Args:
            context: 上下文信息

        Returns:
            系统提示词
        """
        # 基础提示词
        base_prompt = """你是一个智能助手,能够理解用户需求并调用相应的工具来完成任务。

            在调用工具时,请遵循以下原则:
            1. 仔细分析用户意图,选择最合适的工具
            2. 准确提取工具所需的参数
            3. 如果参数不明确,可以向用户询问
            4. 工具执行后,根据结果给出清晰的回复

            你可以使用以下工具:
            """

        # 添加工具列表
        if hasattr(self, 'function_list') and self.function_list:
            # logger.info(f"  [get_system_prompt] 开始构建工具指南,共 {len(self.function_list)} 个工具")

            tool_guide = "\n### 可用工具:\n\n"

            # 分类工具
            internal_tools = []
            external_tools = []

            for tool in self.function_list:
                tool_name = getattr(tool, 'name', 'Unknown')
                tool_desc = getattr(tool, 'description', 'No description')

                # 分类: 以 _ai_service_# 开头的是内部工具
                if tool_name.startswith('_ai_service_#'):
                    internal_tools.append((tool_name, tool_desc))
                else:
                    # 其他都是外部平台工具
                    external_tools.append((tool_name, tool_desc))

            # 先列出外部工具(更重要)
            if external_tools:
                tool_guide += "**外部平台工具** (优先使用):\n"
                for tool_name, tool_desc in external_tools:
                    tool_guide += f"- **{tool_name}**: {tool_desc}\n"
                tool_guide += "\n"

            # 再列出内部工具
            if internal_tools:
                tool_guide += "**AI内部工具**:\n"
                for tool_name, tool_desc in internal_tools:
                    tool_guide += f"- **{tool_name}**: {tool_desc}\n"

            tool_guide += f"\n**总计: {len(self.function_list)} 个工具**\n"

            base_prompt += tool_guide

        return base_prompt


# 兼容性导入
__all__ = ['ToolCallingAgent']
