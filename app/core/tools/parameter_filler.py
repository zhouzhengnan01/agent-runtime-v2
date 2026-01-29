"""
简化版参数填充器 - 纯LLM填充策略

核心理念：
1. 去掉复杂的多层逻辑，使用纯LLM填充
2. init_parameters和历史记录作为上下文参考传给LLM
3. LLM负责综合判断并填充所有参数
4. 逻辑简单，易于调试和维护

优势：
- 填充更准确（LLM看到完整上下文）
- 调试更简单（单一填充路径）
- 代码更简洁（去掉层级逻辑）
- 维护更容易（专注优化提示词）
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# 优雅加载环境变量
def _load_env_file():
    """优雅地加载项目根目录的 .env 文件"""
    try:
        from dotenv import load_dotenv
        current_path = Path(__file__).resolve()
        for parent in current_path.parents:
            env_path = parent / '.env'
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f"✅ 已加载 .env 文件: {env_path}")
                return True
        logger.warning("⚠️ 未找到 .env 文件")
        return False
    except ImportError:
        logger.warning("⚠️ python-dotenv 未安装，无法加载 .env 文件")
        return False

_load_env_file()


class ParameterFiller:
    """简化版参数填充器 - 纯LLM填充策略"""

    def __init__(self, llm=None, auto_init=True):
        """
        初始化简化版参数填充器

        Args:
            llm: LangChain LLM实例（可选）
            auto_init: 是否自动初始化LLM（默认True）
        """
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self.llm_initialized = False

        # 时间映射缓存
        self._time_cache = self._init_time_cache()

        # 初始化Redis管理器（用于获取历史记录）
        try:
            from app.core.memory.redis_chat_memory import redis_memory_manager
            self.redis_manager = redis_memory_manager
        except ImportError:
            self.logger.warning("⚠️ 无法导入 Redis 管理器，历史记录功能将被禁用")
            self.redis_manager = None

        # 自动初始化LLM
        if not self.llm and auto_init:
            self._auto_init_llm()

        logger.info("✅ ParameterFiller 初始化完成")

    def _init_time_cache(self) -> Dict[str, Any]:
        """初始化时间映射缓存"""
        import calendar

        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")

        # 时间戳
        now_timestamp = int(now.timestamp() * 1000)
        today_start = int(datetime(now.year, now.month, now.day, 0, 0, 0).timestamp() * 1000)
        today_end = int(datetime(now.year, now.month, now.day, 23, 59, 59).timestamp() * 1000)

        # 昨天时间范围
        yesterday_date = now - timedelta(days=1)
        yesterday_start = int(datetime(yesterday_date.year, yesterday_date.month, yesterday_date.day, 0, 0, 0).timestamp() * 1000)
        yesterday_end = int(datetime(yesterday_date.year, yesterday_date.month, yesterday_date.day, 23, 59, 59).timestamp() * 1000)

        # 本周和本月
        week_start_date = now - timedelta(days=now.weekday())
        week_start = int(datetime(week_start_date.year, week_start_date.month, week_start_date.day, 0, 0, 0).timestamp() * 1000)
        month_start = int(datetime(now.year, now.month, 1, 0, 0, 0).timestamp() * 1000)

        # 常用时间范围
        days_7_ago = int((now - timedelta(days=7)).timestamp() * 1000)
        days_30_ago = int((now - timedelta(days=30)).timestamp() * 1000)
        hours_1_ago = int((now - timedelta(hours=1)).timestamp() * 1000)
        hours_24_ago = int((now - timedelta(hours=24)).timestamp() * 1000)

        return {
            "now": now_timestamp,
            "today": today,
            "tomorrow": tomorrow,
            "yesterday": yesterday,
            "today_start": today_start,
            "today_end": today_end,
            "today_range": f"{today_start},{today_end}",
            "yesterday_start": yesterday_start,
            "yesterday_end": yesterday_end,
            "yesterday_range": f"{yesterday_start},{yesterday_end}",
            "week_start": week_start,
            "month_start": month_start,
            "recent_7_days": f"{days_7_ago},{now_timestamp}",
            "recent_30_days": f"{days_30_ago},{now_timestamp}",
            "recent_1_hour": f"{hours_1_ago},{now_timestamp}",
            "recent_24_hours": f"{hours_24_ago},{now_timestamp}"
        }

    def _auto_init_llm(self):
        """自动初始化LLM"""
        try:
            # 优先检查OPENAI兼容配置
            if os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"):
                self._init_openai_llm()
            # 检查DASHSCOPE配置
            elif os.getenv("DASHSCOPE_API_KEY"):
                self._init_dashscope_llm()
            else:
                self.logger.warning("⚠️ 未找到LLM配置，参数填充器将无法使用")
                return

            self.llm_initialized = True
            self.logger.info(f"✅ LLM自动初始化成功: {type(self.llm).__name__}")

        except Exception as e:
            self.logger.error(f"❌ LLM自动初始化失败: {e}")

    def _init_openai_llm(self):
        """初始化OpenAI兼容的LLM"""
        try:
            from langchain_openai import ChatOpenAI

            model_name = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL", "glm-4.6")
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            api_base = (os.getenv("OPENAI_API_BASE") or
                       os.getenv("LLM_BASE_URL") or
                       os.getenv("LLM_API_BASE"))

            if not api_key:
                raise ValueError("未找到有效的API_KEY配置")

            self.llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=api_base,
                temperature=0.1,  # 降低随机性，提高一致性
                streaming=False
            )
            self.logger.info(f"✅ OpenAI LLM初始化成功: {model_name}")

        except ImportError:
            self.logger.error("❌ langchain_openai 未安装")
            raise

    def _init_dashscope_llm(self):
        """初始化DashScope LLM"""
        try:
            from langchain_community.chat_models import ChatTongyi

            model_name = os.getenv("DASHSCOPE_MODEL", "qwen-turbo")
            api_key = os.getenv("DASHSCOPE_API_KEY")

            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")

            self.llm = ChatTongyi(
                model_name=model_name,
                dashscope_api_key=api_key,
                temperature=0.1,
                streaming=False
            )

        except ImportError:
            self.logger.error("❌ langchain_community 未安装")
            raise

    def set_llm(self, llm):
        """设置LLM实例"""
        self.llm = llm
        self.llm_initialized = True
        self.logger.info(f"✅ LLM已设置: {type(llm).__name__}")

    async def fill_parameters(
        self,
        parameters: Dict[str, Any],
        user_message: str,
        context: Dict[str, Any],
        tool_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        简化版参数填充 - 纯LLM填充策略

        Args:
            parameters: 参数定义（JSON Schema格式）
            user_message: 用户消息
            context: 上下文信息，包含：
                - init_parameters: 初始化参数（作为参考）
                - pre_filled_arguments: 预填充参数（作为建议）
                - history: 对话历史
                - session_id: 会话ID
                - agent_id: 智能体ID
                - system_prompt: 系统提示词
            tool_name: 工具名称

        Returns:
            填充的参数字典
        """
        self.logger.info(f"🚀 [简化填充] 开始填充参数: {tool_name}")
        self.logger.info(f"📝 用户消息: {user_message}")

        # 检查LLM是否可用
        if not self.llm:
            self.logger.warning("⚠️ LLM未初始化，使用基础规则填充")
            return self._fallback_fill_parameters(parameters, user_message, context)

        try:
            # 构建完整的提示词
            prompt = await self._build_comprehensive_prompt(
                parameters, user_message, context, tool_name
            )

            # 调用LLM
            result = await self._call_llm(prompt, tool_name)

            if result:
                self.logger.info(f"✅ [简化填充] LLM填充成功: {tool_name}")
                self.logger.info(f"🎯 填充结果: {json.dumps(result, ensure_ascii=False)}")
                return result
            else:
                self.logger.warning(f"⚠️ [简化填充] LLM填充失败，使用规则填充")
                return self._fallback_fill_parameters(parameters, user_message, context)

        except Exception as e:
            self.logger.error(f"❌ [简化填充] 参数填充异常: {e}")
            return self._fallback_fill_parameters(parameters, user_message, context)

    async def _build_comprehensive_prompt(
        self,
        parameters: Dict[str, Any],
        user_message: str,
        context: Dict[str, Any],
        tool_name: str
    ) -> str:
        """构建完整的LLM提示词"""

        # 1. 格式化参数定义
        params_text = self._format_parameters(parameters)

        # 2. 提取上下文信息
        init_parameters = context.get('init_parameters', {})
        pre_filled = context.get('pre_filled_arguments', {})
        system_prompt = context.get('system_prompt', '')[:200]  # 限制长度

        # 3. 获取对话历史
        history_text = self._format_conversation_history(context.get('history', []))

        # 4. 获取工具调用历史
        tool_history_text = await self._get_tool_call_history(context, tool_name)

        # 5. 格式化时间转换规则
        time_rules = self._format_time_rules()

        # 6. 构建最终提示词
        prompt = f"""你是专业的工具参数提取专家。请根据用户消息和上下文信息，智能填充工具的所有参数。

**工具信息**:
- 工具名称: {tool_name}
- 用户消息: {user_message}

**参数定义**:
{params_text}

**上下文参考信息**:
{self._format_context_info(init_parameters, pre_filled, system_prompt, history_text, tool_history_text)}

**时间转换规则**:
{time_rules}

**填充要求**:
1. 根据用户消息理解意图，填充所有必要的参数
2. 参考上下文信息，但以用户当前消息为准
3. 时间相关参数使用上述时间转换规则
4. 缺少信息的可选参数可以不填充
5. 必填参数必须填充，可以根据常识推理

**输出格式**:
只输出JSON格式的参数，不要任何解释：
{{
  "param1": "value1",
  "param2": "value2"
}}"""

        self.logger.info(f"🔍 [提示词构建] 完整提示词长度: {len(prompt)} 字符")
        self.logger.debug(f"🔍 [提示词详情]:\n{prompt}")

        return prompt

    def _format_parameters(self, parameters: Dict[str, Any]) -> str:
        """格式化参数定义为易读的文本"""
        properties = parameters.get('properties', {})
        required = parameters.get('required', [])

        if not properties:
            return "无参数定义"

        param_lines = []
        for name, prop in properties.items():
            param_type = prop.get('type', 'string')
            desc = prop.get('description', '无描述')
            req_flag = "【必填】" if name in required else "【可选】"

            # 添加默认值信息
            default_info = ""
            if 'default' in prop:
                default_info = f" (默认值: {prop['default']})"

            # 添加枚举值信息
            enum_info = ""
            if 'enum' in prop:
                enum_info = f" (可选值: {prop['enum']})"

            param_lines.append(
                f"- {name} ({param_type}) {req_flag}: {desc}{default_info}{enum_info}"
            )

        return "\n".join(param_lines)

    def _format_context_info(
        self,
        init_parameters: Dict[str, Any],
        pre_filled: Dict[str, Any],
        system_prompt: str,
        history_text: str,
        tool_history_text: str
    ) -> str:
        """格式化上下文信息"""
        context_parts = []

        if system_prompt:
            context_parts.append(f"- 系统角色: {system_prompt}")

        if init_parameters:
            init_text = ", ".join([f"{k}={v}" for k, v in init_parameters.items()])
            context_parts.append(f"- 初始参数: {init_text}")

        if pre_filled:
            pre_text = ", ".join([f"{k}={v}" for k, v in pre_filled.items()])
            context_parts.append(f"- 建议参数: {pre_text}")

        if history_text:
            context_parts.append(f"- 对话历史: {history_text}")

        if tool_history_text:
            context_parts.append(f"- 工具调用历史: {tool_history_text}")

        return "\n".join(context_parts) if context_parts else "无上下文信息"

    def _format_conversation_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        if not history:
            return ""

        # 只取最近3轮对话
        recent_history = history[-3:] if len(history) > 3 else history
        history_items = []

        for item in recent_history:
            role = item.get('role', 'user')
            content = item.get('content', '')[:100]  # 限制长度
            history_items.append(f"{role}: {content}")

        return " | ".join(history_items)

    async def _get_tool_call_history(self, context: Dict[str, Any], tool_name: str) -> str:
        """获取相关的工具调用历史"""
        if not self.redis_manager:
            return ""

        try:
            session_id = context.get('session_id')
            agent_id = context.get('agent_id')

            if not session_id:
                return ""

            # 获取工具调用历史
            tool_history = self.redis_manager.create_chat_history(
                session_id=session_id,
                agent_id=agent_id,
                key_prefix="tool_history"
            )

            if not tool_history:
                return ""

            # 解析最近的工具调用记录
            messages = tool_history.messages[-5:]  # 最近5条
            relevant_calls = []

            for msg in messages:
                try:
                    tool_call = json.loads(msg.content)
                    if tool_name in tool_call.get('tool_name', ''):
                        call_args = tool_call.get('arguments', {})
                        call_result = str(tool_call.get('result', ''))[:50]
                        relevant_calls.append(f"参数:{call_args} 结果:{call_result}")
                except:
                    continue

            if relevant_calls:
                return " | ".join(relevant_calls[-2:])  # 最多2个相关调用

        except Exception as e:
            self.logger.debug(f"获取工具历史失败: {e}")

        return ""

    def _format_time_rules(self) -> str:
        """格式化时间转换规则"""
        return f"""
- 现在/当前 → {self._time_cache['now']}
- 今天 → {self._time_cache['today_range']}
- 昨天 → {self._time_cache['yesterday_range']}
- 最近7天 → {self._time_cache['recent_7_days']}
- 最近30天 → {self._time_cache['recent_30_days']}
- 最近1小时 → {self._time_cache['recent_1_hour']}
- 最近24小时 → {self._time_cache['recent_24_hours']}"""

    async def _call_llm(self, prompt: str, tool_name: str) -> Dict[str, Any]:
        """调用LLM获取参数填充结果"""
        try:
            from langchain_core.messages import HumanMessage

            self.logger.info(f"🤖 [LLM调用] 开始调用LLM: {tool_name}")

            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()

            self.logger.info(f"📤 [LLM响应] 原始响应长度: {len(response_text)} 字符")
            self.logger.debug(f"📤 [LLM响应内容]:\n{response_text}")

            # 解析JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                self.logger.info(f"✅ [JSON解析] 成功解析参数: {len(result)} 个")
                return result
            else:
                self.logger.warning(f"⚠️ [JSON解析] 无法找到JSON格式: {response_text[:200]}")
                return {}

        except Exception as e:
            self.logger.error(f"❌ [LLM调用] 调用失败: {e}")
            import traceback
            self.logger.debug(f"🔍 [错误堆栈]:\n{traceback.format_exc()}")
            return {}

    def _fallback_fill_parameters(
        self,
        parameters: Dict[str, Any],
        user_message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """LLM失败时的基础规则填充"""
        self.logger.info("🔧 [规则填充] 使用基础规则填充参数")

        result = {}
        properties = parameters.get('properties', {})
        init_parameters = context.get('init_parameters', {})
        pre_filled = context.get('pre_filled_arguments', {})

        for param_name, param_def in properties.items():
            param_type = param_def.get('type', 'string')

            # 1. 优先使用预填充参数
            if param_name in pre_filled:
                result[param_name] = pre_filled[param_name]
                continue

            # 2. 使用初始化参数
            if param_name in init_parameters:
                result[param_name] = init_parameters[param_name]
                continue

            # 3. 使用参数定义的默认值
            if 'default' in param_def:
                result[param_name] = param_def['default']
                continue

            # 4. 简单的规则推理
            if param_type == 'string':
                if 'time' in param_name.lower() or 'date' in param_name.lower():
                    if '今天' in user_message:
                        result[param_name] = self._time_cache['today_range']
                    elif '昨天' in user_message:
                        result[param_name] = self._time_cache['yesterday_range']

            elif param_type in ['number', 'integer']:
                numbers = re.findall(r'\d+', user_message)
                if numbers:
                    result[param_name] = int(numbers[0])

        self.logger.info(f"🔧 [规则填充] 填充了 {len(result)} 个参数")
        return result

    # 兼容性方法，保持接口一致
    async def fill_parameters_layered(self, *args, **kwargs):
        """兼容性方法，重定向到新的填充方法"""
        return await self.fill_parameters(*args, **kwargs)

    def fill_parameters_sync(self, *args, **kwargs):
        """同步版本的参数填充（兼容性）"""
        return self._fallback_fill_parameters(*args, **kwargs)

    async def fill_parameters_with_confirmation(
        self,
        tool: Any,
        user_message: str,
        context: Dict[str, Any],
        pre_filled_arguments: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        参数填充并生成JAIP协议的tools.confirm格式

        Args:
            tool: 工具实例
            user_message: 用户消息
            context: 上下文信息
            pre_filled_arguments: 预填充参数

        Returns:
            JAIP协议的tools.confirm消息
        """
        import uuid

        # 获取工具信息
        tool_name = getattr(tool, 'name', 'unknown')
        tool_description = getattr(tool, 'description', '')
        parameters = getattr(tool, 'parameters', {})

        # 生成执行ID
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"

        # 合并预填充参数到上下文
        if pre_filled_arguments:
            context = dict(context)
            existing_pre_filled = context.get('pre_filled_arguments', {})
            context['pre_filled_arguments'] = {**existing_pre_filled, **pre_filled_arguments}

        # 使用简化填充逻辑
        filled_params = await self.fill_parameters(parameters, user_message, context, tool_name)

        # 转换为inputs格式
        inputs = self._convert_parameters_to_inputs(parameters)

        # 构建tools.confirm格式
        confirmation = {
            'jsonrpc': '2.0',
            'method': 'tools.confirm',
            'id': execution_id,
            'params': {
                'message': f'需要调用工具：{tool_name}',
                'call': {
                    'toolName': tool_name.replace('_ai_service_#', ''),
                    'arguments': filled_params,
                    'executionId': execution_id
                },
                'command': {
                    'id': tool_name.replace('_ai_service_#', ''),
                    'name': tool_name.replace('_ai_service_#', ''),
                    'description': tool_description,
                    'inputs': inputs
                }
            }
        }

        # 知识库工具自动执行
        if 'knowledgeService#Search' == tool_name:
            confirmation['params']['auto'] = True

        self.logger.info(f"✅ 生成工具确认消息: {tool_name}")
        return confirmation

    def _convert_parameters_to_inputs(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将JSON Schema参数转换为inputs列表格式"""
        if not isinstance(parameters, dict):
            return []

        properties = parameters.get('properties', {})
        required_fields = parameters.get('required', [])

        inputs = []
        for param_id, param_def in properties.items():
            input_item = {
                'id': param_id,
                'name': param_def.get('title', param_id),
                'type': param_def.get('type', 'string'),
                'description': param_def.get('description', ''),
                'required': param_id in required_fields
            }

            if 'enum' in param_def:
                input_item['enum'] = param_def['enum']
            if 'default' in param_def:
                input_item['default'] = param_def['default']

            inputs.append(input_item)

        return inputs

    # 一些兼容性和工具方法
    def save_tool_call_history(self, session_id, tool_name, arguments, result, agent_id=None):
        """保存工具调用记录（兼容性方法）"""
        try:
            from app.core.memory.redis_chat_memory import redis_memory_manager
            from langchain_core.messages import HumanMessage
            import json
            from datetime import datetime

            if not self.redis_manager:
                return

            history = self.redis_manager.create_chat_history(
                session_id=session_id,
                agent_id=agent_id,
                key_prefix="tool_history"
            )

            if history:
                tool_call_record = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": str(result)[:500],
                    "timestamp": datetime.now().isoformat()
                }
                history.add_message(HumanMessage(content=json.dumps(tool_call_record, ensure_ascii=False)))
                logger.info(f"💾 已保存工具调用记录: {tool_name}")

        except Exception as e:
            logger.warning(f"⚠️ 保存工具历史失败: {e}")

    def load_tool_call_history(self, session_id, agent_id=None):
        """加载工具调用历史（兼容性方法）"""
        try:
            if not self.redis_manager:
                return []

            history = self.redis_manager.create_chat_history(
                session_id=session_id,
                agent_id=agent_id,
                key_prefix="tool_history"
            )

            if not history:
                return []

            tool_calls = []
            for msg in history.messages[-10:]:
                try:
                    tool_call = json.loads(msg.content)
                    tool_calls.append(tool_call)
                except:
                    continue

            return tool_calls

        except Exception as e:
            logger.warning(f"⚠️ 加载工具历史失败: {e}")
            return []


# 全局实例
parameter_filler = ParameterFiller()


def get_parameter_filler() -> ParameterFiller:
    """获取全局参数填充器实例"""
    return parameter_filler


def init_parameter_filler(llm):
    """初始化全局参数填充器"""
    global parameter_filler
    parameter_filler.set_llm(llm)
    return parameter_filler