"""
工具选择器 - 简化版

核心功能：
1. 工具选择（select_tools）：从工具列表中智能选择最相关的1个工具
2. 参数填充（fill_parameters）：AI自动填充工具所需参数

使用示例：
    # 1. 工具选择（自动选择最相关的1个工具）
    selected = await tool_selector.select_tools(
        user_message="查询设备ID为device_001的详情",
        tool_instances=tool_list
    )
    # 返回: [{"tool": tool_obj, "score": 1.0, "reason": "最相关工具"}]

    # 2. 参数填充（直接传入参数定义）
    parameters = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"},
            "date": {"type": "string", "description": "日期"}
        },
        "required": ["city"]
    }
    params = await tool_selector.fill_parameters(
        parameters=parameters,
        user_message="今天北京的天气",
        tool_name="weather"
    )
    # 返回: {"city": "北京", "date": "2025-10-16"}

    # 或者使用便捷方法（传入工具实例）
    params = await tool_selector.fill_tool_parameters(
        tool=selected[0]["tool"],
        user_message="今天北京的天气"
    )
    # 返回: {"city": "北京", "date": "2025-10-16"}
"""

import logging
import re
import os
import json
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    # 查找项目根目录的 .env 文件
    current_dir = Path(__file__).resolve()
    project_root = current_dir.parent.parent.parent.parent  # 回到 jetlinks-agent 目录
    env_path = project_root / '.env'

    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"✅ 已加载 .env 文件: {env_path}")
    else:
        logger.warning(f"⚠️ 未找到 .env 文件: {env_path}")
except ImportError:
    logger.warning("⚠️ python-dotenv 未安装，无法加载 .env 文件")


class ToolSelector:
    """智能工具选择器"""

    def __init__(self):
        """初始化工具选择器"""
        self.llm = None
        self.llm_initialized = False
        # 使用统一的 Redis 记忆管理器
        from app.core.memory.redis_chat_memory import redis_memory_manager
        self.redis_manager = redis_memory_manager
        self._cached_arguments = {}  # 缓存工具的填充参数（tool_name -> arguments）
        logger.info("✅ ToolSelector 初始化完成（延迟加载 LLM）")

    def _build_param_summary(self, parameters: Dict[str, Any]) -> str:
        """构建参数摘要（压缩显示）"""
        if not parameters or not isinstance(parameters, dict):
            return "无参数"

        properties = parameters.get('properties', {})
        required = parameters.get('required', [])

        if not properties:
            return "无参数"

        # 只显示参数名和是否必填
        param_list = []
        for param_name in list(properties.keys())[:5]:  # 最多显示5个参数
            is_required = "必填" if param_name in required else "可选"
            param_type = properties[param_name].get('type', 'string')
            param_list.append(f"{param_name}({param_type},{is_required})")

        result = ", ".join(param_list)
        if len(properties) > 5:
            result += f" ...等{len(properties)}个参数"

        return result

    def load_tool_call_history(self, session_id: str, agent_id: str = None) -> List[Dict[str, Any]]:
        """从 Redis 加载工具调用历史

        Args:
            session_id: 会话ID
            agent_id: 智能体ID（可选）

        Returns:
            工具调用历史列表
        """
        try:
            # 使用统一的 Redis 管理器创建历史实例
            history = self.redis_manager.create_chat_history(
                session_id=session_id,
                agent_id=agent_id,
                key_prefix="tool_history"
            )

            if not history:
                return []

            # 读取消息并解析为工具调用记录
            messages = history.messages
            tool_calls = []

            for msg in messages[-10:]:  # 只取最近10条
                try:
                    import json
                    # 消息内容是 JSON 格式的工具调用记录
                    tool_call = json.loads(msg.content)
                    tool_calls.append(tool_call)
                except:
                    continue

            logger.info(f"📝 [Redis工具历史] 加载了 {len(tool_calls)} 条记录")
            return tool_calls

        except Exception as e:
            logger.warning(f"⚠️ 从 Redis 加载工具历史失败: {e}")
            return []

    def save_tool_call_history(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        agent_id: str = None
    ):
        """保存工具调用记录到 Redis

        Args:
            session_id: 会话ID
            tool_name: 工具名称
            arguments: 工具参数
            result: 工具执行结果
            agent_id: 智能体ID（可选）
        """
        try:
            from langchain_core.messages import HumanMessage
            import json
            from datetime import datetime

            # 使用统一的 Redis 管理器创建历史实例
            history = self.redis_manager.create_chat_history(
                session_id=session_id,
                agent_id=agent_id,
                key_prefix="tool_history"
            )

            if not history:
                return

            # 构建工具调用记录
            tool_call_record = {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": str(result)[:500],  # 截断过长结果
                "timestamp": datetime.now().isoformat()
            }

            # 将工具调用记录序列化为消息
            history.add_message(HumanMessage(content=json.dumps(tool_call_record, ensure_ascii=False)))

            logger.info(f"💾 [Redis工具历史] 已保存: {tool_name}")

        except Exception as e:
            logger.warning(f"⚠️ 保存工具历史到 Redis 失败: {e}")

    def _ensure_llm_initialized(self):
        """确保 LLM 已初始化（按需加载）"""
        if self.llm_initialized:
            return True

        if self.llm is not None:
            self.llm_initialized = True
            return True

        try:
            # 检查是否使用本地模型
            use_local_model = os.getenv("USE_LOCAL_QWEN_MODEL", "false").lower() == "true"

            if use_local_model:
                # 使用本地 qwen 模型
                from app.core.llm import get_llm

                model_name = os.getenv("LOCAL_QWEN_MODEL", "qwen2.5-72b-instruct")
                api_base = os.getenv("LOCAL_QWEN_API_BASE", "http://localhost:8000/v1")
                api_key = os.getenv("LOCAL_QWEN_API_KEY", "EMPTY")

                self.llm = get_llm(
                    model_name=model_name,
                    api_base=api_base,
                    api_key=api_key,
                    temperature=0.1,
                    max_tokens=2000
                )

                self.llm_initialized = True
                logger.info(f"✅ 本地 Qwen 模型初始化成功: {model_name}")
                return True
            else:
                # 使用系统通用LLM配置
                try:
                    from app.core.llm.langchain_factory import create_langchain_llm

                    # 使用系统的通用LLM工厂函数
                    # 这会自动使用系统配置的模型、API Key等
                    self.llm = create_langchain_llm(
                        temperature=0.1,
                        max_tokens=2000
                    )

                    self.llm_initialized = True
                    logger.info(f"✅ 工具选择器使用系统通用LLM配置初始化成功")
                    return True

                except Exception as e:
                    logger.error(f"❌ 使用系统LLM配置失败: {e}")
                    # 降级到手动配置
                    try:
                        # 优先使用 langchain-openai 包（新版）
                        from langchain_openai import ChatOpenAI
                        use_new_api = True
                    except ImportError:
                        # 回退到 langchain-community 包（旧版）
                        from langchain_community.chat_models import ChatOpenAI
                        use_new_api = False

                    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        logger.error("❌ 未找到 API Key，无法初始化 LLM")
                        return False

                    model = os.getenv("TOOL_SELECTOR_MODEL", "qwen-max")
                    api_base = os.getenv("TOOL_SELECTOR_API_BASE",
                                        "https://dashscope.aliyuncs.com/compatible-mode/v1")

                    # 根据版本使用不同的参数名
                    if use_new_api:
                        self.llm = ChatOpenAI(
                            model=model,
                            temperature=0.1,
                            max_tokens=2000,
                            base_url=api_base,
                            api_key=api_key
                        )
                    else:
                        self.llm = ChatOpenAI(
                            model=model,
                            temperature=0.1,
                            max_tokens=2000,
                            openai_api_base=api_base,
                            openai_api_key=api_key
                        )

                    self.llm_initialized = True
                    logger.info(f"✅ LLM 手动初始化成功: {model}")
                    return True

        except Exception as e:
            logger.error(f"❌ LLM 初始化失败: {e}")
            return False

    async def select_tools(
        self,
        user_message: str,
        tool_instances: List[Any],
        agent_context: Dict[str, Any],
        top_k: int = 1  # 保留参数（兼容性），实际不使用
    ) -> List[Dict[str, Any]]:
        """工具选择：从工具列表中智能选择最合适的工具（同时返回填充好的参数）

        【重要】该方法现在一次性完成：工具选择 + 参数填充

        Args:
            user_message: 用户消息
            tool_instances: 工具实例列表
            agent_context: Agent上下文（必需），包含：
                - system_prompt: 系统提示词
                - agent_prompt: 智能体提示词
                - init_parameters: 初始化参数
            top_k: 保留参数（兼容性），实际不使用

        Returns:
            选中的工具列表，格式：

            # 找到合适工具时：
            [
                {
                    "tool": tool_instance,    # 工具实例对象
                    "score": 1.0,             # 相关度评分（固定为1.0）
                    "reason": "最相关工具",   # 选择理由
                    "arguments": {...}        # 【新增】填充好的参数
                }
            ]

            # 没有合适工具时：
            [
                {
                    "tool": None,             # None 表示无工具
                    "score": 0.0,             # 评分为0
                    "reason": "没有合适的工具", # 原因说明
                    "arguments": {}           # 空参数
                }
            ]

        使用示例：
            selected = await tool_selector.select_tools(
                user_message="查询设备状态",
                tool_instances=tools,
                agent_context={
                    "system_prompt": "你是专业的运维助手",
                    "agent_prompt": "帮助用户管理设备",
                    "init_parameters": {"user_role": "运维工程师"}
                }
            )
            # selected[0]['arguments'] 中已包含填充好的参数
        """
        _ = top_k  # 忽略参数（兼容性）

        logger.info(f"🔍 开始工具选择+参数填充（一次性完成）")
        logger.info(f"  用户消息: {user_message[:50]}...")
        logger.info(f"  可用工具数: {len(tool_instances)}")

        # 检测是否为工具结果回复（通过提示词特征判断）
        # 必须在消息开头（忽略前导空白）
        message_start = user_message.lstrip()[:50]  # 取开头50字符检测
        if message_start.startswith('用户问题：') or '<tool> 工具回复消息:' in message_start:
            logger.info("⚠️ 检测到工具结果回复阶段（通过提示词特征），跳过工具选择")
            return []  # 返回空列表，表示不选择任何工具

        if not tool_instances:
            logger.warning("⚠️ 工具列表为空")
            return []

        # 初始化 LLM
        if not self._ensure_llm_initialized():
            logger.error("❌ LLM 初始化失败，无法进行工具选择")
            return []

        try:
            # 构建工具列表描述（包含参数schema）
            tools_with_params = []
            tool_map = {}  # 简化名 -> 完整工具对象
            has_knowledge_tool = False

            for i, tool in enumerate(tool_instances, 1):
                tool_full_name = getattr(tool, 'name', 'unknown')
                tool_description = getattr(tool, 'description', '')
                tool_parameters = getattr(tool, 'parameters', {})

                # 检查是否有知识库工具
                if 'knowledgeService#Search' in tool_full_name:
                    has_knowledge_tool = True

                # 直接使用完整工具名，不做简化
                tool_map[tool_full_name] = tool

                # 【DEBUG】打印工具注册信息
                logger.info(f"🔧 注册工具: {tool_full_name}")

                # 提取关键参数（压缩显示）
                param_summary = self._build_param_summary(tool_parameters)

                tools_with_params.append(
                    f"{i}. **{tool_full_name}**\n"
                    f"   描述: {tool_description}\n"
                    f"   参数: {param_summary}"
                )

            tools_text = "\n\n".join(tools_with_params)

            # 获取上下文信息
            import json
            from datetime import datetime, timedelta
            import calendar

            system_prompt = agent_context.get('system_prompt', '')
            agent_prompt = agent_context.get('agent_prompt', '')
            init_parameters = agent_context.get('init_parameters', {})
            conversation_history = agent_context.get('conversation_history', [])
            tool_call_history = agent_context.get('tool_call_history', [])

            # 计算所有时间戳（一次性完成）
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")

            now_timestamp = int(now.timestamp() * 1000)
            today_start = int(datetime(now.year, now.month, now.day, 0, 0, 0).timestamp() * 1000)
            today_end = int(datetime(now.year, now.month, now.day, 23, 59, 59).timestamp() * 1000)

            yesterday_date = now - timedelta(days=1)
            yesterday_start = int(datetime(yesterday_date.year, yesterday_date.month, yesterday_date.day, 0, 0, 0).timestamp() * 1000)
            yesterday_end = int(datetime(yesterday_date.year, yesterday_date.month, yesterday_date.day, 23, 59, 59).timestamp() * 1000)

            # 最近30天（用于QueryPropertyAgg默认时间）
            days_30_ago = int((now - timedelta(days=30)).timestamp() * 1000)

            # 构建简洁的历史文本
            history_text = ""
            if conversation_history:
                recent = conversation_history[-3:]  # 只保留最近3轮
                history_text = "**最近对话**：\n" + "\n".join(recent)

            # 构建工具调用历史文本（包含参数和结果）
            tool_history_text = ""
            last_kb_search_text = ""  # 保存最近一次知识库搜索的关键词
            if tool_call_history:
                tool_history_text = "**最近工具调用**（包含返回结果）：\n"
                for tool_call in tool_call_history[-3:]:  # 最近3次工具调用
                    tool_name = tool_call.get('tool_name', 'unknown')
                    arguments = tool_call.get('arguments', {})
                    result = tool_call.get('result', {})

                    tool_history_text += f"- {tool_name}\n"
                    tool_history_text += f"  参数: {json.dumps(arguments, ensure_ascii=False)}\n"

                    # 显示结果（限制长度，避免提示词过长）
                    result_str = json.dumps(result, ensure_ascii=False)
                    if len(result_str) > 500:
                        result_str = result_str[:500] + "..."
                    tool_history_text += f"  返回: {result_str}\n"

                    # 记录最近的知识库搜索关键词
                    if 'knowledgeService#Search' in tool_name and 'text' in arguments:
                        last_kb_search_text = arguments.get('text', '')

            # 构建系统初始参数文本（突出关键信息）
            params_text = ""
            if init_parameters:
                key_params = []
                if 'deviceId' in init_parameters:
                    key_params.append(f"设备ID: {init_parameters['deviceId']}")
                if 'userId' in init_parameters:
                    key_params.append(f"用户ID: {init_parameters['userId']}")
                # 添加其他关键参数
                for key, value in init_parameters.items():
                    if key not in ['deviceId', 'userId']:
                        key_params.append(f"{key}: {value}")
                params_text = ", ".join(key_params) if key_params else json.dumps(init_parameters, ensure_ascii=False)

            # ═══════════════════════════════════════════════════════════════
            # 统一Prompt：工具选择 + 参数填充
            # ═══════════════════════════════════════════════════════════════

            # 检测是否为追问/补充说明
            follow_up_keywords = ['更具体', '详细', '再说', '补充', '继续', '还有', '进一步', '更多', '展开']
            is_follow_up = any(kw in user_message for kw in follow_up_keywords)

            # 构建上下文提示
            context_hint = ""
            if is_follow_up and last_kb_search_text:
                context_hint = f"\n⚠️ **重要提示**：用户的问题是对之前内容的追问，需要结合上一次的搜索主题进行理解。\n上一次搜索的主题是：「{last_kb_search_text}」\n当前用户的追问：「{user_message}」\n你应该理解为：「对'{last_kb_search_text}'进行更详细的说明」\n"

            prompt = f"""# 工具选择与参数填充专家

## 任务
根据用户问题选择唯一工具并填充参数，返回JSON格式。

## 输入
- **问题**: {user_message}{context_hint}
- **工具**: {tools_text}
- **默认参数**: {params_text if params_text else '无'}
- **上下文**: {history_text if history_text else '无'}{tool_history_text if tool_history_text else ''}

## 时间参考
- 今天: {today_start}~{today_end}
- 昨天: {yesterday_start}~{yesterday_end}
- 近30天: {days_30_ago}~{now_timestamp}

---

## 工具选择决策树

### 第1步：关键词匹配（立即匹配）
| 关键词模式 | 目标工具 |
|-----------|----------|
| 告警+数量/统计 | alarm#Count |
| 物模型/属性列表 | GetMetadata |
| 设备详情/信息 | QueryById |
| 设备数量/统计 | device#Count |
| 平均/最大/最小/聚合 | QueryPropertyAgg |
| 全量/历史数据 | QueryPropertyEach |
| 图片链接(http/https) | ImageDescription |

### 第2步：知识库兜底
**必须同时满足**:
- [ ] 第1步未匹配
- [ ] 问题含"什么是/如何/教程"
- [ ] 有knowledgeService#Search
- [ ] 非闲聊
→ 使用knowledgeService#Search

### 第3步：无匹配
→ 返回 {{"tool": "none", "arguments": {{}}}}

---

## 参数填充策略

### 参数优先级（严格顺序）
1. **最近工具调用**（追问必须复用）
2. **默认参数**（{json.dumps(init_parameters, ensure_ascii=False) if init_parameters else '无'}）
3. **用户明确信息**
4. **上下文推断**

### 特殊规则
- **QueryPropertyAgg**: 无时间→默认近30天
- **QueryPropertyEach**: "全量/历史"→不加时间
- **属性查询**: 须先GetMetadata获取属性ID
- **追问识别**: "更详细/补充/继续"且上一轮是知识库→复用text

---

## 输出格式

**仅返回JSON（无其他内容）**:
```json
{{
  "tool": "工具名",
  "arguments": {{
    "参数名": "参数值"
  }}
}}
```

⚠ **约束**: 工具名、参数名必须完全匹配，时间戳为13位整数"""

            # 调用 LLM
            from langchain_core.messages import HumanMessage

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "🔍 [统一选择+填充] 用户消息=%s | 可用工具数=%d | Prompt长度=%d",
                    user_message,
                    len(tool_instances),
                    len(prompt),
                )

            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("🔍 [LLM返回] 原始内容:\n%s", response_text)

            # 解析响应（新格式：包含tool和arguments）
            # 策略：尝试多种解析方式
            result = None

            # 方法1：尝试从markdown代码块中提取（优先，格式最规范）
            code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response_text)
            if code_block_match:
                try:
                    result = json.loads(code_block_match.group(1))
                    logger.debug("✅ 从markdown代码块解析成功")
                except Exception:
                    pass

            # 方法2：尝试提取完整的JSON对象（支持多层嵌套）
            if not result:
                try:
                    # 使用花括号计数来找到完整的JSON对象
                    start_idx = response_text.find('{')
                    if start_idx != -1:
                        brace_count = 0
                        for i, char in enumerate(response_text[start_idx:], start=start_idx):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    # 找到完整的JSON对象
                                    json_str = response_text[start_idx:i+1]
                                    result = json.loads(json_str)
                                    logger.debug("✅ 成功提取完整JSON对象")
                                    break
                except json.JSONDecodeError as e:
                    logger.debug("⚠️ JSON解析失败: %s", e)

            # 方法3：尝试移除注释后再解析
            if not result:
                try:
                    # 移除 // 注释
                    cleaned_text = re.sub(r'//.*', '', response_text)
                    # 再次使用花括号计数
                    start_idx = cleaned_text.find('{')
                    if start_idx != -1:
                        brace_count = 0
                        for i, char in enumerate(cleaned_text[start_idx:], start=start_idx):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = cleaned_text[start_idx:i+1]
                                    result = json.loads(json_str)
                                    logger.debug("✅ 移除注释后解析成功")
                                    break
                except Exception:
                    pass

            if not result or 'tool' not in result:
                logger.error(f"❌ LLM 返回格式不正确，无法解析: {response_text[:200]}")
                return []

            selected_tool_name = result.get('tool', '')
            filled_arguments = result.get('arguments', {})

            if not selected_tool_name:
                logger.error("❌ 未找到选中的工具")
                return []

            logger.debug("✅ 解析结果: tool=%s, arguments=%s", selected_tool_name, filled_arguments)

            # 检查是否为"无合适工具" - 启用知识库兜底
            if selected_tool_name.lower() == 'none':
                # 如果有知识库工具，使用知识库作为兜底
                if has_knowledge_tool:
                    # 判断是否为娱乐/闲聊（这些不使用知识库）
                    chat_keywords = ['笑话', '唱歌', '你好', '在吗', '谢谢', '再见', '早上好', '晚安']
                    is_chat = any(kw in user_message for kw in chat_keywords)

                    if not is_chat:
                        logger.info("💡 没有精确匹配的工具，使用知识库兜底")
                        # 查找知识库工具
                        knowledge_tool = None
                        for tool in tool_instances:
                            if 'knowledgeService#Search' in getattr(tool, 'name', ''):
                                knowledge_tool = tool
                                break

                        if knowledge_tool:
                            # 知识库兜底，需要填充text参数
                            kb_arguments = filled_arguments if filled_arguments else {"text": user_message}

                            # 【新增】缓存知识库参数
                            kb_tool_name = getattr(knowledge_tool, 'name', 'unknown')
                            self._cached_arguments[kb_tool_name] = kb_arguments

                            return [{
                                "tool": knowledge_tool,
                                "score": 0.8,
                                "reason": "知识库兜底检索",
                                "arguments": kb_arguments  # 【新增】
                            }]

                # 没有知识库或是闲聊，返回 None
                logger.info("⚠️ 没有找到合适的工具")
                return [{
                    "tool": None,
                    "score": 0.0,
                    "reason": "没有合适的工具",
                    "arguments": {}  # 【新增】
                }]

            # 查找对应的工具实例（使用tool_map，支持简化名）
            selected_tool = tool_map.get(selected_tool_name)

            if not selected_tool:
                # 尝试模糊匹配
                for tool in tool_instances:
                    tool_full_name = getattr(tool, 'name', '')
                    # 精确匹配
                    if tool_full_name == selected_tool_name:
                        selected_tool = tool
                        break
                    # 模糊匹配：去掉前缀后匹配
                    if '#' in tool_full_name and tool_full_name.split('#')[-1] == selected_tool_name:
                        selected_tool = tool
                        logger.info(f"💡 模糊匹配成功: {selected_tool_name} → {tool_full_name}")
                        break

            if not selected_tool:
                logger.error(f"❌ 工具 {selected_tool_name} 不存在")
                return [{
                    "tool": None,
                    "score": 0.0,
                    "reason": f"工具 {selected_tool_name} 不存在",
                    "arguments": {}  # 【新增】
                }]

            # 【新增】缓存填充好的参数（供 fill_parameters_with_confirmation 使用）
            tool_full_name = getattr(selected_tool, 'name', 'unknown')
            self._cached_arguments[tool_full_name] = filled_arguments
            logger.info(f"💾 缓存参数: {tool_full_name} -> {filled_arguments}")

            # 返回格式：包含填充好的参数
            selected_tools = [{
                "tool": selected_tool,
                "score": 1.0,
                "reason": "最相关工具",
                "arguments": filled_arguments  # 【新增】填充好的参数
            }]

            logger.info(f"✅ 工具选择+参数填充完成: {selected_tool_name}, 参数: {filled_arguments}")
            return selected_tools

        except Exception as e:
            logger.error(f"❌ 工具选择失败: {e}")
            return []

    async def fill_parameters(
        self,
        parameters: Dict[str, Any],
        user_message: str,
        agent_context: Dict[str, Any],
        tool_name: str = "unknown"
    ) -> Dict[str, Any]:
        """参数填充：AI自动填充工具所需参数

        Args:
            parameters: 参数定义（JSON Schema 格式），包含 properties 和 required 字段
            user_message: 用户消息
            agent_context: Agent上下文（必需），包含：
                - system_prompt: 系统提示词
                - agent_prompt: 智能体提示词
                - init_parameters: 初始化参数
            tool_name: 工具名称（可选，用于日志显示）

        Returns:
            填充的参数字典，格式：
            {
                "param_name1": "value1",
                "param_name2": "value2",
                ...
            }

        示例:
            params = await fill_parameters(
                parameters,
                "查询服务器状态",
                agent_context={
                    "system_prompt": "你是运维助手",
                    "agent_prompt": "帮助管理服务器",
                    "init_parameters": {"user_role": "运维工程师"}
                },
                tool_name="server_monitor"
            )
        """
        logger.info(f"📝 开始参数填充")
        logger.info(f"  工具: {tool_name}")

        # 检查参数定义
        if not parameters or not isinstance(parameters, dict):
            logger.warning(f"⚠️ 参数定义为空或格式错误")
            return {}

        properties = parameters.get('properties', {})
        required_params = parameters.get('required', [])

        if not properties:
            logger.warning(f"⚠️ 工具 {tool_name} 参数定义为空")
            return {}

        # 初始化 LLM
        if not self._ensure_llm_initialized():
            logger.error("❌ LLM 初始化失败，无法进行参数填充")
            return {}

        try:
            # 构建参数描述
            param_descriptions = []
            for param_name, param_def in properties.items():
                param_type = param_def.get('type', 'string')
                param_desc = param_def.get('description', '')
                is_required = "必填" if param_name in required_params else "可选"
                param_descriptions.append(
                    f"  - {param_name} ({param_type}, {is_required}): {param_desc}"
                )

            params_text = "\n".join(param_descriptions)

            # 获取当前日期和时间戳（用于时间关键词转换）
            from datetime import datetime, timedelta
            import calendar

            now = datetime.now()

            # 日期字符串
            today = now.strftime("%Y-%m-%d")
            tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
            yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")

            # 当前时间戳
            now_timestamp = int(now.timestamp() * 1000)

            # 今天 00:00:00 - 23:59:59
            today_start = int(datetime(now.year, now.month, now.day, 0, 0, 0).timestamp() * 1000)
            today_end = int(datetime(now.year, now.month, now.day, 23, 59, 59).timestamp() * 1000)

            # 昨天 00:00:00 - 23:59:59
            yesterday_date = now - timedelta(days=1)
            yesterday_start = int(datetime(yesterday_date.year, yesterday_date.month, yesterday_date.day, 0, 0, 0).timestamp() * 1000)
            yesterday_end = int(datetime(yesterday_date.year, yesterday_date.month, yesterday_date.day, 23, 59, 59).timestamp() * 1000)

            # 明天 00:00:00 - 23:59:59
            tomorrow_date = now + timedelta(days=1)
            tomorrow_start = int(datetime(tomorrow_date.year, tomorrow_date.month, tomorrow_date.day, 0, 0, 0).timestamp() * 1000)
            tomorrow_end = int(datetime(tomorrow_date.year, tomorrow_date.month, tomorrow_date.day, 23, 59, 59).timestamp() * 1000)

            # 本周（周一 00:00:00 到现在）
            week_start_date = now - timedelta(days=now.weekday())
            week_start = int(datetime(week_start_date.year, week_start_date.month, week_start_date.day, 0, 0, 0).timestamp() * 1000)
            week_end = now_timestamp

            # 上周（上周一 00:00:00 到上周日 23:59:59）
            last_week_start_date = now - timedelta(days=now.weekday() + 7)
            last_week_start = int(datetime(last_week_start_date.year, last_week_start_date.month, last_week_start_date.day, 0, 0, 0).timestamp() * 1000)
            last_week_end_date = last_week_start_date + timedelta(days=6)
            last_week_end = int(datetime(last_week_end_date.year, last_week_end_date.month, last_week_end_date.day, 23, 59, 59).timestamp() * 1000)

            # 本月（本月1号 00:00:00 到现在）
            month_start = int(datetime(now.year, now.month, 1, 0, 0, 0).timestamp() * 1000)
            month_end = now_timestamp

            # 上月（上月1号 00:00:00 到上月最后一天 23:59:59）
            if now.month == 1:
                last_month_year = now.year - 1
                last_month = 12
            else:
                last_month_year = now.year
                last_month = now.month - 1
            last_month_start = int(datetime(last_month_year, last_month, 1, 0, 0, 0).timestamp() * 1000)
            last_month_days = calendar.monthrange(last_month_year, last_month)[1]
            last_month_end = int(datetime(last_month_year, last_month, last_month_days, 23, 59, 59).timestamp() * 1000)

            # N天前
            days_7_ago = int((now - timedelta(days=7)).timestamp() * 1000)
            days_30_ago = int((now - timedelta(days=30)).timestamp() * 1000)

            # N小时前
            hours_1_ago = int((now - timedelta(hours=1)).timestamp() * 1000)
            hours_24_ago = int((now - timedelta(hours=24)).timestamp() * 1000)

            # 判断是否为知识库检索工具
            is_knowledge_search = 'knowledgeService#Search' in tool_name

            # 根据工具类型构建提示词
            if is_knowledge_search:
                # 知识库检索
                import json
                system_prompt = agent_context.get('system_prompt', '')
                agent_prompt = agent_context.get('agent_prompt', '')
                init_parameters = agent_context.get('init_parameters', {})
                tool_call_history = agent_context.get('tool_call_history', [])

                # 检测追问并获取上一次搜索关键词
                follow_up_keywords = ['更具体', '详细', '再说', '补充', '继续', '还有', '进一步', '更多', '展开']
                is_follow_up = any(kw in user_message for kw in follow_up_keywords)

                last_search_text = ""
                if tool_call_history:
                    for tool_call in reversed(tool_call_history):  # 从最近往前找
                        if 'knowledgeService#Search' in tool_call.get('tool_name', ''):
                            last_search_text = tool_call.get('arguments', {}).get('text', '')
                            break

                follow_up_hint = ""
                if is_follow_up and last_search_text:
                    follow_up_hint = f"\n\n⚠️ **重要**：用户的问题是对之前搜索内容的追问！\n上一次搜索的主题：「{last_search_text}」\n当前追问：「{user_message}」\n你应该输出上一次的搜索关键词「{last_search_text}」，而不是直接使用当前追问的文字。\n"

                prompt = f"""你是知识库检索专家。请根据系统角色、智能体角色和系统初始参数，从用户消息中提取最适合检索的关键词。

系统角色：
{system_prompt}

系统初始参数：
{json.dumps(init_parameters, ensure_ascii=False, indent=2) if init_parameters else '无'}

用户消息：
{user_message}{follow_up_hint}

工具名称：{tool_name}

参数定义：
{params_text}

请综合考虑系统角色、智能体角色、系统初始参数和具体需求，提取最核心的检索关键词。只输出 JSON 格式：
{{
"text": "核心关键词",
"limit": "数量"
}}

**关键词提取规则**：
1. **追问检测**：检测到追问时，复用上一次搜索关键词
2. **保留核心**：技术术语、专业名词、关键动词
3. **去除冗余**：疑问词、语气词、无关描述
4. **格式要求**：2-5个关键词，空格分隔

只输出 JSON，不要其他内容。"""
            else:
                # 通用工具参数填充
                import json
                system_prompt = agent_context.get('system_prompt', '')
                agent_prompt = agent_context.get('agent_prompt', '')
                init_parameters = agent_context.get('init_parameters', {})
                conversation_history = agent_context.get('conversation_history', [])
                tool_call_history = agent_context.get('tool_call_history', [])

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "🔍 [参数填充调试] system_prompt_len=%d | agent_prompt_len=%d | init_parameters=%s | "
                        "conversation_history_len=%d | tool_call_history_len=%d | agent_prompt_preview=%s",
                        len(system_prompt),
                        len(agent_prompt),
                        init_parameters,
                        len(conversation_history),
                        len(tool_call_history),
                        agent_prompt[:500] if agent_prompt else "空",
                    )

                # 构建历史会话文本
                history_text = ""
                if conversation_history:
                    history_text = "\n\n最近的对话历史：\n" + "\n".join(conversation_history)

                # 构建工具调用历史文本
                tool_history_text = ""
                if tool_call_history:
                    tool_history_text = "\n\n最近的工具调用记录：\n"
                    for tool_call in tool_call_history[-3:]:  # 只取最近3次
                        tool_name = tool_call.get('tool_name', 'unknown')
                        arguments = tool_call.get('arguments', {})
                        result = tool_call.get('result', '')

                        # ⚠️ 改进：根据结果类型智能截断
                        if isinstance(result, dict) or isinstance(result, list):
                            # 结构化数据，保留更多内容
                            result_summary = str(result)[:1000]  # 增加到1000字符
                        else:
                            # 普通文本，适度截断
                            result_summary = str(result)[:500]  # 增加到500字符

                        tool_history_text += f"- 工具: {tool_name}\n"
                        tool_history_text += f"  参数: {json.dumps(arguments, ensure_ascii=False)}\n"
                        tool_history_text += f"  结果: {result_summary}\n"

                # 构建系统初始参数文本（突出关键信息）
                params_text_formatted = ""
                if init_parameters:
                    key_params = []
                    if 'deviceId' in init_parameters:
                        key_params.append(f"设备ID: {init_parameters['deviceId']}")
                    if 'userId' in init_parameters:
                        key_params.append(f"用户ID: {init_parameters['userId']}")
                    # 添加其他关键参数
                    for key, value in init_parameters.items():
                        if key not in ['deviceId', 'userId']:
                            key_params.append(f"{key}: {value}")
                    params_text_formatted = ", ".join(key_params) if key_params else json.dumps(init_parameters, ensure_ascii=False)

                prompt = f"""# 智能参数填充专家系统

## 任务目标
根据工具定义、用户输入和上下文信息，**精确提取并验证参数**，确保类型匹配和格式正确。

## 工具信息分析
**工具名称**: {tool_name}
**参数定义**:
{params_text}

## 上下文数据
**用户消息**: {user_message}
{f"**系统默认参数**: {params_text_formatted}" if params_text_formatted else ""}
**历史对话**: {history_text if history_text else "无"}
**工具调用历史**: {tool_history_text if tool_history_text else "无"}

---

## 参数提取策略（严格优先级）

### 1. 参数复用机制
- **工具调用历史**: 直接复用已成功调用的 deviceId、propertyId 等参数
- **系统默认值**: 优先使用 init_parameters 中的预配置值
- **上下文继承**: 从对话历史中推断省略的主语和对象

### 2. 用户输入解析
- **直接提取**: 明确提到的参数值（设备ID、时间范围、关键词等）
- **代词解析**: "它"、"该设备"等指代词需要根据上下文确定具体对象
- **隐含信息**: 从描述中推断未明确提及的必要参数

### 3. 特殊参数处理规则

#### 时间参数（精确映射）
| 用户表达 | 转换规则 | 输出格式 |
|---------|---------|---------|
| "现在"/"当前" | 立即时间戳 | {now_timestamp} |
| "今天" | 当天范围 | {today_start} ~ {today_end} |
| "昨天" | 昨天范围 | {yesterday_start} ~ {yesterday_end} |
| "最近7天" | 7天范围 | {days_7_ago} ~ {now_timestamp} |
| "最近30天" | 30天范围 | {days_30_ago} ~ {now_timestamp} |

#### 数据类型验证（严格匹配）
```json
{{
  "string": "必须用双引号包围的文本",
  "number": 12345.67,
  "boolean": true,
  "array": ["item1", "item2"],
  "object": {{"key": "value"}}
}}
```

#### JetLinks 特定格式
**时间范围查询**: `{{"filter": {{"timestamp$btw": "开始时间戳,结束时间戳"}}}}`
**模糊匹配**: `{{"filter": {{"name$like": "%关键词%"}}}}`
**集合查询**: `{{"filter": {{"status$in": ["active", "pending"]}}}}`

**聚合查询**: `{{"columns": [{{"property": "属性ID", "agg": "AVG"}}], "query": {{"from": 开始时间, "to": 结束时间}}}}`

---

## 参数验证清单

### 类型一致性检查
- [ ] **字符串参数**: 确保用双引号包围，检查特殊字符转义
- [ ] **数值参数**: 确保不加引号，验证为有效数字
- [ ] **布尔参数**: 只能为 true/false，小写
- [ ] **数组参数**: 确保格式为 [item1, item2]
- [ ] **对象参数**: 确保格式为 {{"key": "value"}}

### 必填参数验证
- [ ] **required 列表中的参数必须全部填充**
- [ ] **如果无法获取必填参数值，不填充该参数（让系统处理）**

### 格式规范验证
- [ ] **时间戳**: 必须为13位数字毫秒格式
- [ ] **JSON格式**: 确保所有括号、引号、逗号正确匹配
- [ ] **特殊字符**: 确保引号、换行符等正确转义

---

## 输出格式标准

**严格输出JSON格式（无任何解释文字）**:
```json
{{
  "参数名": "参数值",
  "数值参数": 123,
  "布尔参数": true,
  "数组参数": ["item1", "item2"],
  "对象参数": {{"key": "value"}}
}}
```

## 错误处理原则

1. **无法确定参数值**: 不填充该参数（不要猜测或编造）
2. **类型不匹配**: 按参数定义转换，无法转换则不填充
3. **格式验证失败**: 修正格式或放弃填充该参数
4. **必填参数缺失**: 不填充（让上层系统处理错误）

---

## 重要约束

⚠ **严格禁止**:
- 输出任何解释性文字或注释
- 猜测不确定的参数值
- 混合数据类型（如数字加引号）

✅ **必须遵守**:
- 参数名与定义完全一致
- 数据类型严格匹配
- JSON格式绝对正确

请根据以上规则分析并填充参数。"""

            # 调用 LLM
            from langchain_core.messages import HumanMessage
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("🧩 参数填充提示词(len=%d): %s", len(prompt), prompt[:500])
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content

            # 解析 JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                logger.warning("⚠️ LLM 返回格式不正确")
                return {}

            params = json.loads(json_match.group())
            logger.info(f"✅ 参数填充完成: {params}")
            return params

        except Exception as e:
            logger.error(f"❌ 参数填充失败: {e}")
            return {}

    async def fill_tool_parameters(
        self,
        tool: Any,
        user_message: str,
        agent_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """便捷方法：从工具实例中提取参数并填充

        Args:
            tool: 工具实例（必须有 parameters 和 name 属性）
            user_message: 用户消息
            agent_context: Agent上下文（必需）

        Returns:
            填充的参数字典
        """
        tool_name = getattr(tool, 'name', 'unknown')
        parameters = getattr(tool, 'parameters', {})

        return await self.fill_parameters(parameters, user_message, agent_context, tool_name)

    async def fill_parameters_with_confirmation(
        self,
        tool: Any,
        user_message: str,
        agent_context: Dict[str, Any],
        pre_filled_arguments: Dict[str, Any] = None  # 【新增】预填充的参数
    ) -> Dict[str, Any]:
        """参数填充并生成 JAIP 协议的 tools.confirm 格式

        【重要】如果提供了 pre_filled_arguments，则直接使用，不再调用 LLM

        Args:
            tool: 工具实例（必须有 name, description, parameters 属性）
            user_message: 用户消息
            agent_context: Agent上下文（必需），包含：
                - system_prompt: 系统提示词
                - agent_prompt: 智能体提示词
                - init_parameters: 初始化参数
            pre_filled_arguments: 【新增】预填充的参数（来自select_tools）

        Returns:
            JAIP 协议的 tools.confirm 消息，格式：
            {
                'jsonrpc': '2.0',
                'method': 'tools.confirm',
                'id': execution_id,
                'params': {
                    'message': '需要调用工具：xxx',
                    'call': {
                        'toolName': 'xxx',
                        'arguments': {...},
                        'executionId': 'xxx'
                    },
                    'command': {
                        'id': 'xxx',
                        'name': 'xxx',
                        'description': 'xxx',
                        'inputs': [...]
                    }
                }
            }
        """
        import uuid

        # 获取工具信息
        tool_name = getattr(tool, 'name', 'unknown')
        tool_description = getattr(tool, 'description', '')
        parameters = getattr(tool, 'parameters', {})

        # 生成执行ID
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"

        # 【修改】参数获取优先级：
        # 1. 显式传入的 pre_filled_arguments（向后兼容）
        # 2. 从缓存中获取（select_tools 已填充）
        # 3. 调用 LLM 填充（降级方案）
        if pre_filled_arguments is not None:
            filled_params = pre_filled_arguments
            logger.info(f"✅ 使用显式传入的参数: {filled_params}")
        elif tool_name in self._cached_arguments:
            filled_params = self._cached_arguments.get(tool_name, {})
            logger.info(f"✅ 使用缓存参数（来自select_tools）: {filled_params}")
            # 使用后清除缓存
            del self._cached_arguments[tool_name]
        else:
            # 兼容旧的调用方式：没有缓存参数时，调用 LLM 填充
            filled_params = await self.fill_parameters(parameters, user_message, agent_context, tool_name)
            logger.info(f"✅ LLM填充参数（降级方案）: {filled_params}")

        # 将 parameters 转换为 inputs 格式
        inputs = self._convert_parameters_to_inputs(parameters)

        # 构建 tools.confirm 格式
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
        # 如果工具是知识库工具
        if 'knowledgeService#Search' == tool_name:
            confirmation['params']['auto'] = True
            pass
        logger.info(f"✅ 生成工具确认消息: {tool_name}, execution_id={execution_id}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("🔍 [工具确认] 完整的 arguments: %s", filled_params)
            logger.debug("🔍 [工具确认] 完整的 confirmation: %s", json.dumps(confirmation, ensure_ascii=False, indent=2))
        return confirmation

    def _convert_parameters_to_inputs(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将 JSON Schema 的 parameters 转换为 inputs 列表格式

        Args:
            parameters: JSON Schema 格式的参数定义
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"},
                        ...
                    },
                    "required": ["city"]
                }

        Returns:
            inputs 列表格式：
                [
                    {
                        "id": "city",
                        "name": "城市",
                        "type": "string",
                        "valueType": {"type": "string"},
                        "description": "城市名称",
                        "required": true
                    },
                    ...
                ]
        """
        if not isinstance(parameters, dict):
            return []

        properties = parameters.get('properties', {})
        required_fields = parameters.get('required', [])

        inputs = []
        for param_id, param_def in properties.items():
            param_type = param_def.get('type', 'string')

            input_item = {
                'id': param_id,
                'name': param_def.get('title', param_id),  # 使用 title 作为显示名称
                'type': param_type,
                'valueType': self._get_value_type(param_id, param_def),  # ← 新增
                'description': param_def.get('description', ''),
                'required': param_id in required_fields
            }

            # 如果有枚举值，添加到 input_item
            if 'enum' in param_def:
                input_item['enum'] = param_def['enum']

            # 如果有默认值，添加到 input_item
            if 'default' in param_def:
                input_item['default'] = param_def['default']

            inputs.append(input_item)

        return inputs

    def _get_value_type(self, param_id: str, param_def: Dict[str, Any]) -> Dict[str, str]:
        """推断参数的 valueType

        优先级：
        1. 如果有 x-valueType 扩展字段，直接使用
        2. 如果有 JSON Schema 的 format 字段，使用它
        3. 基于参数名和描述智能推断
        4. 回退到与 type 相同

        Args:
            param_id: 参数ID
            param_def: 参数定义

        Returns:
            {"type": "string"} 或 {"type": "date"} 等
        """
        param_type = param_def.get('type', 'string')
        description = param_def.get('description', '').lower()

        # 优先级1: 扩展字段 x-valueType
        if 'x-valueType' in param_def:
            return {'type': param_def['x-valueType']}

        # 优先级2: JSON Schema 标准的 format 字段
        if 'format' in param_def:
            format_value = param_def['format']
            # 标准 format 映射
            format_mapping = {
                'date': 'date',
                'date-time': 'datetime',
                'time': 'time',
                'email': 'email',
                'uri': 'url',
                'url': 'url',
                'ipv4': 'ip',
                'ipv6': 'ip'
            }
            value_type = format_mapping.get(format_value, format_value)
            return {'type': value_type}

        # 优先级3: 基于参数名和描述智能推断
        # 日期类型推断
        if param_type == 'string':
            # 参数名包含 date/time
            if 'date' in param_id.lower():
                if 'yyyy-mm-dd' in description or '日期' in param_def.get('description', ''):
                    return {'type': 'date'}
            if 'time' in param_id.lower():
                return {'type': 'time'}
            if 'datetime' in param_id.lower() or 'timestamp' in param_id.lower():
                return {'type': 'datetime'}
            # 描述包含日期格式
            if 'yyyy-mm-dd' in description or 'yyyy/mm/dd' in description:
                return {'type': 'date'}
            if 'hh:mm:ss' in description or '时间' in param_def.get('description', ''):
                return {'type': 'time'}

        # 数字类型推断
        if param_type in ['integer', 'number']:
            return {'type': 'number'}

        # 布尔类型
        if param_type == 'boolean':
            return {'type': 'boolean'}

        # 数组类型
        if param_type == 'array':
            return {'type': 'array'}

        # 对象类型
        if param_type == 'object':
            return {'type': 'object'}

        # 优先级4: 回退到与 type 相同
        return {'type': param_type}

    def set_llm(self, llm):
        """手动设置 LLM（可选）"""
        self.llm = llm
        self.llm_initialized = True
        logger.info("✅ LLM 已手动设置")


# 全局工具选择器实例
tool_selector = ToolSelector()


def init_tool_selector_with_llm(llm):
    """手动初始化 LLM（可选）"""
    global tool_selector
    tool_selector.set_llm(llm)
    logger.info("✅ 全局工具选择器已初始化（手动设置 LLM）")
