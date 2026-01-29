"""
智能计划模型

职责：
1. 统一的工具选择和任务规划
2. 判断任务复杂度（对话/单工具/多工具）
3. 生成执行计划和工具链路
4. 历史计划复用
5. 超时保护和降级机制

特点：
- 独立可测试
- 高度可复用
- 支持多种策略
- 完整的错误处理
"""

import json
import logging
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class IntelligentPlanner:
    """
    智能计划模型 - 专门负责工具选择和任务规划
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化智能计划器

        Args:
            config: 配置字典，包含：
                - model: LLM模型名称
                - api_key: API密钥
                - base_url: API基础URL
                - temperature: 温度参数
                - max_tokens: 最大token数
                - timeout: 超时时间（秒）
        """
        self.model = config.get("model", "glm-4.6")
        self.temperature = config.get("temperature", 0.05)
        self.max_tokens = config.get("max_tokens", 800)
        self.timeout = config.get("timeout", 30.0)

        # API配置
        self.api_key = (config.get("api_key") or
                       os.getenv("OPENAI_API_KEY") or
                       os.getenv("LLM_API_KEY"))
        self.base_url = (config.get("base_url") or
                        os.getenv("OPENAI_API_BASE") or
                        os.getenv("LLM_BASE_URL"))

        # 计划缓存和复用
        self.plan_cache = {}
        self.similarity_threshold = config.get("similarity_threshold", 0.85)

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "fallbacks": 0,
            "timeouts": 0
        }

        logger.info(f"✅ IntelligentPlanner 初始化完成: {self.model}")

    async def generate_plan(
        self,
        user_input: str,
        available_tools: List[Any],
        context: Optional[Dict[str, Any]] = None,
        historical_plans: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        🚀 核心：生成智能执行计划

        Args:
            user_input: 用户输入
            available_tools: 可用工具列表
            context: 上下文信息
            historical_plans: 历史成功计划

        Returns:
            计划结果字典，包含：
            - task_type: 任务类型 ("conversation" | "single_tool" | "complex")
            - tools_needed: 需要的工具列表
            - steps: 执行步骤
            - reasoning: 推理过程
            - confidence: 置信度
            - is_cached: 是否来自缓存
            - execution_time: 生成耗时
        """
        start_time = datetime.now()
        self.stats["total_requests"] += 1

        try:
            # 构建上下文
            planning_context = {
                "tools": available_tools,
                "user_context": context or {},
                "historical_plans": historical_plans or []
            }

            logger.info(f"🎯 开始智能规划: {user_input[:50]}...")
            logger.info(f"   🔧 可用工具数: {len(available_tools)}")
            logger.info(f"   📚 历史计划数: {len(historical_plans or [])}")

            # 第一步：尝试快速规则匹配
            quick_plan = self._try_quick_rules(user_input, planning_context)
            if quick_plan:
                execution_time = (datetime.now() - start_time).total_seconds()
                quick_plan.update({
                    "is_cached": False,
                    "execution_time": execution_time,
                    "strategy": "quick_rules"
                })
                logger.info(f"🚀 使用快速规则生成计划: {quick_plan['task_type']}")
                return quick_plan

            # 第二步：尝试历史计划复用
            if historical_plans:
                cached_plan = self._try_historical_plan(user_input, historical_plans)
                if cached_plan:
                    self.stats["cache_hits"] += 1
                    execution_time = (datetime.now() - start_time).total_seconds()
                    cached_plan.update({
                        "is_cached": True,
                        "execution_time": execution_time,
                        "strategy": "historical_cache"
                    })
                    logger.info(f"🎯 复用历史计划: {cached_plan['task_type']}")
                    return cached_plan

            # 第三步：使用LLM进行智能规划
            self.stats["llm_calls"] += 1
            llm_plan = await self._llm_planning(user_input, planning_context)

            execution_time = (datetime.now() - start_time).total_seconds()
            llm_plan.update({
                "is_cached": False,
                "execution_time": execution_time,
                "strategy": "llm_intelligent"
            })

            logger.info(f"🤖 LLM智能规划完成: {llm_plan['task_type']}")
            return llm_plan

        except Exception as e:
            self.stats["fallbacks"] += 1
            logger.error(f"❌ 智能规划失败: {e}，使用降级策略")

            # 降级到简单规则
            fallback_plan = self._fallback_planning(user_input, planning_context)
            execution_time = (datetime.now() - start_time).total_seconds()
            fallback_plan.update({
                "is_cached": False,
                "execution_time": execution_time,
                "strategy": "fallback"
            })

            return fallback_plan

    def _try_quick_rules(self, user_input: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        快速规则匹配（仅处理极简单场景）
        """
        user_input_lower = user_input.lower()
        tools = context.get("tools", [])

        # 仅处理非常明确的简单搜索
        simple_search_keywords = ['搜索', '查找', '检索', 'search']
        complex_keywords = ['设备', '属性', '历史', '数据', '平均', '最大', '最小', '趋势', '分析']

        has_simple_search = any(keyword in user_input_lower for keyword in simple_search_keywords)
        has_complex_content = any(keyword in user_input_lower for keyword in complex_keywords)

        if has_simple_search and not has_complex_content:
            # 寻找搜索工具
            for tool in tools:
                if any(search_word in tool.name.lower() for search_word in ['search', 'knowledge', '检索', '查询']):
                    return {
                        "task_type": "single_tool",
                        "tools_needed": [tool.name],
                        "steps": [f"搜索: {user_input[:30]}..."],
                        "reasoning": "简单搜索模式",
                        "confidence": 0.9
                    }

        return None

    def _try_historical_plan(self, user_input: str, historical_plans: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        历史计划复用
        """
        # 简单的相似度匹配（实际应用中可以用更复杂的算法）
        for plan_record in historical_plans:
            similarity = self._calculate_similarity(user_input, plan_record.get("user_input", ""))

            if similarity >= self.similarity_threshold:
                plan = plan_record.get("plan", {})
                plan["similarity"] = similarity
                return plan

        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度（简化版）
        """
        # 简单的关键词重叠度计算
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    async def _llm_planning(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM智能规划
        """
        # 构建提示词
        system_prompt = self._build_planning_prompt(context)

        # 转换为LangChain消息格式
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]

        # 初始化LLM
        from langchain_openai import ChatOpenAI
        planning_llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=False,
            request_timeout=self.timeout,
        )

        # 调用LLM（带超时）
        import asyncio
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(planning_llm.invoke, messages),
                timeout=self.timeout
            )
            result_text = response.content.strip()
            logger.info(f"✅ LLM规划响应成功: {len(result_text)} 字符")

        except asyncio.TimeoutError:
            self.stats["timeouts"] += 1
            logger.warning(f"⏰ LLM规划超时 ({self.timeout}s)")
            raise Exception("LLM规划超时")

        # 解析结果
        parsed_result = self._parse_llm_result(result_text)

        # 标准化输出格式
        return self._normalize_plan_result(parsed_result)

    def _build_planning_prompt(self, context: Dict[str, Any]) -> str:
        """
        构建规划提示词
        """
        tools = context.get("tools", [])

        # 工具描述
        tools_desc = "\n".join([
            f"- {getattr(t, 'name', 'unknown')}: {getattr(t, 'description', '')[:80]}"
            for t in tools[:10]  # 最多10个工具
        ])

        return f"""你是智能任务规划专家，一次分析完成所有判断：是否需要工具、选择哪些工具、如何调用。

用户需求将在下一条消息中提供。

可用工具：
{tools_desc}

🎯 任务分析规则：

1. 无需工具（纯对话）：
   - 问候、聊天、常识问答等
   - 返回：{{"task_type": "conversation", "tools_needed": [], "reasoning": "纯对话任务"}}

2. 单工具任务：
   - 简单搜索、查询、计算等
   - 返回：{{"task_type": "single_tool", "tools_needed": ["工具名"], "steps": ["执行步骤"], "reasoning": "原因"}}

3. 复杂任务（需要多个工具）：
   - 设备历史数据查询：GetMetadata → QueryProperty/QueryPropertyAgg
   - 多步骤分析、复合查询等
   - 返回：{{"task_type": "complex", "tools_needed": ["工具1", "工具2"], "steps": ["步骤1", "步骤2"], "reasoning": "原因"}}

⚠️ 重要：设备数据查询必须用两步：
- GetMetadata获取属性ID
- QueryProperty查询实际数据

4. 知识库查询任务：
   - 平台功能介绍、使用方法、技术问题等
   - 关键词：平台、功能、能做什么、如何、怎么、什么是、教程、指南、帮助等
   - 返回：{{"task_type": "single_tool", "tools_needed": ["knowledgeService#Search"], "steps": ["搜索知识库"], "reasoning": "查询平台相关信息"}}

输出JSON格式：
{{
    "task_type": "conversation|single_tool|complex",
    "tools_needed": ["工具1", "工具2"],
    "steps": ["步骤描述"],
    "reasoning": "选择原因"
}}

只输出JSON，不要解释。"""

    def _parse_llm_result(self, result_text: str) -> Dict[str, Any]:
        """
        解析LLM返回结果
        """
        try:
            # 提取JSON内容
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            # 提取JSON对象
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                json_text = json_match.group()
                return json.loads(json_text)
            else:
                logger.warning("⚠️ 未找到JSON格式结果")
                return {"task_type": "conversation", "tools_needed": [], "reasoning": "解析失败"}

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON解析失败: {e}")
            return {"task_type": "conversation", "tools_needed": [], "reasoning": "JSON格式错误"}
        except Exception as e:
            logger.error(f"❌ 结果解析失败: {e}")
            return {"task_type": "conversation", "tools_needed": [], "reasoning": "解析异常"}

    def _normalize_plan_result(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化计划结果格式
        """
        task_type = parsed_result.get("task_type", "conversation")
        tools_needed = parsed_result.get("tools_needed", [])
        steps = parsed_result.get("steps", [])
        reasoning = parsed_result.get("reasoning", "")

        if task_type == "conversation":
            return {
                "task_type": "conversation",
                "tools_needed": [],
                "steps": [],
                "reasoning": reasoning or "纯对话任务，无需工具",
                "confidence": 0.8
            }
        elif task_type == "single_tool":
            return {
                "task_type": "single_tool",
                "tools_needed": tools_needed,
                "steps": steps or [f"使用{tools_needed[0] if tools_needed else '工具'}完成任务"],
                "reasoning": reasoning,
                "confidence": 0.9
            }
        else:  # complex
            return {
                "task_type": "complex",
                "tools_needed": tools_needed,
                "steps": steps or ["分析需求", "执行工具", "整合结果"],
                "reasoning": reasoning,
                "confidence": 0.85
            }

    def _fallback_planning(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        降级规划策略
        """
        tools = context.get("tools", [])
        user_input_lower = user_input.lower()

        # 简单关键词匹配
        search_keywords = ['搜索', '查找', '检索', 'search']
        if any(keyword in user_input_lower for keyword in search_keywords):
            for tool in tools:
                if any(search_word in tool.name.lower() for search_word in ['search', 'knowledge', '检索', '查询']):
                    return {
                        "task_type": "single_tool",
                        "tools_needed": [tool.name],
                        "steps": [f"搜索: {user_input[:30]}..."],
                        "reasoning": "降级搜索模式",
                        "confidence": 0.6
                    }

        return {
            "task_type": "conversation",
            "tools_needed": [],
            "steps": [],
            "reasoning": "降级到对话模式",
            "confidence": 0.5
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        """
        cache_hit_rate = (self.stats["cache_hits"] / max(self.stats["total_requests"], 1)) * 100
        fallback_rate = (self.stats["fallbacks"] / max(self.stats["total_requests"], 1)) * 100
        timeout_rate = (self.stats["timeouts"] / max(self.stats["total_requests"], 1)) * 100

        return {
            **self.stats,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "fallback_rate": f"{fallback_rate:.1f}%",
            "timeout_rate": f"{timeout_rate:.1f}%"
        }

    def reset_statistics(self):
        """
        重置统计信息
        """
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "fallbacks": 0,
            "timeouts": 0
        }
        logger.info("📊 统计信息已重置")


# 全局计划器实例（可选）
_global_planner: Optional[IntelligentPlanner] = None


def get_global_planner(config: Optional[Dict[str, Any]] = None) -> IntelligentPlanner:
    """
    获取全局计划器实例
    """
    global _global_planner

    if _global_planner is None:
        if config is None:
            config = {
                "model": os.getenv("LLM_MODEL", "glm-4.6"),
                "temperature": 0.05,
                "max_tokens": 800,
                "timeout": 30.0
            }

        _global_planner = IntelligentPlanner(config)

    return _global_planner


def reset_global_planner():
    """
    重置全局计划器
    """
    global _global_planner
    _global_planner = None
