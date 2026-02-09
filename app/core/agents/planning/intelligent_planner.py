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
                quick_plan = self._restrict_plan_tools(quick_plan, available_tools)
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
                    cached_plan = self._restrict_plan_tools(cached_plan, available_tools)
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
            llm_plan = self._restrict_plan_tools(llm_plan, available_tools)

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
            fallback_plan = self._restrict_plan_tools(fallback_plan, available_tools)
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

        # ──────────────────────────────────────────────────────────────
        # 可视化大屏（固定多步骤工具链 + 缺失工具显式报出）
        # ──────────────────────────────────────────────────────────────
        if any(k in user_input for k in ("大屏", "可视化")) and tools:
            def _tool_text(t: Any) -> str:
                if isinstance(t, dict):
                    name = str(t.get("name") or t.get("id") or "")
                    desc = str(t.get("description") or "")
                    display = str(t.get("display_name") or t.get("displayName") or "")
                else:
                    name = str(getattr(t, "name", "") or "")
                    desc = str(getattr(t, "description", "") or "")
                    display = str(getattr(t, "display_name", "") or "")
                text = " ".join([name, display, desc]).strip()
                return text.lower()

            def _tool_id(t: Any) -> str:
                if isinstance(t, dict):
                    return str(t.get("name") or t.get("id") or "")
                return str(getattr(t, "name", "") or "")

            def _norm(value: Any) -> str:
                try:
                    text = str(value or "")
                except Exception:
                    text = ""
                return re.sub(r"[^0-9a-zA-Z]+", "", text).lower()

            def _tool_schema_keys(t: Any) -> List[str]:
                schema = None
                if isinstance(t, dict):
                    schema = t.get("parameters")
                else:
                    schema = getattr(t, "parameters", None)
                if not isinstance(schema, dict):
                    return []
                props = schema.get("properties")
                if not isinstance(props, dict):
                    return []
                return [str(k) for k in props.keys() if k]

            def _infer_group_id() -> Optional[str]:
                for t in tools:
                    tid = _tool_id(t)
                    if isinstance(tid, str) and "#" in tid:
                        prefix = tid.rsplit("#", 1)[0].strip()
                        if prefix:
                            return prefix
                return None

            # Heuristic: scope the “大屏 toolchain” to visualization tools only. Otherwise,
            # generic tools like knowledgeService#Search (also has `text`) may be mis-selected
            # as the skeleton tool, causing confusing 2-step plans in “通用工具”.
            visualization_tools: List[Any] = []
            for t in tools:
                tid = _tool_id(t)
                text = _tool_text(t)
                if "visualization" in tid.lower() or "visualization" in text or "可视化" in text:
                    visualization_tools.append(t)

            # No visualization tools registered for this session → do not force the dashboard pipeline.
            if not visualization_tools:
                return None

            def _infer_group_id_from(tool_list: List[Any]) -> Optional[str]:
                for t in tool_list:
                    tid = _tool_id(t)
                    if isinstance(tid, str) and "#" in tid:
                        prefix = tid.rsplit("#", 1)[0].strip()
                        if prefix:
                            return prefix
                return None

            group_id = _infer_group_id_from(visualization_tools) or _infer_group_id()

            # Prefer schema-based role inference (more robust than keyword matching).
            role_candidates: Dict[str, str] = {}
            for t in visualization_tools:
                tid = _tool_id(t)
                if not tid:
                    continue
                keys = set(_tool_schema_keys(t))
                if "text" in keys:
                    role_candidates.setdefault("skeleton", tid)
                    continue
                if "terms" in keys:
                    role_candidates.setdefault("component_info", tid)
                    continue
                if "component" in keys and "id" in keys:
                    role_candidates.setdefault("fill_component", tid)
                    continue
                if keys == {"id"}:
                    role_candidates.setdefault("query_template", tid)
                    continue

            # Keyword fallback when schema is unavailable.
            def _pick(keywords: List[str], used: set) -> Optional[str]:
                best_score = 0
                best_id = None
                for t in visualization_tools:
                    tid = _tool_id(t)
                    if not tid or tid in used:
                        continue
                    text = _tool_text(t)
                    score = 0
                    for kw in keywords:
                        if kw.lower() in text:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_id = tid
                return best_id if best_score > 0 else None

            used: set = set()

            def _placeholder(suffix: str) -> str:
                if group_id:
                    return f"{group_id}#{suffix}"
                return suffix

            required_roles = ("skeleton", "component_info", "fill_component", "query_template")
            has_full_chain = all(role_candidates.get(r) for r in required_roles)

            # Fallback when we don't have the full 4-tool chain registered.
            # Keep the plan executable with current tools, but surface which tools are missing.
            if not has_full_chain:
                # Optional knowledge search step (helps users understand what is missing / how to configure).
                knowledge_tool_id = None
                for t in tools:
                    tid = _tool_id(t)
                    if _norm(tid) == _norm("knowledgeService#Search"):
                        knowledge_tool_id = tid
                        break

                fallback_tools: List[str] = []
                fallback_steps: List[str] = []
                if knowledge_tool_id:
                    fallback_tools.append(knowledge_tool_id)
                    fallback_steps.append("检索知识库")

                component_info_tool = role_candidates.get("component_info")
                if component_info_tool:
                    fallback_tools.append(component_info_tool)
                    fallback_steps.append("查询组件详细信息")

                if not fallback_tools:
                    return None

                unavailable: List[str] = []
                if not role_candidates.get("skeleton"):
                    unavailable.append(_placeholder("GetTemplateInitMetadata"))
                if not role_candidates.get("component_info"):
                    unavailable.append(_placeholder("SearchComponentInfo"))
                if not role_candidates.get("fill_component"):
                    unavailable.append(_placeholder("FillComponent"))
                if not role_candidates.get("query_template"):
                    unavailable.append(_placeholder("QueryGenerateTemplateInfo"))

                task_type = "complex" if len(fallback_tools) >= 2 else "single_tool"
                return {
                    "task_type": task_type,
                    "tools_needed": fallback_tools,
                    "steps": fallback_steps,
                    "reasoning": "当前会话未注册完整的大屏可视化工具链，只能先使用已注册工具进行检索/组件查询；要生成最终大屏JSON请在 session.initialize 的 availableTools 中提供 GetTemplateInitMetadata/FillComponent/QueryGenerateTemplateInfo 等工具。",
                    "confidence": 0.65,
                    "unavailable_tools": unavailable,
                }

            # Ordered pipeline (matches agent system prompt workflow).
            tool_chain: List[Tuple[str, Optional[str], List[str], str]] = [
                (
                    "获取大屏骨架/模板结构",
                    role_candidates.get("skeleton"),
                    ["gettemplateinitmetadata", "initmetadata", "templateinit", "初始化", "骨架", "modulemeta"],
                    _placeholder("GetTemplateInitMetadata"),
                ),
                (
                    "查询平台组件默认信息",
                    role_candidates.get("component_info"),
                    ["searchcomponentinfo", "组件默认", "默认信息", "组件信息", "组件配置"],
                    _placeholder("SearchComponentInfo"),
                ),
                (
                    "填充组件到大屏模板",
                    role_candidates.get("fill_component"),
                    ["fillcomponent", "填充组件", "添加组件"],
                    _placeholder("FillComponent"),
                ),
                (
                    "查询生成的大屏最终内容",
                    role_candidates.get("query_template"),
                    ["querygeneratetemplateinfo", "querygenerate", "生成大屏", "最终", "渲染"],
                    _placeholder("QueryGenerateTemplateInfo"),
                ),
            ]

            ordered: List[str] = []
            ordered_steps: List[str] = []
            for step_name, fixed_tid, kws, fallback_tid in tool_chain:
                picked = fixed_tid
                if not picked:
                    picked = _pick(kws, used)
                if picked:
                    used.add(picked)
                    ordered.append(picked)
                else:
                    # Put a semantic placeholder so the caller can surface missing tools clearly.
                    ordered.append(fallback_tid)
                ordered_steps.append(step_name)

            # Stepwise execution support:
            # When the caller runs one tool per step (e.g. StepwiseCognitiveAgent), we should
            # return the *next* unfinished tool instead of restarting from skeleton each time.
            executed_tools: set[str] = set()
            try:
                user_ctx = context.get("user_context") if isinstance(context, dict) else None
                if not isinstance(user_ctx, dict):
                    user_ctx = {}
                agent_ctx = user_ctx.get("agent_context") if isinstance(user_ctx.get("agent_context"), dict) else user_ctx
                history = agent_ctx.get("tool_call_history") or user_ctx.get("tool_call_history") or []
                if isinstance(history, list):
                    for entry in history:
                        if not isinstance(entry, dict):
                            continue
                        name = entry.get("tool_name") or entry.get("toolName") or entry.get("name")
                        if isinstance(name, str) and name.strip():
                            executed_tools.add(_norm(name))
            except Exception:
                executed_tools = set()

            if executed_tools:
                start_idx = None
                for idx, tid in enumerate(ordered):
                    if _norm(tid) not in executed_tools:
                        start_idx = idx
                        break
                if start_idx is None:
                    return {
                        "task_type": "conversation",
                        "tools_needed": [],
                        "steps": [],
                        "reasoning": "可视化大屏工具链已完成，进入结果汇总阶段",
                        "confidence": 0.9,
                    }
                if start_idx > 0:
                    ordered = ordered[start_idx:]
                    ordered_steps = ordered_steps[start_idx:]

            return {
                "task_type": "complex",
                "tools_needed": ordered,
                "steps": ordered_steps,
                "reasoning": "可视化大屏生成需要固定多步骤工具链（缺失工具将提示重新初始化/注册）",
                "confidence": 0.9 if all(role_candidates.get(r) for r in ("skeleton", "component_info", "fill_component", "query_template")) else 0.7,
            }

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
        from langchain_community.callbacks.manager import get_openai_callback
        from app.core.llm.usage import usage_from_openai_callback
        usage_tracker = context.get("usage_tracker")
        if not usage_tracker:
            agent_context = context.get("agent_context") or {}
            usage_tracker = agent_context.get("usage_tracker")
        try:
            with get_openai_callback() as cb:
                response = await asyncio.wait_for(
                    asyncio.to_thread(planning_llm.invoke, messages),
                    timeout=self.timeout
                )
            result_text = response.content.strip()
            logger.info(f"✅ LLM规划响应成功: {len(result_text)} 字符")
            usage = usage_from_openai_callback(cb, model=self.model, provider="langchain")
            if usage_tracker and usage:
                usage_tracker.add(usage, source="planner")

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
        user_ctx = context.get("user_context") or {}

        # Optional: include agent/system constraints so planner can follow fixed sequences.
        extra_instructions = (
            user_ctx.get("planning_instructions")
            or user_ctx.get("system_prompt")
            or ""
        )
        extra_instructions = str(extra_instructions or "").strip()
        if extra_instructions:
            max_chars = 1800
            if len(extra_instructions) > max_chars:
                extra_instructions = extra_instructions[:max_chars] + "..."

        tool_names = []
        for tool in tools:
            if isinstance(tool, dict):
                name = tool.get("name") or tool.get("id") or tool.get("tool_id")
            else:
                name = getattr(tool, "name", None) or getattr(tool, "id", None) or getattr(tool, "tool_id", None)
            if name:
                tool_names.append(str(name))
        tool_names = list(dict.fromkeys(tool_names))
        tool_names_text = ", ".join(tool_names) if tool_names else "无"

        # 工具描述
        tools_desc = "\n".join([
            f"- {getattr(t, 'name', 'unknown')}: {getattr(t, 'description', '')[:80]}"
            for t in tools[:10]  # 最多10个工具
        ])

        # Skills summary (optional)
        skills = context.get("skills") or []
        has_skill_tool = "skill" in tool_names
        skills_desc = "无"
        if skills and has_skill_tool:
            skills_lines = []
            for item in skills[:10]:
                name = str(item.get("name") or "").strip()
                desc = str(item.get("description") or "").strip()
                if not name:
                    continue
                skills_lines.append(f"- {name}: {desc}")
            if skills_lines:
                skills_desc = "\n".join(skills_lines)

        extra_rules_block = ""
        if extra_instructions:
            extra_rules_block = f"""
额外业务/系统约束（必须严格遵守）：
{extra_instructions}
"""

        return f"""你是智能任务规划专家，一次分析完成所有判断：是否需要工具、选择哪些工具、如何调用。

用户需求将在下一条消息中提供。

可用工具：
{tools_desc}

可用工具名称列表（必须从中选择，名称需完全一致）：
{tool_names_text}

可用技能（当需求匹配时，优先调用 skill 工具加载技能，再执行其他工具）：
{skills_desc}
{extra_rules_block}

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

5. 技能匹配任务（仅当可用技能非空且存在 skill 工具时）：
   - 若用户需求与某个技能描述匹配，优先选择 skill 工具加载该技能
   - 返回：{{"task_type": "single_tool", "tools_needed": ["skill"], "steps": ["加载技能"], "reasoning": "任务匹配技能描述"}}

输出JSON格式：
{{
    "task_type": "conversation|single_tool|complex",
    "tools_needed": ["工具1", "工具2"],
    "steps": ["步骤描述"],
    "reasoning": "选择原因"
}}

只允许从“可用工具名称列表”中选择工具名；如果没有合适工具，返回 conversation。

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

    def _restrict_plan_tools(self, plan: Dict[str, Any], available_tools: List[Any]) -> Dict[str, Any]:
        """
        只保留 available_tools 中存在的工具名，避免规划出现不可用工具。
        """
        if not plan:
            return plan

        tools_needed = [str(t) for t in plan.get("tools_needed", []) if t]
        if not tools_needed:
            return plan

        allowed = set()
        for tool in available_tools or []:
            if isinstance(tool, dict):
                candidates = [tool.get("name"), tool.get("id"), tool.get("tool_id")]
            else:
                candidates = [getattr(tool, "name", None), getattr(tool, "id", None), getattr(tool, "tool_id", None)]
            for candidate in candidates:
                if candidate:
                    allowed.add(str(candidate))

        if not allowed:
            plan["missing_tools"] = tools_needed
            plan["tools_needed"] = []
            plan["steps"] = []
            plan["task_type"] = "conversation"
            if not plan.get("reasoning"):
                plan["reasoning"] = "无可用工具"
            return plan

        filtered = [name for name in tools_needed if name in allowed]
        removed = [name for name in tools_needed if name not in allowed]
        if removed:
            plan["missing_tools"] = removed

        plan["tools_needed"] = filtered

        if not filtered:
            plan["task_type"] = "conversation"
            plan["steps"] = []
            if not plan.get("reasoning"):
                plan["reasoning"] = "无可用工具"
        elif len(filtered) == 1:
            plan["task_type"] = "single_tool"
            if not plan.get("steps"):
                plan["steps"] = [f"使用{filtered[0]}完成任务"]
        else:
            plan["task_type"] = "complex"
            if not plan.get("steps"):
                plan["steps"] = ["分析需求", "执行工具", "整合结果"]

        if removed:
            logger.warning(f"⚠️ 计划包含不可用工具，已过滤: {removed}")

        return plan

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
